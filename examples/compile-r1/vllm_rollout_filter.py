#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from humaneval_format import extract_answer_code

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "python_sandbox"
import sys

if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from sandbox import run_humaneval_in_sandbox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM rollout + reward scoring + length filtering for compile-r1")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--reward-workers", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=12)
    parser.add_argument("--memory-mb", type=int, default=1536)

    parser.add_argument("--quantile-positive", type=float, default=0.995)
    parser.add_argument("--quantile-fallback", type=float, default=0.99)
    parser.add_argument("--round-to", type=int, default=64)
    parser.add_argument("--min-threshold", type=int, default=1024)
    parser.add_argument("--max-threshold", type=int, default=7000)
    parser.add_argument("--keep-positive-only", action="store_true")

    parser.add_argument("--drain-multiplier", type=int, default=4)
    return parser.parse_args()


def _to_float_reward(reward: Any) -> float:
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        for key in ("reward", "total", "score", "value"):
            if key in reward and isinstance(reward[key], (int, float)):
                return float(reward[key])
        for value in reward.values():
            if isinstance(value, (int, float)):
                return float(value)
    return 0.0


def _stats(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def _resolve_label(label: Any) -> tuple[str, dict[str, Any]]:
    target_answer = ""
    ground_truth: dict[str, Any] = {}

    if isinstance(label, dict):
        if isinstance(label.get("ground_truth"), dict):
            ground_truth = label["ground_truth"]

        if isinstance(label.get("target"), str):
            target_answer = label["target"]
        elif isinstance(label.get("answer"), str):
            target_answer = label["answer"]
        elif isinstance(label.get("canonical_solution"), str):
            target_answer = label["canonical_solution"]

        if not target_answer and isinstance(ground_truth.get("canonical_solution"), str):
            target_answer = ground_truth["canonical_solution"]
    elif isinstance(label, str):
        target_answer = label

    return target_answer.strip(), ground_truth


def _extract_test_cases(ground_truth: dict[str, Any]) -> list[str]:
    raw_cases = ground_truth.get("test_cases")
    cases: list[str] = []

    if isinstance(raw_cases, list):
        for item in raw_cases:
            text = str(item).strip()
            if text:
                cases.append(text)
    elif isinstance(raw_cases, str):
        for line in raw_cases.splitlines():
            text = line.strip()
            if text:
                cases.append(text)

    if not cases:
        test_code = str(ground_truth.get("test", "") or "")
        for line in test_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert "):
                cases.append(stripped)

    return cases


def _normalize_case_for_candidate(case: str, entry_point: str) -> str:
    text = str(case).strip()
    if not text:
        return text
    if entry_point:
        text = text.replace(f"{entry_point}(", "candidate(")
    return text


def _single_case_test_code(case: str, entry_point: str) -> str:
    normalized = _normalize_case_for_candidate(case, entry_point)
    lines = [ln.rstrip() for ln in normalized.splitlines() if ln.strip()]
    if not lines:
        return "def check(candidate):\n    assert True"
    body = "\n".join(f"    {ln}" for ln in lines)
    return f"def check(candidate):\n{body}"


def _run_one_case(answer_code: str, ground_truth: dict[str, Any], timeout_seconds: int, memory_mb: int) -> bool:
    result = run_humaneval_in_sandbox(
        code=answer_code,
        prompt=str(ground_truth.get("prompt", "")),
        test=str(ground_truth.get("test", "")),
        entry_point=str(ground_truth.get("entry_point", "")),
        timeout_seconds=timeout_seconds,
        memory_mb=memory_mb,
    )
    return bool(result.get("passed") is True)


def _score_response(response_text: str, label: Any, timeout_seconds: int, memory_mb: int) -> dict[str, Any]:
    answer_code = extract_answer_code(response_text or "")
    if not answer_code:
        return {"reward": 0.0, "passed_cases": 0, "total_cases": 0}

    _target_answer, ground_truth = _resolve_label(label)
    if not isinstance(ground_truth, dict):
        ground_truth = {}

    entry_point = str(ground_truth.get("entry_point") or "").strip()
    prompt = str(ground_truth.get("prompt") or "").strip()
    cases = _extract_test_cases(ground_truth)

    if entry_point and prompt and cases:
        passed = 0
        total = len(cases)
        for case in cases:
            gt_case = dict(ground_truth)
            gt_case["test"] = _single_case_test_code(case, entry_point)
            try:
                if _run_one_case(answer_code, gt_case, timeout_seconds, memory_mb):
                    passed += 1
            except Exception:
                pass
        reward = float(passed / total) if total > 0 else 0.0
        return {"reward": reward, "passed_cases": int(passed), "total_cases": int(total)}

    if all(k in ground_truth for k in ("prompt", "test", "entry_point")):
        try:
            passed = 1 if _run_one_case(answer_code, ground_truth, timeout_seconds, memory_mb) else 0
        except Exception:
            passed = 0
        return {"reward": float(passed), "passed_cases": int(passed), "total_cases": 1}

    return {"reward": 0.0, "passed_cases": 0, "total_cases": 0}


def _choose_threshold(
    lengths: np.ndarray,
    rewards: np.ndarray,
    max_threshold: int,
    min_threshold: int,
    quantile_positive: float,
    quantile_fallback: float,
    round_to: int,
) -> int:
    pos = lengths[rewards > 0]
    base = pos if pos.size > 0 else lengths
    q = quantile_positive if pos.size > 0 else quantile_fallback
    suggested = int(math.ceil(float(np.quantile(base, q)) / round_to) * round_to)
    suggested = max(suggested, min_threshold)
    suggested = min(suggested, max_threshold)
    return suggested


def _dataset_size(parquet_path: Path) -> int:
    return pq.ParquetFile(parquet_path).metadata.num_rows


def _write_filtered_parquet(src: Path, dst: Path, keep_index_set: set[int]) -> int:
    pf = pq.ParquetFile(src)
    writer = None
    row_idx = 0
    kept = 0

    try:
        for batch in pf.iter_batches(batch_size=2048):
            batch_dict = batch.to_pydict()
            n = len(next(iter(batch_dict.values()))) if batch_dict else 0
            if n == 0:
                continue

            keep_pos = [i for i in range(n) if (row_idx + i) in keep_index_set]
            if keep_pos:
                out_dict = {k: [v[i] for i in keep_pos] for k, v in batch_dict.items()}
                table = pa.Table.from_pydict(out_dict)
                if writer is None:
                    writer = pq.ParquetWriter(dst, table.schema)
                writer.write_table(table)
                kept += len(keep_pos)
            row_idx += n
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(pa.table({}), dst)
    return kept


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.output_dir / f"{args.dataset_name}_rollout_metrics.parquet"
    summary_path = args.output_dir / f"{args.dataset_name}_summary.json"
    filtered_path = args.output_dir / f"{args.dataset_name}_filtered.parquet"
    keep_idx_path = args.output_dir / f"{args.dataset_name}_keep_indices.parquet"

    total_rows = _dataset_size(args.data)
    target_rows = min(total_rows, args.max_samples) if args.max_samples > 0 else total_rows

    print(f"[vllm-rollout] model={args.model}")
    print(f"[vllm-rollout] data={args.data}, total_rows={total_rows}, target_rows={target_rows}")
    print(f"[vllm-rollout] tp={args.tensor_parallel_size}, batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    records: list[dict[str, Any]] = []
    pending: list[tuple[int, int, int, Future]] = []

    drain_cap = max(args.batch_size * args.drain_multiplier, args.batch_size)

    def flush_done(force: bool = False) -> None:
        nonlocal pending
        if not pending:
            return
        if not force and len(pending) < drain_cap:
            return

        futures = [item[3] for item in pending]
        done, _ = wait(futures, timeout=0 if not force else None, return_when=FIRST_COMPLETED if not force else None)
        if force:
            done = set(futures)

        if not done:
            return

        keep_pending: list[tuple[int, int, int, Future]] = []
        for idx, resp_len, truncated, fut in pending:
            if fut in done:
                result = fut.result()
                records.append(
                    {
                        "index": int(idx),
                        "reward": float(_to_float_reward(result.get("reward", 0.0))),
                        "response_length": int(resp_len),
                        "truncated": int(truncated),
                        "passed_cases": int(result.get("passed_cases", 0)),
                        "total_cases": int(result.get("total_cases", 0)),
                    }
                )
            else:
                keep_pending.append((idx, resp_len, truncated, fut))
        pending = keep_pending

    pf = pq.ParquetFile(args.data)
    seen = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.reward_workers) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=["prompt", "answer"]):
            if seen >= target_rows:
                break

            data = batch.to_pydict()
            prompts = data.get("prompt", [])
            labels = data.get("answer", [])
            n = len(prompts)
            if n == 0:
                continue

            if seen + n > target_rows:
                cut = target_rows - seen
                prompts = prompts[:cut]
                labels = labels[:cut]
                n = cut

            prompts_text = []
            for p in prompts:
                if isinstance(p, str):
                    prompts_text.append(p)
                elif isinstance(p, list):
                    prompts_text.append(tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True))
                else:
                    prompts_text.append(str(p))

            outputs = llm.generate(prompts_text, sampling, use_tqdm=False)
            for i, out in enumerate(outputs):
                output = out.outputs[0] if out.outputs else None
                response_text = output.text if output is not None else ""
                token_ids = output.token_ids if output is not None else []
                finish_reason = str(output.finish_reason) if output is not None else ""
                truncated = int(finish_reason == "length")
                resp_len = len(token_ids)

                fut = pool.submit(_score_response, response_text, labels[i], args.timeout_seconds, args.memory_mb)
                pending.append((seen + i, resp_len, truncated, fut))

            seen += n
            flush_done(force=False)

            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[vllm-rollout] progress={seen}/{target_rows} ({seen/target_rows:.2%}) "
                f"pending_rewards={len(pending)} speed={seen/elapsed:.2f} samples/s",
                flush=True,
            )

        while pending:
            flush_done(force=True)
            print(f"[vllm-rollout] draining rewards... remaining={len(pending)}", flush=True)

    if not records:
        raise RuntimeError("No rollout records produced.")

    records.sort(key=lambda x: x["index"])
    metrics_table = pa.Table.from_pylist(records)
    pq.write_table(metrics_table, metrics_path)

    rewards = np.asarray([row["reward"] for row in records], dtype=np.float64)
    lengths = np.asarray([row["response_length"] for row in records], dtype=np.int64)
    truncated = np.asarray([row["truncated"] for row in records], dtype=np.int64)

    threshold = _choose_threshold(
        lengths=lengths,
        rewards=rewards,
        max_threshold=args.max_threshold,
        min_threshold=args.min_threshold,
        quantile_positive=args.quantile_positive,
        quantile_fallback=args.quantile_fallback,
        round_to=args.round_to,
    )

    keep_mask = lengths <= threshold
    if args.keep_positive_only:
        keep_mask = np.logical_and(keep_mask, rewards > 0)

    keep_indices = np.asarray([row["index"] for row in records], dtype=np.int64)[keep_mask]
    keep_idx_set = set(int(i) for i in keep_indices.tolist())

    keep_idx_table = pa.table({"index": keep_indices})
    pq.write_table(keep_idx_table, keep_idx_path)

    kept_rows = _write_filtered_parquet(args.data, filtered_path, keep_idx_set)

    positive_mask = rewards > 0
    pos_total = int(max(1, positive_mask.sum()))
    pos_kept = int(np.logical_and(positive_mask, keep_mask).sum())

    summary = {
        "dataset": args.dataset_name,
        "model": str(args.model),
        "data": str(args.data),
        "rows_scored": int(len(records)),
        "reward_stats": _stats(rewards),
        "response_length_stats": _stats(lengths),
        "truncated_ratio": float(truncated.mean()) if truncated.size else 0.0,
        "positive_reward_ratio": float(positive_mask.mean()) if positive_mask.size else 0.0,
        "suggested_threshold": int(threshold),
        "retention": {
            "kept_ratio_all": float(keep_mask.mean()) if keep_mask.size else 0.0,
            "kept_ratio_positive_reward": float(pos_kept / pos_total),
            "kept_rows": int(kept_rows),
            "dropped_rows": int(len(records) - kept_rows),
        },
        "paths": {
            "metrics_parquet": str(metrics_path),
            "summary_json": str(summary_path),
            "filtered_parquet": str(filtered_path),
            "keep_indices_parquet": str(keep_idx_path),
        },
        "args": vars(args),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
