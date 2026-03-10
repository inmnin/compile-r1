#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import glob
import json
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from humaneval_format import extract_answer_code, extract_last_action

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "python_sandbox"
import sys

if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from sandbox import run_humaneval_in_sandbox  # noqa: E402

ASSERT_CALL_RE = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
DEF_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SFT checkpoints on cold-start val set with compile-style parse/execute backfill rollout."
    )
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("/mnt/workspace/jkh/LLaMA-Factory/data/compile_r1_cold_start_val300_result_masked_alpaca.jsonl"),
    )
    parser.add_argument(
        "--source-train-glob",
        type=str,
        default="/mnt/workspace/jkh/slime/examples/compile-r1/data/train/data/*.parquet",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/mnt/workspace/jkh/slime/examples/compile-r1/train_log/sft_ckpt_eval_rollout_val300_result_masked.json"),
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=7000)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--save-details", action="store_true")
    return parser.parse_args()


def infer_entry_point(question: str, test_cases: list[str]) -> str:
    for case in test_cases:
        m = ASSERT_CALL_RE.search(case)
        if m:
            return m.group(1)
    m = DEF_RE.search(question or "")
    if m:
        return m.group(1)
    return "solution"


def build_ground_truth_map(source_glob: str) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for fp in sorted(glob.glob(source_glob)):
        table = pq.read_table(fp)
        for row in table.to_pylist():
            task_id = str(row.get("id") or "").strip()
            if not task_id:
                continue
            question = str(row.get("question") or "").strip()
            raw_cases = row.get("test_cases") or []
            test_cases = [str(x).strip() for x in raw_cases if str(x).strip()]
            if not question or not test_cases:
                continue
            mapping[task_id] = {
                "task_id": task_id,
                "prompt": question,
                "test": "\n".join(test_cases),
                "test_cases": test_cases,
                "entry_point": infer_entry_point(question, test_cases),
            }
    return mapping


def load_val_samples(val_jsonl: Path, gt_map: dict[str, dict], max_samples: int) -> list[dict]:
    rows: list[dict] = []
    with val_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_id = str(obj.get("id") or "").strip()
            if task_id not in gt_map:
                continue
            rows.append({"alpaca": obj, "ground_truth": gt_map[task_id]})
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def list_checkpoints(ckpt_root: Path) -> list[Path]:
    ckpts = []
    for item in ckpt_root.glob("checkpoint-*"):
        if not item.is_dir():
            continue
        suffix = item.name.split("-")[-1]
        if suffix.isdigit():
            ckpts.append((int(suffix), item))
    ckpts.sort(key=lambda x: x[0])
    return [p for _, p in ckpts]


def normalize_case_for_candidate(case: str, entry_point: str) -> str:
    text = str(case).strip()
    if not text:
        return text
    if entry_point:
        text = re.sub(rf"\b{re.escape(entry_point)}\s*\(", "candidate(", text)
    return text


def single_case_test_code(case: str, entry_point: str) -> str:
    normalized = normalize_case_for_candidate(case, entry_point)
    lines = [ln.rstrip() for ln in normalized.splitlines() if ln.strip()]
    if not lines:
        return "def check(candidate):\n    assert True"
    body = "\n".join(f"    {ln}" for ln in lines)
    return f"def check(candidate):\n{body}"


def evaluate_answer_case_pass_rate(answer_code: str, ground_truth: dict, timeout_seconds: int, memory_mb: int) -> tuple[float, int, int]:
    cases = list(ground_truth.get("test_cases") or [])
    prompt = str(ground_truth.get("prompt") or "")
    entry_point = str(ground_truth.get("entry_point") or "")
    if not cases or not prompt or not entry_point:
        return 0.0, 0, 0

    passed = 0
    total = 0
    for case in cases:
        case_test = single_case_test_code(str(case), entry_point)
        result = run_humaneval_in_sandbox(
            code=answer_code,
            prompt=prompt,
            test=case_test,
            entry_point=entry_point,
            timeout_seconds=timeout_seconds,
            memory_mb=memory_mb,
        )
        total += 1
        if result.get("passed") is True:
            passed += 1
    if total == 0:
        return 0.0, 0, 0
    return passed / total, passed, total


def build_prompt_text(tokenizer, alpaca_obj: dict) -> str:
    instruction = str(alpaca_obj.get("instruction") or "").strip()
    user_input = str(alpaca_obj.get("input") or "").strip()
    if user_input:
        user_text = f"{instruction}\n\n{user_input}"
    else:
        user_text = instruction
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def rollout_generate(
    llm: LLM,
    tokenizer,
    prompt_text: str,
    ground_truth: dict,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_turns: int,
    timeout_seconds: int,
    memory_mb: int,
) -> tuple[str, int, bool]:
    response = ""
    tool_call_count = 0
    truncated = False
    remaining = max_new_tokens

    for _ in range(max_turns):
        if remaining <= 0:
            truncated = True
            break

        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=remaining,
            stop=["</code>", "</answer>"],
            include_stop_str_in_output=True,
        )
        out = llm.generate([prompt_text + response], sampling, use_tqdm=False)[0]
        cur = out.outputs[0].text if out.outputs else ""
        if not cur:
            truncated = True
            break

        response += cur
        cur_token_ids = tokenizer(cur, add_special_tokens=False)["input_ids"]
        remaining -= len(cur_token_ids)

        action, content = extract_last_action(cur)
        if action == "code":
            if not content.strip():
                tool_result = {
                    "passed": False,
                    "status": "invalid_code",
                    "error": "Empty <code> block",
                    "stdout": "",
                    "stderr": "",
                }
            else:
                tool_result = run_humaneval_in_sandbox(
                    code=content,
                    prompt=str(ground_truth.get("prompt", "")),
                    test=str(ground_truth.get("test", "")),
                    entry_point=str(ground_truth.get("entry_point", "")),
                    timeout_seconds=timeout_seconds,
                    memory_mb=memory_mb,
                )
            response += f"\n\n<result>{json.dumps(tool_result, ensure_ascii=False)}</result>\n\n"
            tool_call_count += 1
            continue

        if action == "answer":
            return response, tool_call_count, truncated

        # Keep behavior aligned with training rollout when model emits invalid action.
        response += (
            "\nMy previous action is invalid. "
            "I must output either <code>...</code> to execute Python code or "
            "<answer>...</answer> to provide final code.\n"
        )

    return response, tool_call_count, truncated


def eval_one_checkpoint(ckpt: Path, samples: list[dict], args: argparse.Namespace) -> dict:
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    llm = LLM(
        model=str(ckpt),
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )

    solved = 0
    sum_case_pass_rate = 0.0
    sum_tool_calls = 0
    no_answer = 0
    truncated = 0
    details = []

    for sample in samples:
        alpaca_obj = sample["alpaca"]
        ground_truth = sample["ground_truth"]
        prompt_text = build_prompt_text(tokenizer, alpaca_obj)
        response, tool_calls, was_truncated = rollout_generate(
            llm=llm,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            ground_truth=ground_truth,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_turns=args.max_turns,
            timeout_seconds=args.timeout_seconds,
            memory_mb=args.memory_mb,
        )

        answer_code = extract_answer_code(response or "")
        if not answer_code:
            no_answer += 1
            case_pass_rate = 0.0
            passed_cases = 0
            total_cases = len(ground_truth.get("test_cases") or [])
        else:
            case_pass_rate, passed_cases, total_cases = evaluate_answer_case_pass_rate(
                answer_code=answer_code,
                ground_truth=ground_truth,
                timeout_seconds=args.timeout_seconds,
                memory_mb=args.memory_mb,
            )

        if passed_cases == total_cases and total_cases > 0:
            solved += 1
        sum_case_pass_rate += case_pass_rate
        sum_tool_calls += tool_calls
        if was_truncated:
            truncated += 1

        if args.save_details:
            details.append(
                {
                    "id": alpaca_obj.get("id"),
                    "case_pass_rate": round(case_pass_rate, 6),
                    "passed_cases": int(passed_cases),
                    "total_cases": int(total_cases),
                    "tool_call_count": int(tool_calls),
                    "truncated": bool(was_truncated),
                    "has_answer": bool(answer_code),
                }
            )

    total = len(samples)
    result = {
        "checkpoint": str(ckpt),
        "total": total,
        "full_pass_acc": round((solved / total) if total else 0.0, 6),
        "avg_case_pass_rate": round((sum_case_pass_rate / total) if total else 0.0, 6),
        "avg_tool_call_count": round((sum_tool_calls / total) if total else 0.0, 6),
        "no_answer": no_answer,
        "truncated": truncated,
        "elapsed_sec": round(time.time() - started, 2),
    }
    if args.save_details:
        result["details"] = details

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main() -> None:
    args = parse_args()
    gt_map = build_ground_truth_map(args.source_train_glob)
    samples = load_val_samples(args.val_jsonl, gt_map, args.max_samples)
    if not samples:
        raise RuntimeError("No validation samples loaded. Check val-jsonl or source mapping.")

    ckpts = list_checkpoints(args.ckpt_root)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* found under: {args.ckpt_root}")

    results = []
    for ckpt in ckpts:
        result = eval_one_checkpoint(ckpt, samples, args)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    best = max(results, key=lambda x: x["avg_case_pass_rate"])
    summary = {
        "best_checkpoint": best["checkpoint"],
        "best_avg_case_pass_rate": best["avg_case_pass_rate"],
        "best_full_pass_acc": best["full_pass_acc"],
        "num_checkpoints": len(results),
        "num_samples": len(samples),
        "ranking_metric": "avg_case_pass_rate",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.output_json}")
    print(
        f"best_checkpoint={summary['best_checkpoint']} "
        f"best_avg_case_pass_rate={summary['best_avg_case_pass_rate']} "
        f"best_full_pass_acc={summary['best_full_pass_acc']}"
    )


if __name__ == "__main__":
    main()
