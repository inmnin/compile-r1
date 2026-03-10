#!/usr/bin/env python3

import argparse
import gc
import json
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Evaluate SFT checkpoints by execution accuracy on compile-r1 eval set.")
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=Path("/mnt/workspace/jkh/slime/examples/compile-r1/data/test_answer_rl.parquet"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/mnt/workspace/jkh/slime/examples/compile-r1/train_log/sft_ckpt_eval_accuracy.json"),
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    parser.add_argument("--memory-mb", type=int, default=1024)
    return parser.parse_args()


def load_samples(eval_data: Path, max_samples: int) -> list[dict]:
    table = pq.read_table(eval_data)
    rows = table.to_pylist()
    if max_samples > 0:
        rows = rows[:max_samples]
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


def build_prompt_text(tokenizer, prompt_obj) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, list):
        return tokenizer.apply_chat_template(prompt_obj, tokenize=False, add_generation_prompt=True)
    return str(prompt_obj)


def eval_one_checkpoint(ckpt: Path, samples: list[dict], args: argparse.Namespace) -> dict:
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)

    prompts = [build_prompt_text(tokenizer, row.get("prompt")) for row in samples]
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    llm = LLM(
        model=str(ckpt),
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )

    outputs = llm.generate(prompts, sampling)

    solved = 0
    no_answer = 0
    runtime_errors = 0
    for row, out in zip(samples, outputs):
        response = out.outputs[0].text if out.outputs else ""
        answer_code = extract_answer_code(response or "")
        if not answer_code:
            no_answer += 1
            continue

        answer = row.get("answer") or {}
        ground_truth = answer.get("ground_truth") if isinstance(answer, dict) else {}
        if not isinstance(ground_truth, dict):
            ground_truth = {}

        result = run_humaneval_in_sandbox(
            code=answer_code,
            prompt=str(ground_truth.get("prompt", "")),
            test=str(ground_truth.get("test", "")),
            entry_point=str(ground_truth.get("entry_point", "")),
            timeout_seconds=args.timeout_seconds,
            memory_mb=args.memory_mb,
        )
        if result.get("passed") is True:
            solved += 1
        else:
            runtime_errors += 1

    total = len(samples)
    acc = (solved / total) if total else 0.0
    elapsed = time.time() - started

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "checkpoint": str(ckpt),
        "total": total,
        "solved": solved,
        "acc": round(acc, 6),
        "no_answer": no_answer,
        "runtime_errors": runtime_errors,
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> None:
    args = parse_args()
    samples = load_samples(args.eval_data, args.max_samples)
    ckpts = list_checkpoints(args.ckpt_root)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* found under: {args.ckpt_root}")

    results = []
    for ckpt in ckpts:
        result = eval_one_checkpoint(ckpt, samples, args)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    best = max(results, key=lambda x: x["acc"])
    summary = {
        "best_checkpoint": best["checkpoint"],
        "best_acc": best["acc"],
        "num_checkpoints": len(results),
        "num_samples": len(samples),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.output_json}")
    print(f"best_checkpoint={best['checkpoint']} best_acc={best['acc']}")


if __name__ == "__main__":
    main()
