#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

STRICT_CODE_RULE = (
    "Important formatting rule: when you need code execution, place only the executable code inside <code></code>. "
    "Inside <code></code>, include runnable Python code only. "
    "Do NOT include explanations, prose, JSON, XML/HTML tags, markdown fences (```), placeholders, or any non-code text. "
    "If code execution is not needed, do not output <code></code>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM inference on compile-r1 train set and save full trajectories.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="/mnt/workspace/jkh/slime/examples/compile-r1/data/train/data/*.parquet",
        help="Parquet shard glob for training prompts.",
    )
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint/model directory.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL with all trajectories.")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-field", type=str, default="prompt")
    parser.add_argument("--id-field", type=str, default="id")
    return parser.parse_args()


def _prompt_from_obj(tokenizer: AutoTokenizer, prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return f"{prompt_obj}\n\n{STRICT_CODE_RULE}"

    if isinstance(prompt_obj, list):
        messages: list[dict[str, Any]] = []
        for item in prompt_obj:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = item.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": "user", "content": str(item)})

        if messages and messages[-1].get("role") == "user":
            messages[-1]["content"] = f"{messages[-1].get('content', '')}\n\n{STRICT_CODE_RULE}"
        else:
            messages.append({"role": "user", "content": STRICT_CODE_RULE})

        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"{str(prompt_obj)}\n\n{STRICT_CODE_RULE}"


def _iter_batches(files: list[Path], prompt_field: str, id_field: str, batch_size: int):
    for file_path in files:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[prompt_field, id_field]):
            data = batch.to_pydict()
            prompts = data.get(prompt_field, [])
            ids = data.get(id_field, [None] * len(prompts))
            yield file_path, prompts, ids


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(Path(p) for p in __import__("glob").glob(args.input_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched: {args.input_glob}")

    total_rows = 0
    for fp in parquet_files:
        total_rows += pq.ParquetFile(fp).metadata.num_rows
    target_rows = min(total_rows, args.max_samples) if args.max_samples > 0 else total_rows

    print(f"[vllm-infer] model={args.model}")
    print(f"[vllm-infer] files={len(parquet_files)} total_rows={total_rows} target_rows={target_rows}")
    print(
        f"[vllm-infer] tp={args.tensor_parallel_size} batch_size={args.batch_size} "
        f"gpu_mem_util={args.gpu_memory_utilization} cpu_offload_gb={args.cpu_offload_gb} "
        f"dtype={args.dtype} max_model_len={args.max_model_len} max_new_tokens={args.max_new_tokens}"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        max_model_len=args.max_model_len,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    started = time.time()
    written = 0
    global_idx = 0

    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for src_file, raw_prompts, raw_ids in _iter_batches(
            parquet_files,
            prompt_field=args.prompt_field,
            id_field=args.id_field,
            batch_size=args.batch_size,
        ):
            if written >= target_rows:
                break

            if not raw_prompts:
                continue

            remain = target_rows - written
            if len(raw_prompts) > remain:
                raw_prompts = raw_prompts[:remain]
                raw_ids = raw_ids[:remain]

            prompt_texts = [_prompt_from_obj(tokenizer, p) for p in raw_prompts]
            outputs = llm.generate(prompt_texts, sampling, use_tqdm=False)

            for i, out in enumerate(outputs):
                output = out.outputs[0] if out.outputs else None
                response_text = output.text if output is not None else ""
                finish_reason = str(output.finish_reason) if output is not None else ""
                token_ids = output.token_ids if output is not None else []

                rec = {
                    "global_index": int(global_idx),
                    "source_file": str(src_file),
                    "sample_id": raw_ids[i],
                    "strict_rule": STRICT_CODE_RULE,
                    "prompt": prompt_texts[i],
                    "response": response_text,
                    "finish_reason": finish_reason,
                    "response_token_len": int(len(token_ids)),
                    "truncated": bool(finish_reason == "length"),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                global_idx += 1
                written += 1

            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[vllm-infer] progress={written}/{target_rows} ({written/target_rows:.2%}) "
                f"speed={written/elapsed:.2f} samples/s",
                flush=True,
            )

    elapsed = time.time() - started
    print(f"[vllm-infer] done rows={written} elapsed={elapsed:.2f}s output={args.output_jsonl}")


if __name__ == "__main__":
    main()
