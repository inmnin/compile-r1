#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


ORIG_INSTRUCTION = """Answer the given Python coding question.
Use tags in a natural way:
- <think>...</think>: your reasoning for the current step.
- <code>...</code>: Python code to execute when you need verification/debugging.
- <result>...</result>: execution feedback returned by the environment.
- <answer>...</answer>: the final runnable Python solution.

You may call <code> multiple rounds before finishing.
When tool feedback is needed, prefer running code instead of guessing.
Always put the final solution code in <answer>...</answer>."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ckpt-400 trajectories on original vs isomorphic prompt formats."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/mnt/workspace/jkh/slime/examples/compile-r1/train_log/cold_start_sft_qwen3_8b_full_natural/checkpoint-400"
        ),
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("/mnt/workspace/jkh/slime/examples/compile-r1/data/train/answer_rl.parquet"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/mnt/workspace/jkh/slime/examples/compile-r1/train_log/ckpt400_prompt_compare_4samples.jsonl"),
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--backend", choices=["cpu", "vllm"], default="cpu")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    return parser.parse_args()


def to_list_prompt(prompt_obj: Any) -> list[dict[str, str]]:
    if isinstance(prompt_obj, list):
        return prompt_obj
    if hasattr(prompt_obj, "tolist"):
        converted = prompt_obj.tolist()
        if isinstance(converted, list):
            return converted
    raise TypeError(f"Unsupported prompt object type: {type(prompt_obj)}")


def render_with_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_with_vllm(args: argparse.Namespace, prompt_texts: list[str]) -> list[str]:
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompt_texts, sampling)
    return [out.outputs[0].text if out.outputs else "" for out in outputs]


def run_with_cpu_transformers(args: argparse.Namespace, tokenizer, prompt_texts: list[str]) -> list[str]:
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    texts: list[str] = []
    do_sample = args.temperature > 0.0
    for i, prompt in enumerate(prompt_texts, start=1):
        print(f"[cpu-generate] {i}/{len(prompt_texts)}")
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attn_mask = encoded.get("attention_mask")
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = generated[0, input_ids.shape[1] :]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return texts


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.train_data).head(args.num_samples)
    rows = df.to_dict(orient="records")
    if len(rows) < args.num_samples:
        raise ValueError(f"Expected at least {args.num_samples} rows in {args.train_data}, got {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    requests: list[dict[str, Any]] = []
    prompt_texts: list[str] = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("id", f"row_{idx}"))
        question = str(row.get("question", "")).strip()
        train_prompt_messages = to_list_prompt(row.get("prompt"))
        if not train_prompt_messages:
            raise ValueError(f"Empty prompt messages for sample {sample_id}")

        train_prompt_content = str(train_prompt_messages[0].get("content", "")).strip()
        if not train_prompt_content:
            raise ValueError(f"Empty prompt content for sample {sample_id}")

        # Variant A: current cold-start alpaca style (instruction + raw question).
        original_messages = [{"role": "user", "content": f"{ORIG_INSTRUCTION}\n\n{question}"}]
        # Variant B: isomorphic to RL train prompt (identical content as answer_rl prompt).
        isomorphic_messages = [{"role": "user", "content": train_prompt_content}]

        for variant_name, messages in (
            ("original_question_only", original_messages),
            ("new_isomorphic_train_prompt", isomorphic_messages),
        ):
            rendered = render_with_chat_template(tokenizer, messages)
            requests.append(
                {
                    "id": sample_id,
                    "row_index": idx,
                    "variant": variant_name,
                    "question": question,
                    "rendered_prompt": rendered,
                }
            )
            prompt_texts.append(rendered)

    if args.backend == "vllm":
        generated_texts = run_with_vllm(args, prompt_texts)
    else:
        generated_texts = run_with_cpu_transformers(args, tokenizer, prompt_texts)
    assert len(generated_texts) == len(requests)

    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for req, text in zip(requests, generated_texts, strict=True):
            record = {
                "id": req["id"],
                "row_index": req["row_index"],
                "variant": req["variant"],
                "question": req["question"],
                "trajectory": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved_jsonl={args.output_jsonl}")
    print("========== TRAJECTORIES ==========")
    for req, text in zip(requests, generated_texts, strict=True):
        print(f"\n--- sample_id={req['id']} variant={req['variant']} ---")
        print(text)


if __name__ == "__main__":
    main()
