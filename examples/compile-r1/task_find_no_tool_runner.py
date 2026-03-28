#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_find_common import build_no_tool_messages, grade_prediction, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-4B no-tool baseline on task_find candidate data.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/workspace/jkh/model/Qwen3-4B"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    has_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16 if has_cuda else torch.float32,
        low_cpu_mem_usage=True,
    )
    if has_cuda:
        model = model.to("cuda")
    model.eval()
    return tokenizer, model


def render_prompt(tokenizer, row: dict[str, Any]) -> str:
    messages = build_no_tool_messages(row)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_one(tokenizer, model, prompt_text: str, max_new_tokens: int, temperature: float) -> str:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
    input_ids = encoded["input_ids"]
    attn_mask = encoded.get("attention_mask")
    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_tokens = output[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl, limit=args.max_samples)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model_path)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            prompt_text = render_prompt(tokenizer, row)
            pred = generate_one(tokenizer, model, prompt_text, args.max_new_tokens, args.temperature)
            ok, judge = grade_prediction(row, pred)
            record = {
                "id": row["id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "pred_no_tool": pred,
                "ok_no_tool": ok,
                "judge_no_tool": judge,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"index": idx, "id": row["id"], "ok_no_tool": ok}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
