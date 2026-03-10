#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


NATURAL_INSTRUCTION = (
    "Answer the given Python coding question.\n"
    "Use tags in a natural way:\n"
    "- <think>...</think>: your reasoning for the current step.\n"
    "- <code>...</code>: Python code to execute when you need verification/debugging.\n"
    "- <result>...</result>: execution feedback returned by the environment.\n"
    "- <answer>...</answer>: the final runnable Python solution.\n\n"
    "You may call <code> multiple rounds before finishing.\n"
    "When tool feedback is needed, prefer running code instead of guessing.\n"
    "Always put the final solution code in <answer>...</answer>."
)


def rewrite_file(path: Path) -> int:
    rows = 0
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as fin, temp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["instruction"] = NATURAL_INSTRUCTION
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rows += 1
    temp_path.replace(path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite cold-start Alpaca instruction to natural prompt.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "/mnt/workspace/jkh/LLaMA-Factory/data/compile_r1_cold_start_train_alpaca.jsonl",
            "/mnt/workspace/jkh/LLaMA-Factory/data/compile_r1_cold_start_eval_alpaca.jsonl",
        ],
    )
    args = parser.parse_args()

    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        rows = rewrite_file(path)
        print(f"updated {rows} rows -> {path}")


if __name__ == "__main__":
    main()
