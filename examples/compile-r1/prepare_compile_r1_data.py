#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare openai/openai_humaneval data for compile-r1 training.")
    parser.add_argument("--dataset-name", type=str, default="openai/openai_humaneval")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--output-train", type=str, required=True)
    parser.add_argument("--output-test", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260226)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, keep at most this many examples after shuffle (for smoke/precheck).",
    )
    return parser.parse_args()


def build_user_prompt(task_prompt: str, entry_point: str) -> list[dict[str, str]]:
    instruction = (
        "You are solving a Python coding task with an execution tool.\n"
        "You MUST follow this protocol exactly:\n"
        "1) Start with <think>...</think>.\n"
        "2) Then output exactly one action:\n"
        "   - <code>...</code> to run Python code in the tool, OR\n"
        "   - <answer>...</answer> for the final solution.\n"
        "3) The environment will inject tool feedback as <result>...</result> after each <code>.\n"
        "4) Use <result> to iterate, then end with one final <answer>...</answer>.\n\n"
        "Strict constraints:\n"
        "- Never fabricate <result>; only the environment writes <result>.\n"
        "- Keep tags valid and not nested.\n"
        "- If uncertain, prefer <code> to verify behavior before finalizing.\n"
        "- For non-trivial tasks, at least one <code> check is strongly recommended.\n"
        f"- Final <answer> must define function `{entry_point}` and contain code only.\n\n"
        "Problem:\n"
        "```python\n"
        f"{task_prompt.rstrip()}\n"
        "```"
    )
    return [{"role": "user", "content": instruction}]


def main():
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"--train-ratio must be in (0,1), got {args.train_ratio}")

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    records = []

    for idx, row in enumerate(dataset):
        task_id = str(row["task_id"])
        task_prompt = str(row["prompt"])
        entry_point = str(row["entry_point"])
        test_code = str(row["test"])
        canonical_solution = str(row.get("canonical_solution", ""))

        reward_model = {
            "ground_truth": {
                "task_id": task_id,
                "prompt": task_prompt,
                "test": test_code,
                "entry_point": entry_point,
                "canonical_solution": canonical_solution,
            },
            "style": "rule",
        }

        records.append(
            {
                "id": task_id,
                "task_id": task_id,
                "question": task_prompt,
                "prompt": build_user_prompt(task_prompt, entry_point),
                "ability": "code-generation",
                "reward_model": reward_model,
                "extra_info": {"index": idx, "split": args.dataset_split},
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.max_rows and args.max_rows > 0:
        records = records[: args.max_rows]

    n_total = len(records)
    n_train = int(n_total * args.train_ratio)
    n_train = max(1, min(n_total - 1, n_train))
    train_records = records[:n_train]
    test_records = records[n_train:]

    out_train = Path(args.output_train)
    out_test = Path(args.output_test)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_records).to_parquet(out_train, index=False)
    pd.DataFrame(test_records).to_parquet(out_test, index=False)

    print(
        "Prepared compile-r1 data: "
        f"dataset={args.dataset_name}[{args.dataset_split}] "
        f"total={n_total} train={len(train_records)} test={len(test_records)} "
        f"seed={args.seed} max_rows={args.max_rows}"
    )
    print(f"train={out_train}")
    print(f"test={out_test}")


if __name__ == "__main__":
    main()
