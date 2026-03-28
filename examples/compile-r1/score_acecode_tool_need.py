#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from acecode_tool_coldstart_common import normalize_test_cases, score_static_tool_need


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score AceCode samples for direct-vs-tool cold-start routing.")
    parser.add_argument(
        "--input",
        type=str,
        default="examples/compile-r1/data/train/data/*.parquet",
        help="Parquet glob or HF dataset name.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def load_records(input_spec: str, split: str):
    if "*" in input_spec or input_spec.endswith(".parquet"):
        return load_dataset("parquet", data_files=input_spec, split="train")
    return load_dataset(input_spec, split=split)


def main() -> None:
    args = parse_args()
    ds = load_records(args.input, args.split)
    rows = []
    total = len(ds) if args.max_rows <= 0 else min(len(ds), args.max_rows)
    for idx in range(total):
        row = ds[idx]
        tests = normalize_test_cases(row.get("test_cases"))
        static = score_static_tool_need(
            question=str(row.get("question", "")),
            test_cases=tests,
            inferences=row.get("inferences"),
        )
        rows.append(
            {
                "id": str(row.get("id") or idx),
                "source": str(row.get("source", "")),
                "static_bucket": static.bucket,
                "static_score": static.score,
                "num_tests": static.num_tests,
                "max_inference_pass_rate": static.max_inference_pass_rate,
                "mean_inference_pass_rate": static.mean_inference_pass_rate,
                "risk_flags": static.risk_flags,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        pd.DataFrame(rows).to_parquet(output_path, index=False)

    print(f"Scored {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
