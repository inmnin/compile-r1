#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a reproducible train/test split for Search-R1 parquet data.")
    parser.add_argument("--input-parquet", type=str, required=True, help="Source parquet path.")
    parser.add_argument("--output-train", type=str, required=True, help="Output train parquet path.")
    parser.add_argument("--output-test", type=str, required=True, help="Output test parquet path.")
    parser.add_argument("--test-ratio", type=float, default=0.03, help="Fraction used as test set.")
    parser.add_argument("--seed", type=int, default=20260222, help="Random seed for shuffling/sampling.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If > 0, first sample this many rows from input before splitting (budget control).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_parquet)
    out_train = Path(args.output_train)
    out_test = Path(args.output_test)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError(f"--test-ratio must be in (0,1), got {args.test_ratio}")

    df = pd.read_parquet(input_path)
    original_rows = len(df)

    if args.max_rows and args.max_rows > 0:
        sample_n = min(args.max_rows, original_rows)
        df = df.sample(n=sample_n, random_state=args.seed, replace=False).reset_index(drop=True)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_rows = len(df)
    n_test = max(1, int(n_rows * args.test_ratio))
    n_train = n_rows - n_test
    if n_train <= 0:
        raise ValueError(f"Train rows <= 0 after split: n_rows={n_rows}, n_test={n_test}")

    test_df = df.iloc[:n_test].copy()
    train_df = df.iloc[n_test:].copy()

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_train, index=False)
    test_df.to_parquet(out_test, index=False)

    print(
        "Prepared split successfully: "
        f"original_rows={original_rows}, used_rows={n_rows}, train_rows={len(train_df)}, test_rows={len(test_df)}, "
        f"test_ratio={args.test_ratio}, seed={args.seed}, max_rows={args.max_rows}"
    )
    print(f"train={out_train}")
    print(f"test={out_test}")


if __name__ == "__main__":
    main()
