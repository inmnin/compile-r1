#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter train parquet by rollout length statistics.")
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--index-metrics-parquet", type=Path, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument(
        "--preserve-positive-reward",
        action="store_true",
        help="If set, keep samples with max_reward>0 even if they exceed threshold.",
    )
    args = parser.parse_args()

    table = pq.read_table(args.input_parquet)
    total_rows = table.num_rows

    metrics = pd.read_parquet(args.index_metrics_parquet)
    required_cols = {"index", "max_response_length", "max_reward"}
    if not required_cols.issubset(set(metrics.columns)):
        raise ValueError(f"index metrics parquet must contain columns: {sorted(required_cols)}")

    long_mask = metrics["max_response_length"] > args.threshold
    if args.preserve_positive_reward:
        drop_df = metrics[long_mask & (metrics["max_reward"] <= 0)]
    else:
        drop_df = metrics[long_mask]
    drop_indices = set(drop_df["index"].astype(int).tolist())

    keep_mask = np.ones(total_rows, dtype=bool)
    for idx in drop_indices:
        if 0 <= idx < total_rows:
            keep_mask[idx] = False

    filtered_table = table.filter(pa.array(keep_mask))
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered_table, args.output_parquet, compression="zstd")

    preserved_positive_long = int(
        (
            metrics[
                (metrics["max_response_length"] > args.threshold)
                & (metrics["max_reward"] > 0)
            ].shape[0]
        )
    )
    report = {
        "input_parquet": str(args.input_parquet),
        "index_metrics_parquet": str(args.index_metrics_parquet),
        "threshold": int(args.threshold),
        "preserve_positive_reward": bool(args.preserve_positive_reward),
        "total_rows": int(total_rows),
        "drop_count": int((~keep_mask).sum()),
        "keep_count": int(keep_mask.sum()),
        "drop_ratio": float((~keep_mask).mean()),
        "preserved_positive_long_count": preserved_positive_long,
        "output_parquet": str(args.output_parquet),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

