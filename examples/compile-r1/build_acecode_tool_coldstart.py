#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ShareGPT cold-start splits from distilled AceCode tool records.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-size", type=int, default=18000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--smoke-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260310)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("bucket") == "discard":
                continue
            if not row.get("sharegpt_record"):
                continue
            records.append(row)
    return records


def select_counts(total: int, available: dict[str, int]) -> dict[str, int]:
    desired_repair = int(round(total * 0.2))
    repair = min(available.get("repair", 0), desired_repair)
    remaining = total - repair
    direct = min(available.get("direct", 0), remaining // 2)
    single_tool = min(available.get("single_tool", 0), remaining - direct)
    leftover = total - repair - direct - single_tool
    if leftover > 0:
        add_direct = min(available.get("direct", 0) - direct, leftover)
        direct += max(0, add_direct)
        leftover -= max(0, add_direct)
    if leftover > 0:
        add_single = min(available.get("single_tool", 0) - single_tool, leftover)
        single_tool += max(0, add_single)
    return {"direct": direct, "single_tool": single_tool, "repair": repair}


def sample_split(pool: dict[str, list[dict[str, Any]]], size: int) -> list[dict[str, Any]]:
    available = {k: len(v) for k, v in pool.items()}
    counts = select_counts(size, available)
    sampled = []
    for bucket, n in counts.items():
        sampled.extend(pool[bucket][:n])
        del pool[bucket][:n]
    return sampled


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        pool[str(row["bucket"])].append(row["sharegpt_record"])
    for rows in pool.values():
        rng.shuffle(rows)

    train_rows = sample_split(pool, args.train_size)
    val_rows = sample_split(pool, args.val_size)
    smoke_rows = sample_split(pool, args.smoke_size)

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "smoke_eval.jsonl", smoke_rows)

    manifest = {
        "input_jsonl": str(input_path),
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "smoke_size": len(smoke_rows),
        "remaining_pool": {k: len(v) for k, v in pool.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
