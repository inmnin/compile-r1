#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ASSERT_CALL_RE = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
DEF_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cold-start alpaca split (train/val) while keeping raw text unchanged."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/mnt/workspace/jkh/LLaMA-Factory/data/compile_r1_cold_start_train_alpaca.jsonl"),
    )
    parser.add_argument(
        "--output-train-jsonl",
        type=Path,
        default=Path("/mnt/workspace/jkh/LLaMA-Factory/data/compile_r1_cold_start_train_result_masked_val300_alpaca.jsonl"),
    )
    parser.add_argument(
        "--output-val-jsonl",
        type=Path,
        default=Path("/mnt/workspace/jkh/LLaMA-Factory/data/compile_r1_cold_start_val300_result_masked_alpaca.jsonl"),
    )
    parser.add_argument(
        "--output-meta-json",
        type=Path,
        default=Path("/mnt/workspace/jkh/LLaMA-Factory/data/compile_r1_cold_start_train_result_masked_val300_split_meta.json"),
    )
    parser.add_argument(
        "--source-train-glob",
        type=str,
        default="/mnt/workspace/jkh/slime/examples/compile-r1/data/train/data/*.parquet",
    )
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260307)
    return parser.parse_args()


def infer_entry_point(question: str, test_cases: list[str]) -> str:
    for case in test_cases:
        m = ASSERT_CALL_RE.search(str(case))
        if m:
            return m.group(1)
    m = DEF_RE.search(question or "")
    if m:
        return m.group(1)
    return "solution"


def best_completion(inferences: object) -> str:
    if not isinstance(inferences, list):
        return ""
    best = ""
    best_score = -1.0
    for item in inferences:
        if not isinstance(item, dict):
            continue
        comp = str(item.get("completion") or "").strip()
        if not comp:
            continue
        score = float(item.get("pass_rate") or 0.0)
        if score >= best_score:
            best_score = score
            best = comp
    return best


def load_ground_truth_map(source_glob: str) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for fp in sorted(glob.glob(source_glob)):
        table = pq.read_table(fp)
        for row in table.to_pylist():
            task_id = str(row.get("id") or "").strip()
            if not task_id:
                continue
            question = str(row.get("question") or "").strip()
            test_cases_raw = row.get("test_cases") or []
            test_cases = [str(x).strip() for x in test_cases_raw if str(x).strip()]
            if not question or not test_cases:
                continue
            entry_point = infer_entry_point(question, test_cases)
            mapping[task_id] = {
                "task_id": task_id,
                "prompt": question,
                "test": "\n".join(test_cases),
                "test_cases": test_cases,
                "entry_point": entry_point,
                "canonical_solution": best_completion(row.get("inferences")),
            }
    return mapping


def load_alpaca_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input not found: {args.input_jsonl}")

    rows = load_alpaca_rows(args.input_jsonl)
    if args.val_size <= 0 or args.val_size >= len(rows):
        raise ValueError(f"Invalid val_size={args.val_size} for total={len(rows)}")

    gt_map = load_ground_truth_map(args.source_train_glob)
    missing_gt = [str(r.get("id") or "") for r in rows if str(r.get("id") or "") not in gt_map]
    if missing_gt:
        raise KeyError(f"Found {len(missing_gt)} rows without ground-truth mapping. Example: {missing_gt[:5]}")

    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    val_indices = set(indices[: args.val_size])
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for idx, row in enumerate(rows):
        if idx in val_indices:
            val_rows.append(row)
        else:
            train_rows.append(row)

    write_jsonl(args.output_train_jsonl, train_rows)
    write_jsonl(args.output_val_jsonl, val_rows)

    meta = {
        "input_jsonl": str(args.input_jsonl),
        "output_train_jsonl": str(args.output_train_jsonl),
        "output_val_jsonl": str(args.output_val_jsonl),
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "val_size": args.val_size,
        "seed": args.seed,
        "all_have_ground_truth": True,
        "val_ids": [str(r.get("id") or "") for r in val_rows],
    }
    args.output_meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared result-masked split: total={len(rows)} train={len(train_rows)} val={len(val_rows)} seed={args.seed}")
    print(f"train -> {args.output_train_jsonl}")
    print(f"val   -> {args.output_val_jsonl}")
    print(f"meta  -> {args.output_meta_json}")


if __name__ == "__main__":
    main()
