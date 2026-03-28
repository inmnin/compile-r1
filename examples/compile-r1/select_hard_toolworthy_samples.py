#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HARD_KEYWORDS = [
    "percentage",
    "percent change",
    "percent",
    "ratio",
    "margin",
    "growth",
    "contribution",
    "share of",
    "what percent",
    "how much of total",
    "excluding",
    "net of",
    "change from",
    "basis points",
    "bps",
]

SCALE_MARKERS = ["million", "millions", "billion", "thousand", "%", "percent", "basis points", "bps"]
COMPLEX_MARKERS = [
    "overall",
    "of total",
    "as a percentage of",
    "of the balance increase",
    "from prior periods",
    "without",
    "excluding",
    "net of",
    "after reclassification",
    "double the",
    "average of",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select hard tool-worthy samples for high-value tool-use distillation.")
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--review-pack-jsonl", type=Path, required=True)
    parser.add_argument("--max-output", type=int, default=400)
    parser.add_argument("--review-pack-size", type=int, default=60)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text_blob(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("question", "") or ""),
            str(row.get("table", "") or ""),
            str(row.get("context", "") or ""),
        ]
    ).lower()


def table_row_count(table: str) -> int:
    if not table:
        return 0
    return sum(1 for line in str(table).splitlines() if line.strip())


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") or {}
    blob = text_blob(row)
    question = str(row.get("question", "") or "").lower()
    context = str(row.get("context", "") or "")
    table = str(row.get("table", "") or "")
    score = 0
    features: list[str] = []

    matched_hard = [kw for kw in HARD_KEYWORDS if kw in blob]
    if matched_hard:
        score += 3 + min(3, len(matched_hard))
        features.append("hard_keyword")

    scale_hits = sum(1 for token in SCALE_MARKERS if token in blob)
    if scale_hits >= 2:
        score += 2
        features.append("scale_marker")

    if meta.get("has_table") and meta.get("has_text_context"):
        score += 3
        features.append("table_plus_context")

    if int(meta.get("num_steps", 0) or 0) >= 2:
        score += 3
        features.append("multi_step")

    hard_reasons = set(meta.get("candidate_hard_reason") or [])
    if "scale" in hard_reasons:
        score += 2
        features.append("meta_scale")
    if "multi_step" in hard_reasons:
        score += 2
        features.append("meta_multi_step")
    if "table" in hard_reasons:
        score += 1
        features.append("meta_table")

    if table_row_count(table) > 4:
        score += 2
        features.append("dense_table")

    if len(question.split()) >= 9:
        score += 1
        features.append("long_question")

    if len(context.split()) >= 80:
        score += 2
        features.append("long_context")

    numeric_mentions = len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", table + "\n" + context))
    if numeric_mentions >= 8:
        score += 2
        features.append("many_numbers")

    complex_hits = [kw for kw in COMPLEX_MARKERS if kw in blob]
    if complex_hits:
        score += 4
        features.append("complex_marker")

    if meta.get("has_ann_text") and meta.get("has_ann_table"):
        score += 2
        features.append("annotated_hybrid")
    elif meta.get("has_ann_text") or meta.get("has_ann_table"):
        score += 1
        features.append("annotated_signal")

    easy_flags: list[str] = []
    if not meta.get("has_text_context"):
        easy_flags.append("no_context")
    if int(meta.get("num_steps", 0) or 0) <= 1 and len(matched_hard) == 0:
        easy_flags.append("simple_program")
    if len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", question)) <= 2 and len(question.split()) <= 12 and "percent" not in question and "ratio" not in question:
        easy_flags.append("short_direct_question")

    generic_percent_change = bool(
        re.search(r"\b(percent|percentage|growth rate|change)\b", question)
        and re.search(r"\b(from|between)\b", question)
        and not any(marker in blob for marker in COMPLEX_MARKERS)
    )
    if generic_percent_change:
        score -= 4
        easy_flags.append("generic_percent_change")

    route = "hard_tool_worthy" if score >= 10 and "generic_percent_change" not in easy_flags and len(easy_flags) < 2 else "easy_or_direct"
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "split": row["split"],
        "route": route,
        "hard_score": score,
        "rule_features": features,
        "easy_flags": easy_flags,
    }


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for path in args.input_jsonl:
        all_rows.extend(load_jsonl(path))

    scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in all_rows:
        scored.append((row, score_row(row)))

    hard = [(row, info) for row, info in scored if info["route"] == "hard_tool_worthy"]
    hard.sort(key=lambda x: (x[1]["hard_score"], x[0]["dataset"], x[0]["id"]), reverse=True)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.review_pack_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row, info in hard[: args.max_output]:
            payload = {
                **info,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "meta": row.get("meta", {}),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with args.review_pack_jsonl.open("w", encoding="utf-8") as f:
        for row, info in hard[: args.review_pack_size]:
            payload = {
                **info,
                "question": row.get("question", ""),
                "table": row.get("table", ""),
                "context": row.get("context", ""),
                "answer": row.get("answer", ""),
                "meta": row.get("meta", {}),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "total_input": len(all_rows),
                "hard_count": len(hard),
                "output_jsonl": str(args.output_jsonl),
                "review_pack_jsonl": str(args.review_pack_jsonl),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
