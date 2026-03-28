#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize high-value tool-use pilot metrics.")
    parser.add_argument("--tool-jsonl", type=Path, required=True)
    parser.add_argument("--keep-jsonl", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    tool_rows = load_jsonl(args.tool_jsonl)
    keep_rows = load_jsonl(args.keep_jsonl) if args.keep_jsonl and args.keep_jsonl.exists() else []

    total = len(tool_rows) or 1
    protocol_errors = sum(1 for row in tool_rows if str(row.get("tool_protocol_error", "")).strip())
    ok_tool = sum(1 for row in tool_rows if bool(row.get("ok_tool", False)))
    calls = [int(((row.get("tool_stats") or {}).get("num_calls", 0)) or 0) for row in tool_rows]
    success_calls = [int(((row.get("tool_stats") or {}).get("num_success_calls", 0)) or 0) for row in tool_rows]
    repair_triggered = [int(((row.get("repair_stats") or {}).get("repair_triggered", 0)) or 0) for row in tool_rows]
    utility_repair = [int(((row.get("repair_stats") or {}).get("utility_repair_triggered", 0)) or 0) for row in tool_rows]

    utility_rows = [row.get("utility_summary", {}) or {} for row in tool_rows]
    assert_rate = sum(1 for u in utility_rows if int(u.get("assert_calls", 0) or 0) > 0) / total
    comparison_rate = sum(1 for u in utility_rows if int(u.get("comparison_calls", 0) or 0) > 0) / total
    low_utility_rate = sum(1 for u in utility_rows if bool(u.get("all_low_utility", False))) / total
    repair_like_rate = sum(1 for u in utility_rows if bool(u.get("repair_like", False))) / total
    print_only_rate = sum(
        1
        for u in utility_rows
        if any("print_only" in (call.get("tags") or []) for call in (u.get("call_summaries") or []))
    ) / total

    summary = {
        "count": len(tool_rows),
        "protocol_error_rate": protocol_errors / total,
        "tool_success_rate": sum(1 for x in success_calls if x > 0) / total,
        "avg_tool_turns": mean(calls) if calls else 0.0,
        "repair_rate": sum(1 for x in calls if x >= 2) / total,
        "repair_with_second_distinct_call_rate": repair_like_rate,
        "assert_rate": assert_rate,
        "comparison_rate": comparison_rate,
        "low_utility_rate": low_utility_rate,
        "trivial_print_only_rate": print_only_rate,
        "final_answer_accuracy": ok_tool / total,
        "repair_prompt_trigger_rate": sum(1 for x in repair_triggered if x > 0) / total,
        "utility_repair_trigger_rate": sum(1 for x in utility_repair if x > 0) / total,
        "kept_count": len(keep_rows),
    }

    out = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(out + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
