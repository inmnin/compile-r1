#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_find_common import KeepDecision, classify_error_type, keep_tool_positive, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep only tool-positive task_find samples.")
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--no-tool-jsonl", type=Path, required=True)
    parser.add_argument("--tool-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = {row["id"]: row for row in load_jsonl(args.source_jsonl)}
    no_tool_rows = {row["id"]: row for row in load_jsonl(args.no_tool_jsonl)}
    tool_rows = {row["id"]: row for row in load_jsonl(args.tool_jsonl)}
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for sample_id, row in source_rows.items():
            if sample_id not in no_tool_rows or sample_id not in tool_rows:
                continue
            total += 1
            no_tool = no_tool_rows[sample_id]
            tool = tool_rows[sample_id]
            decision = keep_tool_positive(
                row,
                bool(no_tool.get("ok_no_tool", False)),
                bool(tool.get("ok_tool", False)),
                int(((tool.get("tool_stats") or {}).get("num_calls", 0)) or 0),
                int(((tool.get("tool_stats") or {}).get("num_success_calls", 0)) or 0),
                tool.get("messages", []) or [],
            )
            if not decision.keep:
                continue
            kept += 1
            record = {
                "id": sample_id,
                "dataset": row["dataset"],
                "split": row["split"],
                "question": row["question"],
                "table": row.get("table", ""),
                "context": row.get("context", ""),
                "answer": row.get("answer", ""),
                "meta": row.get("meta", {}),
                "pred_no_tool": no_tool.get("pred_no_tool", ""),
                "judge_no_tool": no_tool.get("judge_no_tool", {}),
                "pred_tool": tool.get("pred_tool", ""),
                "judge_tool": tool.get("judge_tool", {}),
                "tools": tool.get("tools", []),
                "messages": tool.get("messages", []),
                "qwen3_messages": tool.get("qwen3_messages", tool.get("messages", [])),
                "teacher_messages": tool.get("teacher_messages", []),
                "teacher_protocol": tool.get("teacher_protocol", {}),
                "tool_stats": tool.get("tool_stats", {}),
                "repair_stats": tool.get("repair_stats", {}),
                "utility_summary": tool.get("utility_summary", {}),
                "keep_reason": decision.reason,
                "error_type": decision.error_type,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"total_joined": total, "kept": kept}, ensure_ascii=False))


if __name__ == "__main__":
    main()
