#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task_find_common import TOOL_SCHEMA, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export task-find dual-track records to Qwen3/LLaMA-Factory message format.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _pick_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("qwen3_messages") or row.get("messages") or []
    return _jsonable(messages)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            messages = _pick_messages(row)
            if not messages:
                continue
            record = {
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "tools": [TOOL_SCHEMA],
                "messages": messages,
                "teacher_protocol": _jsonable(row.get("teacher_protocol", {})),
                "teacher_messages": _jsonable(row.get("teacher_messages", [])),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"status": "ok", "count": count, "output": str(args.output_jsonl)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
