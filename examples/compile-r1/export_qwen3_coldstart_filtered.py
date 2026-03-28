#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from datasets import load_dataset

from acecode_tool_coldstart_common import PYTHON_EXEC_TOOL, TOOL_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter distilled cold-start data and export Qwen3 message-format trajectories.")
    parser.add_argument("--input-jsonl", required=True, type=str)
    parser.add_argument("--output-jsonl", required=True, type=str)
    parser.add_argument("--doc-path", required=True, type=str)
    parser.add_argument("--data-root", default="examples/compile-r1/data", type=str)
    parser.add_argument("--examples-per-bucket", default=50, type=int)
    parser.add_argument("--seed", default=20260323, type=int)
    parser.add_argument("--max-workers", default=0, type=int)
    return parser.parse_args()


def load_acecode_records(data_root: Path) -> dict[str, dict[str, Any]]:
    local_pattern = str((data_root / "train" / "data" / "*.parquet").resolve())
    dataset = load_dataset("parquet", data_files=local_pattern, split="train")
    records: dict[str, dict[str, Any]] = {}
    for row in dataset:
        obj = dict(row)
        sample_id = str(obj.get("id") or "")
        if sample_id:
            records[sample_id] = _jsonable(obj)
    return records


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tool_invocation_failed(route: dict[str, Any]) -> bool:
    tool_calls = route.get("tool_calls") or []
    for tc in tool_calls:
        if not tc.get("valid_name", False):
            return True
        if not tc.get("valid_json", False):
            return True
        if not tc.get("self_contained", False):
            return True
        if str(tc.get("execution_status", "")) in {"invalid_call", "invalid_script", "busy", "system_error"}:
            return True
    return False


def _normalize_messages(record: dict[str, Any], route: dict[str, Any]) -> list[dict[str, Any]]:
    question = str(record.get("question") or "").strip()
    raw_messages = list(route.get("raw_messages") or [])
    normalized: list[dict[str, Any]] = []
    normalized.append({"role": "system", "content": TOOL_SYSTEM_PROMPT})
    if raw_messages and raw_messages[0].get("role") == "system":
        raw_messages = raw_messages[1:]

    user_added = False
    tool_records = list(route.get("tool_calls") or [])
    tool_idx = 0
    final_added = False

    for msg in raw_messages:
        role = str(msg.get("role", ""))
        if role == "user":
            normalized.append({"role": "user", "content": str(msg.get("content", ""))})
            user_added = True
            continue
        if role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": str(msg.get("content", ""))}
            if msg.get("tool_calls"):
                assistant_msg["tool_calls"] = _jsonable(msg.get("tool_calls"))
                normalized.append(assistant_msg)
                for tool_call in msg.get("tool_calls") or []:
                    record_tool = tool_records[tool_idx] if tool_idx < len(tool_records) else {}
                    observation = record_tool.get("observation", {})
                    normalized.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tool_call.get("id", "")),
                            "content": json.dumps(_jsonable(observation), ensure_ascii=False, separators=(",", ":")),
                        }
                    )
                    tool_idx += 1
            else:
                normalized.append(assistant_msg)
                if assistant_msg["content"].strip():
                    final_added = True
            continue
        if role == "tool":
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": str(msg.get("tool_call_id", "")),
                    "content": str(msg.get("content", "")),
                }
            )

    if not user_added:
        normalized.insert(
            1,
            {
                "role": "user",
                "content": (
                    "Solve the following Python coding task.\n\n"
                    "Return only the final Python solution in your last assistant message. "
                    "No Markdown fences, no explanation, and no tests.\n\n"
                    f"Task:\n{question}"
                ),
            },
        )

    final_code = str(route.get("final_code") or "").strip()
    if final_code and not final_added:
        normalized.append({"role": "assistant", "content": final_code})
    return normalized


def _convert_record(record: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    bucket = str(record.get("bucket") or "")
    if bucket not in {"direct", "single_tool", "repair"}:
        return None, None

    selected_route = str(record.get("selected_route") or "")
    route = ((record.get("route_results") or {}).get(selected_route) or {}) if selected_route else {}
    if not route or not route.get("passed_all", False):
        return None, None
    if bucket in {"single_tool", "repair"}:
        if _tool_invocation_failed(route):
            return None, None
        if int(route.get("valid_tool_calls", 0) or 0) < 1:
            return None, None

    messages = _normalize_messages(record, route)
    exported = {
        "id": record.get("id"),
        "source": record.get("source"),
        "bucket": bucket,
        "question": record.get("question"),
        "selected_route": selected_route,
        "bucket_reason": record.get("bucket_reason"),
        "tools": [PYTHON_EXEC_TOOL],
        "messages": messages,
        "qwen3_messages": messages,
        "system": TOOL_SYSTEM_PROMPT,
        "conversations": (((record.get("sharegpt_record") or {}).get("conversations")) or route.get("conversation") or []),
    }
    return bucket, exported


def write_markdown_examples(
    *,
    path: Path,
    buckets: dict[str, list[dict[str, Any]]],
    raw_records: dict[str, dict[str, Any]],
    examples_per_bucket: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    lines: list[str] = ["# Cold Start Examples", ""]
    for bucket in ("direct", "single_tool", "repair"):
        items = list(buckets.get(bucket, []))
        if len(items) < examples_per_bucket:
            raise RuntimeError(f"bucket {bucket} only has {len(items)} records, need {examples_per_bucket}")
        chosen = rng.sample(items, examples_per_bucket)
        lines.append(f"## {bucket}")
        lines.append("")
        for idx, item in enumerate(chosen, 1):
            sample_id = str(item.get("id"))
            raw_obj = raw_records.get(sample_id)
            if raw_obj is None:
                raise RuntimeError(f"raw record missing for sample id {sample_id}")
            lines.append(f"### {bucket}-{idx}: {sample_id}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(raw_obj, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    max_workers = args.max_workers if args.max_workers > 0 else max(8, min(64, (os.cpu_count() or 8) * 2))

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    doc_path = Path(args.doc_path)

    raw_records = load_acecode_records(Path(args.data_root))

    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in raw_lines if line.strip()]

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exported_records: list[dict[str, Any]] = []
    filtered_counter: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for bucket, exported in executor.map(_convert_record, records, chunksize=64):
            if bucket is None or exported is None:
                filtered_counter["dropped"] += 1
                continue
            exported_records.append(exported)
            buckets[bucket].append(exported)
            filtered_counter[bucket] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for obj in exported_records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    write_markdown_examples(
        path=doc_path,
        buckets=buckets,
        raw_records=raw_records,
        examples_per_bucket=args.examples_per_bucket,
        seed=args.seed,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "max_workers": max_workers,
                "output_jsonl": str(output_path),
                "doc_path": str(doc_path),
                "bucket_counts": {k: len(v) for k, v in buckets.items()},
                "dropped": filtered_counter["dropped"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
