#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import aiohttp
from acecode_tool_coldstart_common import PYTHON_EXEC_TOOL, normalize_test_cases

from distill_acecode_tool_coldstart import (
    AsyncResultCache,
    DistillDecision,
    choose_indices,
    ensure_compile_server,
    ensure_llm_connectivity,
    load_acecode_dataset,
    parse_args as parse_distill_args,
    process_one_sample,
    route_to_dict,
    run_direct_route,
    run_tool_route,
    stop_compile_server,
)


TOOL_PROTOCOL_FAILURES = {
    "invalid_tool_call_payload",
    "tool_code_not_self_contained",
    "tool_call_turn_contains_text",
}


def _route_has_protocol_failure(route: dict[str, Any] | None) -> bool:
    route = route or {}
    if str(route.get("failed_reason", "")) in TOOL_PROTOCOL_FAILURES:
        return True
    for tc in route.get("tool_calls") or []:
        if not tc.get("valid_name", False):
            return True
        if not tc.get("valid_json", False):
            return True
        if not tc.get("self_contained", False):
            return True
        if str(tc.get("execution_status", "")) in {"invalid_call", "invalid_script", "busy", "system_error"}:
            return True
    return False


def _passrate_100(route: dict[str, Any] | None) -> bool:
    route = route or {}
    return bool(route.get("passed_all", False))


def _raw_record_from_route(
    *,
    sample_id: str,
    source: str,
    question: str,
    bucket: str,
    route: dict[str, Any],
    direct: dict[str, Any] | None,
    auto: dict[str, Any] | None,
    required: dict[str, Any] | None,
) -> dict[str, Any]:
    def ratio(r: dict[str, Any] | None) -> float | None:
        r = r or {}
        total = int(r.get("total_count", 0) or 0)
        if total <= 0:
            return None
        return float(int(r.get("passed_count", 0) or 0)) / float(total)

    return {
        "id": sample_id,
        "bucket": bucket,
        "question": question,
        "source": source,
        "tools": [route.get("tools_schema")] if route.get("tools_schema") else [],
        "messages": route.get("raw_messages") or [],
        "hidden_eval": {
            "pass_direct": ratio(direct),
            "pass_auto": ratio(auto),
            "pass_required": ratio(required),
        },
    }


def _extract_bucket_candidates(result: DistillDecision) -> list[tuple[str, dict[str, Any]]]:
    rr = result.route_results or {}
    direct = rr.get("direct") or {}
    auto = rr.get("auto") or {}
    required = rr.get("required") or {}
    sample_id = result.sample_id
    source = result.source
    question = result.question
    candidates: list[tuple[str, dict[str, Any]]] = []

    if _passrate_100(direct):
        direct = dict(direct)
        direct["tools_schema"] = None
        candidates.append(
            (
                "direct",
                _raw_record_from_route(
                    sample_id=sample_id,
                    source=source,
                    question=question,
                    bucket="direct",
                    route=direct,
                    direct=direct,
                    auto=auto,
                    required=required,
                ),
            )
        )

    tool_routes = []
    for route_name, route in (("auto", auto), ("required", required)):
        if not route:
            continue
        if not _passrate_100(route):
            continue
        if _route_has_protocol_failure(route):
            continue
        if int(route.get("valid_tool_calls", 0) or 0) < 1:
            continue
        tool_routes.append((route_name, route))

    if tool_routes:
        tool_routes.sort(
            key=lambda item: (
                int(item[1].get("valid_tool_calls", 0) or 0),
                int(item[1].get("successful_tool_calls", 0) or 0),
                bool(item[1].get("observation_utilized", False)),
            ),
            reverse=True,
        )
        route_name, route = tool_routes[0]
        route = dict(route)
        route["tools_schema"] = PYTHON_EXEC_TOOL
        bucket = "repair" if int(route.get("valid_tool_calls", 0) or 0) >= 2 and bool(route.get("observation_utilized", False)) else "single_tool"
        candidates.append(
            (
                bucket,
                _raw_record_from_route(
                    sample_id=sample_id,
                    source=source,
                    question=question,
                    bucket=bucket,
                    route=route,
                    direct=direct,
                    auto=auto,
                    required=required,
                ),
            )
        )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 50 examples for each cold-start bucket using teacher distillation.",
        add_help=True,
    )
    parser.add_argument("--target-per-bucket", type=int, default=50)
    parser.add_argument("--max-attempts-per-sample", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="examples/compile-r1/data/cold_start_collect")
    parser.add_argument("--output-prefix", type=str, default="cold_start_collect")
    parser.add_argument("--doc-path", type=str, default="examples/compile-r1/doc/cold_start_example.md")

    known, remaining = parser.parse_known_args()
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        distill_args = parse_distill_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(known).items():
        setattr(distill_args, key, value)
    return distill_args


def _needs_retry(result: DistillDecision) -> bool:
    for route_name in ("auto", "required"):
        route = (result.route_results or {}).get(route_name) or {}
        if str(route.get("failed_reason", "")) in TOOL_PROTOCOL_FAILURES:
            return True
    return False


def _write_md(path: Path, bucket_records: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Cold Start Examples", ""]
    for bucket in ("direct", "single_tool", "repair"):
        lines.append(f"## {bucket}")
        lines.append("")
        for idx, obj in enumerate(bucket_records[bucket], 1):
            lines.append(f"### {bucket}-{idx}: {obj['id']}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(obj, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


async def _process_direct_only_sample(
    *,
    row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    llm_session: aiohttp.ClientSession,
    tool_session: aiohttp.ClientSession,
    cache: AsyncResultCache,
) -> DistillDecision:
    sample_id = str(row.get("id") or f"row_{row_index}")
    source = str(row.get("source") or "")
    question = str(row.get("question") or "").strip()
    tests = normalize_test_cases(row.get("test_cases"))
    direct = await run_direct_route(
        question=question,
        tests=tests,
        args=args,
        llm_session=llm_session,
        tool_session=tool_session,
        cache=cache,
    )
    bucket = "direct" if direct.passed_all else "discard"
    return DistillDecision(
        sample_id=sample_id,
        source=source,
        question=question,
        static_bucket="direct_candidate",
        static_score=0,
        risk_flags=[],
        bucket=bucket,
        bucket_reason="direct_only_collector",
        selected_route="direct" if direct.passed_all else "",
        raw_record=None,
        sharegpt_record=None,
        route_results={"direct": route_to_dict(direct), "auto": None, "required": None},
    )


async def _process_tool_only_sample(
    *,
    row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    llm_session: aiohttp.ClientSession,
    tool_session: aiohttp.ClientSession,
    cache: AsyncResultCache,
) -> DistillDecision:
    sample_id = str(row.get("id") or f"row_{row_index}")
    source = str(row.get("source") or "")
    question = str(row.get("question") or "").strip()
    tests = normalize_test_cases(row.get("test_cases"))
    auto = await run_tool_route(
        question=question,
        tests=tests,
        args=args,
        llm_session=llm_session,
        tool_session=tool_session,
        cache=cache,
        path_name="auto",
        initial_tool_choice="auto",
    )
    required = auto
    if (
        not auto.passed_all
        or auto.valid_tool_calls == 0
        or auto.has_tool_invocation_failure
    ):
        required = await run_tool_route(
            question=question,
            tests=tests,
            args=args,
            llm_session=llm_session,
            tool_session=tool_session,
            cache=cache,
            path_name="required",
            initial_tool_choice="required",
        )

    best_bucket = "discard"
    best_route = None
    candidates = []
    for route in (auto, required if required is not auto else None):
        if route is None:
            continue
        if not route.passed_all:
            continue
        if route.valid_tool_calls < 1:
            continue
        if route.has_tool_invocation_failure:
            continue
        bucket = "repair" if route.valid_tool_calls >= 2 and route.observation_utilized else "single_tool"
        candidates.append((bucket, route))
    if candidates:
        candidates.sort(
            key=lambda item: (
                item[0] == "repair",
                item[1].valid_tool_calls,
                item[1].successful_tool_calls,
                item[1].unique_tool_codes,
            ),
            reverse=True,
        )
        best_bucket, best_route = candidates[0]

    return DistillDecision(
        sample_id=sample_id,
        source=source,
        question=question,
        static_bucket="tool_candidate",
        static_score=0,
        risk_flags=[],
        bucket=best_bucket,
        bucket_reason="tool_only_collector",
        selected_route=best_route.path_name if best_route is not None else "",
        raw_record=None,
        sharegpt_record=None,
        route_results={"direct": None, "auto": route_to_dict(auto), "required": route_to_dict(required if required is not auto else None)},
    )


async def run_collection(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("Missing API key. Set JUDGE_SET_3_API_KEY / JUDGE_SET_2_API_KEY or pass --api-key.")

    data_root = Path(args.data_root)
    dataset = load_acecode_dataset(data_root, args.train_dataset, args.train_split)
    indices, static_cache, _ = choose_indices(dataset, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / f"{args.output_prefix}.raw.jsonl"
    if raw_jsonl.exists():
        raw_jsonl.unlink()

    server_proc = await ensure_compile_server(args)
    await ensure_llm_connectivity(args)

    target = int(args.target_per_bucket)
    bucket_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_ids: dict[str, set[str]] = defaultdict(set)
    counters = Counter()
    cache = AsyncResultCache()
    cursor = 0
    cursor_lock = asyncio.Lock()
    bucket_lock = asyncio.Lock()

    llm_timeout = aiohttp.ClientTimeout(total=args.llm_timeout + 5)
    tool_timeout = aiohttp.ClientTimeout(total=max(args.tool_timeout, args.judge_timeout) + 10)

    try:
        async with aiohttp.ClientSession(timeout=llm_timeout) as llm_session, aiohttp.ClientSession(timeout=tool_timeout) as tool_session:
            async def worker() -> None:
                nonlocal cursor
                while True:
                    async with bucket_lock:
                        need_direct = len(bucket_records["direct"]) < target
                        need_tool = len(bucket_records["single_tool"]) < target or len(bucket_records["repair"]) < target
                        if not need_direct and not need_tool:
                            return
                    async with cursor_lock:
                        if cursor >= len(indices):
                            return
                        row_index = indices[cursor]
                        cursor += 1
                    row = dataset[row_index]
                    static = static_cache[row_index]
                    if static.bucket == "direct_candidate" and not need_direct:
                        continue
                    if static.bucket == "tool_candidate" and not need_tool:
                        continue
                    result: DistillDecision | None = None
                    for _attempt in range(args.max_attempts_per_sample):
                        if static.bucket == "tool_candidate" and need_tool:
                            result = await _process_tool_only_sample(
                                row=row,
                                row_index=row_index,
                                args=args,
                                llm_session=llm_session,
                                tool_session=tool_session,
                                cache=cache,
                            )
                        elif static.bucket == "direct_candidate" and need_direct:
                            result = await _process_direct_only_sample(
                                row=row,
                                row_index=row_index,
                                args=args,
                                llm_session=llm_session,
                                tool_session=tool_session,
                                cache=cache,
                            )
                        else:
                            result = await process_one_sample(
                                row=row,
                                row_index=row_index,
                                args=args,
                                llm_session=llm_session,
                                tool_session=tool_session,
                                cache=cache,
                                static=static,
                            )
                        if not _needs_retry(result):
                            break
                    if result is None:
                        continue
                    async with bucket_lock:
                        counters["processed"] += 1
                        counters[result.bucket] += 1
                        added = []
                        for bucket_name, raw_record in _extract_bucket_candidates(result):
                            if result.sample_id in bucket_ids[bucket_name]:
                                continue
                            if len(bucket_records[bucket_name]) >= target:
                                continue
                            bucket_records[bucket_name].append(raw_record)
                            bucket_ids[bucket_name].add(result.sample_id)
                            with raw_jsonl.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                            added.append(bucket_name)
                        if added:
                            print(
                                json.dumps(
                                    {
                                        "processed": counters["processed"],
                                        "bucket_counts": {b: len(bucket_records[b]) for b in ("direct", "single_tool", "repair")},
                                        "last_added": {"id": result.sample_id, "bucket": added},
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )

            workers = [asyncio.create_task(worker()) for _ in range(max(1, args.sample_concurrency))]
            await asyncio.gather(*workers)

        for bucket in ("direct", "single_tool", "repair"):
            if len(bucket_records[bucket]) < target:
                raise RuntimeError(f"bucket {bucket} only collected {len(bucket_records[bucket])}, target={target}")
        _write_md(Path(args.doc_path), bucket_records)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "raw_jsonl": str(raw_jsonl),
                    "doc_path": args.doc_path,
                    "bucket_counts": {b: len(bucket_records[b]) for b in ("direct", "single_tool", "repair")},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        stop_compile_server(server_proc)


def main() -> None:
    args = parse_args()
    asyncio.run(run_collection(args))


if __name__ == "__main__":
    main()
