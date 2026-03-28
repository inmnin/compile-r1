#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import aiohttp

from distill_acecode_tool_coldstart import (
    AsyncResultCache,
    call_chat_completion,
    call_compile_server_run,
    ensure_compile_server,
    stop_compile_server,
)
from task_find_common import (
    TOOL_SCHEMA,
    build_explicit_tool_messages,
    build_tool_messages,
    grade_prediction,
    is_executable_python_script,
    keep_tool_positive,
    load_jsonl,
    summarize_tool_utility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek teacher + local python_exec on task_find candidate data.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--base-url", type=str, default=os.environ.get("JUDGE_SET_3_BASE_URL") or "https://api.deepseek.com/beta")
    parser.add_argument("--api-key", type=str, default=os.environ.get("JUDGE_SET_3_API_KEY") or "")
    parser.add_argument("--model-name", type=str, default=os.environ.get("JUDGE_SET_3_MODEL_NAME") or "deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--max-tool-turns", type=int, default=2)
    parser.add_argument("--tool-choice", type=str, default="auto", choices=["auto", "required"])
    parser.add_argument("--tool-mode", type=str, default="openai", choices=["openai", "explicit_xml"])
    parser.add_argument("--compile-server-url", type=str, default="http://127.0.0.1:18086")
    parser.add_argument("--tool-timeout", type=int, default=12)
    parser.add_argument("--judge-timeout", type=int, default=12)
    parser.add_argument("--compile-retries", type=int, default=2)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--auto-start-compile-server", action="store_true")
    parser.add_argument("--compile-server-script", type=str, default="tools/RunPythonTool.py")
    parser.add_argument("--runpy-max-workers", type=int, default=8)
    parser.add_argument("--runpy-max-inflight", type=int, default=24)
    parser.add_argument("--runpy-queue-wait-timeout", type=float, default=1.5)
    parser.add_argument("--runpy-worker-max-tasks", type=int, default=400)
    parser.add_argument("--max-protocol-retries", type=int, default=1)
    parser.add_argument("--sample-concurrency", type=int, default=1)
    parser.add_argument("--hard-mode", action="store_true")
    return parser.parse_args()


def parse_tool_call(tool_call: dict[str, Any]) -> tuple[bool, bool, str]:
    name = str(tool_call.get("function", {}).get("name", ""))
    raw_arguments = tool_call.get("function", {}).get("arguments", "")
    if name != "python_exec":
        return False, False, ""
    try:
        parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        return True, False, ""
    if not isinstance(parsed, dict):
        return True, False, ""
    return True, True, str(parsed.get("code", ""))


_TOOL_CALL_TAG_RE = re.compile(r"^\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*$", re.S)
_TOOL_CALL_XML_RE = re.compile(
    r"^\s*<tool_call>\s*<name>\s*([^<]+?)\s*</name>\s*<arguments>\s*<code>(.*?)</code>\s*</arguments>\s*</tool_call>\s*$",
    re.S,
)


def parse_explicit_tool_call(content: str) -> tuple[bool, str, dict[str, Any] | None]:
    text = str(content or "")
    match = _TOOL_CALL_TAG_RE.match(text)
    code = ""
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return True, "invalid_tool_call_payload", None
        if not isinstance(payload, dict):
            return True, "invalid_tool_call_payload", None
        name = str(payload.get("name", ""))
        arguments = payload.get("arguments")
        if name != "python_exec" or not isinstance(arguments, dict):
            return True, "invalid_tool_call_payload", None
        code = str(arguments.get("code", ""))
    else:
        xml_match = _TOOL_CALL_XML_RE.match(text)
        if not xml_match:
            return False, "not_tool_call", None
        name = str(xml_match.group(1) or "").strip()
        if name != "python_exec":
            return True, "invalid_tool_call_payload", None
        code = str(xml_match.group(2) or "")
    if not code.strip():
        return True, "tool_code_empty_script", None
    return True, "", {
        "id": "explicit_call_0",
        "type": "function",
        "function": {
            "name": "python_exec",
            "arguments": json.dumps({"code": code}, ensure_ascii=False, separators=(",", ":")),
        },
    }


def explicit_tool_call_text(code: str) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": "python_exec", "arguments": {"code": code}}, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_call>"
    )


def explicit_tool_response_text(result: dict[str, Any]) -> str:
    return "<tool_response>\n" + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n</tool_response>"


def protocol_repair_message(reason: str, *, require_tool: bool) -> str:
    lines = [
        "Protocol violation.",
        f"Your previous assistant message was rejected: {reason}.",
    ]
    if require_tool:
        lines.append("You must now output exactly one valid <tool_call>...</tool_call> block and nothing else.")
    else:
        lines.append("You must now output either exactly one valid <tool_call>...</tool_call> block and nothing else, or the final answer only.")
    lines.extend(
        [
            "Inside <tool_call>, return JSON with name=python_exec and arguments.code=<executable python script>.",
            "The code must do actual computation or checking on concrete values.",
            "Do not print or restate an English sentence from the prompt.",
            "Do not output Markdown or explanation.",
        ]
    )
    if reason == "tool_call_low_utility":
        lines.extend(
            [
                "Your previous tool call was valid but not useful enough.",
                "The next tool call must target the main uncertainty of the task, not a trivial final arithmetic step.",
                "Prefer an assert, a candidate comparison, or an explicit check on scale/value selection.",
            ]
        )
    return "\n".join(lines)


async def run_one(row: dict[str, Any], args: argparse.Namespace, llm_session: aiohttp.ClientSession, tool_session: aiohttp.ClientSession, cache: AsyncResultCache) -> dict[str, Any]:
    if args.tool_mode == "explicit_xml":
        teacher_messages: list[dict[str, Any]] = build_explicit_tool_messages(
            row, require_tool=args.tool_choice == "required", hard_mode=args.hard_mode
        )
    else:
        teacher_messages = build_tool_messages(row, hard_mode=args.hard_mode)
    qwen3_messages: list[dict[str, Any]] = build_tool_messages(row, hard_mode=args.hard_mode)
    teacher_raw_messages: list[dict[str, Any]] = list(teacher_messages)
    raw_messages: list[dict[str, Any]] = list(qwen3_messages)
    num_calls = 0
    num_success = 0
    last_answer = ""
    tool_protocol_error = ""
    protocol_retries = 0
    repair_triggered = 0
    utility_repair_triggered = 0

    tool_choice: Any = args.tool_choice
    for _ in range(args.max_tool_turns + 1):
        response = await call_chat_completion(
            llm_session,
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            messages=teacher_messages,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            timeout=args.llm_timeout,
            retries=args.llm_retries,
            tools=[TOOL_SCHEMA] if args.tool_mode == "openai" else None,
            tool_choice=tool_choice if args.tool_mode == "openai" else None,
        )
        tool_choice = "auto"
        message = response["message"]
        content = message.get("content", "")
        tool_calls = message.get("tool_calls") or []

        if args.tool_mode == "explicit_xml":
            teacher_raw_messages.append({"role": "assistant", "content": str(content or "")})
            is_tool_call, reason, parsed_tool_call = parse_explicit_tool_call(str(content or ""))
            if is_tool_call:
                if parsed_tool_call is None:
                    if protocol_retries < args.max_protocol_retries:
                        repair = {"role": "user", "content": protocol_repair_message(reason, require_tool=args.tool_choice == "required")}
                        teacher_messages.append(repair)
                        teacher_raw_messages.append(repair)
                        protocol_retries += 1
                        repair_triggered += 1
                        continue
                    tool_protocol_error = reason
                    break
                tool_calls = [parsed_tool_call]
                content = None
        else:
            teacher_raw_messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                }
            )

        if tool_calls:
            if str(content or "").strip():
                if args.tool_mode == "explicit_xml" and protocol_retries < args.max_protocol_retries:
                    repair = {
                        "role": "user",
                        "content": protocol_repair_message("tool_call_turn_contains_text", require_tool=args.tool_choice == "required"),
                    }
                    teacher_messages.append(repair)
                    teacher_raw_messages.append(repair)
                    protocol_retries += 1
                    repair_triggered += 1
                    continue
                tool_protocol_error = "tool_call_turn_contains_text"
                break
            retry_reason = ""
            validated_calls: list[tuple[dict[str, Any], str]] = []
            for tool_call in tool_calls:
                if num_calls >= args.max_tool_turns:
                    tool_protocol_error = "too_many_tool_calls"
                    break
                valid_name, valid_json, code = parse_tool_call(tool_call)
                if not valid_name or not valid_json or not code.strip():
                    retry_reason = "invalid_tool_call_payload"
                    break
                code_ok, code_reason = is_executable_python_script(code)
                if not code_ok:
                    retry_reason = f"tool_code_{code_reason}"
                    break
                utility = summarize_tool_utility(
                    [
                        {
                            "role": "assistant",
                            "tool_calls": [tool_call],
                        }
                    ],
                    row=row,
                )
                if args.tool_choice == "required" and utility["all_low_utility"]:
                    retry_reason = "tool_call_low_utility"
                    break
                validated_calls.append((tool_call, code))
            if retry_reason:
                if args.tool_mode == "explicit_xml" and protocol_retries < args.max_protocol_retries:
                    repair = {
                        "role": "user",
                        "content": protocol_repair_message(retry_reason, require_tool=args.tool_choice == "required"),
                    }
                    teacher_messages.append(repair)
                    teacher_raw_messages.append(repair)
                    protocol_retries += 1
                    repair_triggered += 1
                    if retry_reason == "tool_call_low_utility":
                        utility_repair_triggered += 1
                    continue
                tool_protocol_error = retry_reason
                break
            if tool_protocol_error:
                break
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call for tool_call, _ in validated_calls],
            }
            raw_messages.append(assistant_message)
            if args.tool_mode == "openai":
                teacher_messages.append(assistant_message)
            for tool_call, code in validated_calls:
                result = await call_compile_server_run(
                    tool_session,
                    cache,
                    compile_server_url=args.compile_server_url,
                    code=code,
                    timeout=args.tool_timeout,
                    retries=args.compile_retries,
                )
                num_calls += 1
                if str(result.get("status", "")) == "success":
                    num_success += 1
                tool_message = {
                    "role": "tool",
                    "tool_call_id": str(tool_call.get("id", "")),
                    "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                }
                raw_messages.append(tool_message)
                if args.tool_mode == "explicit_xml":
                    teacher_messages.append({"role": "assistant", "content": explicit_tool_call_text(code)})
                    teacher_tool_response = {"role": "user", "content": explicit_tool_response_text(result)}
                    teacher_messages.append(teacher_tool_response)
                    teacher_raw_messages.append(teacher_tool_response)
                else:
                    teacher_messages.append(tool_message)
            if tool_protocol_error:
                break
            continue

        last_answer = str(content or "").strip()
        raw_messages.append({"role": "assistant", "content": last_answer})
        if args.tool_mode == "openai":
            teacher_messages.append({"role": "assistant", "content": last_answer})
        break

    ok_tool, judge_tool = grade_prediction(row, last_answer)
    keep = keep_tool_positive(row, False, ok_tool, num_calls, num_success, raw_messages)
    utility_summary = summarize_tool_utility(raw_messages, row=row)
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "split": row["split"],
        "tools": [TOOL_SCHEMA],
        "messages": raw_messages,
        "qwen3_messages": raw_messages,
        "teacher_messages": teacher_raw_messages,
        "teacher_protocol": {
            "mode": args.tool_mode,
            "tool_choice": args.tool_choice,
            "api_native_tools": args.tool_mode == "openai",
        },
        "pred_tool": last_answer,
        "ok_tool": ok_tool,
        "judge_tool": judge_tool,
        "tool_stats": {"num_calls": num_calls, "num_success_calls": num_success},
        "repair_stats": {
            "repair_triggered": repair_triggered,
            "utility_repair_triggered": utility_repair_triggered,
        },
        "utility_summary": utility_summary,
        "tool_protocol_error": tool_protocol_error,
        "keep_decision_preview": {
            "keep": keep.keep,
            "reason": keep.reason,
            "error_type": keep.error_type,
        },
    }


async def amain() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key for teacher runner.")
    rows = load_jsonl(args.input_jsonl, limit=args.max_samples)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cache = AsyncResultCache()
    server_proc = await ensure_compile_server(args)
    llm_timeout = aiohttp.ClientTimeout(total=args.llm_timeout + 5)
    tool_timeout = aiohttp.ClientTimeout(total=args.tool_timeout + 10)
    try:
        async with aiohttp.ClientSession(timeout=llm_timeout) as llm_session, aiohttp.ClientSession(timeout=tool_timeout) as tool_session:
            queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
            for idx, row in enumerate(rows, start=1):
                queue.put_nowait((idx, row))
            write_lock = asyncio.Lock()

            async def worker() -> None:
                while True:
                    try:
                        idx, row = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    result = await run_one(row, args, llm_session, tool_session, cache)
                    payload = json.dumps(result, ensure_ascii=False) + "\n"
                    async with write_lock:
                        with args.output_jsonl.open("a", encoding="utf-8") as f:
                            f.write(payload)
                        print(
                            json.dumps(
                                {
                                    "index": idx,
                                    "id": row["id"],
                                    "ok_tool": result["ok_tool"],
                                    "num_calls": result["tool_stats"]["num_calls"],
                                    "num_success_calls": result["tool_stats"]["num_success_calls"],
                                    "tool_protocol_error": result["tool_protocol_error"],
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    queue.task_done()

            args.output_jsonl.write_text("", encoding="utf-8")
            workers = [asyncio.create_task(worker()) for _ in range(max(1, args.sample_concurrency))]
            await asyncio.gather(*workers)
    finally:
        stop_compile_server(server_proc)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
