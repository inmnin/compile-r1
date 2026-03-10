#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk

DEFAULT_TRAIN_DATASET = "TIGER-Lab/AceCode-87K"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_TEST_DATASET = "openai/openai_humaneval"
DEFAULT_TEST_SPLIT = "test"

DEFAULT_MODEL_NAME = os.environ.get("JUDGE_SET_3_MODEL_NAME", "deepseek-chat")
DEFAULT_BASE_URL = os.environ.get("JUDGE_SET_3_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_API_KEY = os.environ.get("JUDGE_SET_3_API_KEY", "")

CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RESULT_TAG_RE = re.compile(r"<result>.*?</result>", re.DOTALL)
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

STRICT_TURN_RE = re.compile(
    r"^\s*<think>[\s\S]*?</think>\s*(?:<code>[\s\S]*?</code>|<answer>[\s\S]*?</answer>)\s*$",
    re.DOTALL,
)
STRICT_TRACE_RE = re.compile(
    r"^\s*(?:<think>[\s\S]*?</think>\s*<code>[\s\S]*?</code>\s*<result>[\s\S]*?</result>\s*)*"
    r"<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$",
    re.DOTALL,
)
TAG_PATTERN = re.compile(r"</?(think|code|result|answer)>")
NETWORK_ERROR_RE = re.compile(
    r"(?:"
    r"clientconnectorerror|name(?:\s|_)?resolution(?:\s|_)?error|"
    r"serverdisconnectederror|clientconnectionerror|"
    r"connectionerror|network(?:\s|_)?error|"
    r"llm_(?:network|timeout)_error|compile_server_network_error|"
    r"cannot\s+connect|failed\s+to\s+establish\s+a\s+new\s+connection"
    r")",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Answer the given Python coding question.
Use tags in a natural way:
- <think>...</think>: your reasoning for the current step.
- <code>...</code>: Python code to execute when you need verification/debugging.
- <result>...</result>: execution feedback returned by the environment.
- <answer>...</answer>: the final runnable Python solution.

You may call <code> multiple rounds before finishing.
When tool feedback is needed, prefer running code instead of guessing.
Always put the final solution code in <answer>...</answer>.
"""

FORMAT_RETRY_PROMPT = (
    "Previous trajectory had tag-usage issues. Retry and follow the tag convention clearly: "
    "think in <think>, execute with <code>, and provide final code in <answer>."
)

INVALID_TURN_FEEDBACK = (
    "Tag usage invalid. Next turn should follow the convention: reason in <think>, then use either <code> or <answer>."
)

TOOL_CONTINUE_FEEDBACK = (
    "Use the <result> feedback to continue solving. If uncertain, run another <code> round; otherwise finalize in <answer>."
)


class InfraNetworkError(RuntimeError):
    pass


@dataclass
class AttemptResult:
    ok: bool
    judge_pass: bool
    answer_code: str
    raw_trace: str
    canonical_trace: str
    raw_format_valid: bool
    num_tool_calls: int
    failure_reason: str = ""
    tool_status_counts: Counter = field(default_factory=Counter)


@dataclass
class DistillRecord:
    passed: bool
    output: dict[str, Any]
    failure_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compile-r1 cold-start trajectories from AceCode with DeepSeek.")
    parser.add_argument("--data-root", type=str, default="examples/compile-r1/data")
    parser.add_argument("--train-dataset", type=str, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--train-split", type=str, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--test-dataset", type=str, default=DEFAULT_TEST_DATASET)
    parser.add_argument("--test-split", type=str, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260304)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--time-budget-sec", type=int, default=0, help="If >0, stop after this many seconds.")

    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--sample-concurrency", type=int, default=8)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--compile-retries", type=int, default=3)
    parser.add_argument("--sample-timeout-sec", type=int, default=300)

    parser.add_argument("--compile-server-url", type=str, default="http://127.0.0.1:18080")
    parser.add_argument("--tool-timeout", type=int, default=12)
    parser.add_argument("--judge-timeout", type=int, default=20)
    parser.add_argument("--auto-start-compile-server", action="store_true")
    parser.add_argument("--compile-server-script", type=str, default="tools/RunPythonTool.py")
    parser.add_argument("--runpy-max-workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    parser.add_argument("--runpy-max-inflight", type=int, default=max(16, (os.cpu_count() or 8) * 2))
    parser.add_argument("--runpy-queue-wait-timeout", type=float, default=1.5)
    parser.add_argument("--runpy-worker-max-tasks", type=int, default=200)

    parser.add_argument("--output-prefix", type=str, default="cold_start")
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def dataset_slug(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def clip_text(text: str, max_chars: int = 4096) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...<truncated {len(text) - max_chars} chars>"


def contains_network_error_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(NETWORK_ERROR_RE.search(value))
    if isinstance(value, dict):
        return any(contains_network_error_text(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_network_error_text(v) for v in value)
    return False


def sanitize_network_error_text(value: Any) -> Any:
    if isinstance(value, str):
        return NETWORK_ERROR_RE.sub("[INFRA_REDACTED]", value)
    if isinstance(value, dict):
        return {k: sanitize_network_error_text(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_network_error_text(v) for v in value]
    return value


def trace_has_network_error_text(trace: str, answer_code: str) -> bool:
    return contains_network_error_text(trace) or contains_network_error_text(answer_code)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def normalize_context_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    out: list[dict[str, str]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"

        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "content" in item:
                        parts.append(str(item.get("content", "")))
                else:
                    parts.append(str(item))
            content = "\n".join(x for x in parts if x)

        content = str(content).strip()
        if content:
            out.append({"role": role, "content": content})
    return out


def normalize_test_cases(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except json.JSONDecodeError:
            pass
        return [text]
    return []


def build_prompt_messages(row: dict[str, Any], retry_format: bool = False) -> tuple[list[dict[str, str]], str]:
    question = str(row.get("question", "")).strip()
    context_messages = normalize_context_messages(row.get("context_messages"))

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    prompt_input = ""

    if context_messages:
        messages.extend(context_messages)
        prompt_input = json.dumps(context_messages, ensure_ascii=False)
    else:
        user_content = (
            "Solve the Python programming task below. "
            "If uncertain, use <code> first and wait for <result>.\n\n"
            f"Problem:\n{question}"
        )
        messages.append({"role": "user", "content": user_content})
        prompt_input = question

    if retry_format:
        messages.append({"role": "user", "content": FORMAT_RETRY_PROMPT})

    messages.append(
        {
            "role": "user",
            "content": (
                "Reminder: use <think> for reasoning, <code> for tool calls, and put the final solution in <answer>."
            ),
        }
    )
    return messages, prompt_input


def parse_exception(error_text: str) -> tuple[str, str]:
    text = (error_text or "").strip()
    if not text:
        return "", ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", ""
    tail = lines[-1]
    if ":" in tail:
        exc_type, exc_msg = tail.split(":", 1)
        return exc_type.strip(), exc_msg.strip()
    return "RuntimeError", tail


def turn_strict_valid(text: str) -> bool:
    if not STRICT_TURN_RE.match(text.strip()):
        return False
    thinks = THINK_TAG_RE.findall(text)
    codes = CODE_TAG_RE.findall(text)
    answers = ANSWER_TAG_RE.findall(text)
    if len(thinks) != 1:
        return False
    return (len(codes) == 1 and len(answers) == 0) or (len(codes) == 0 and len(answers) == 1)


def trace_strict_valid(text: str) -> bool:
    if not STRICT_TRACE_RE.match(text.strip()):
        return False

    stack: list[str] = []
    closed_order: list[str] = []
    for match in TAG_PATTERN.finditer(text):
        token = match.group(0)
        tag = match.group(1)
        if token.startswith("</"):
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
            closed_order.append(tag)
        else:
            if stack:
                return False
            stack.append(tag)

    if stack or not closed_order:
        return False
    if closed_order[-1] != "answer":
        return False

    idx = 0
    while idx + 2 < len(closed_order):
        if closed_order[idx : idx + 3] == ["think", "code", "result"]:
            idx += 3
            continue
        break
    tail = closed_order[idx:]
    return tail in (["think", "answer"], ["answer"])


def extract_last_answer(text: str) -> str | None:
    matches = ANSWER_TAG_RE.findall(text)
    if not matches:
        return None
    return strip_code_fence(matches[-1].strip())


def extract_last_think(text: str) -> str | None:
    matches = THINK_TAG_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def trim_for_protocol(text: str) -> str:
    text = text.strip()
    if "<think>" in text:
        text = text[text.find("<think>") :]
    return text


async def call_llm(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    network_err_types = (
        aiohttp.ClientConnectorError,
        aiohttp.ClientOSError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientConnectionError,
    )
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.post(url, headers=headers, json=payload, timeout=req_timeout) as resp:
                text = await resp.text()
                if resp.status in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"llm_http_{resp.status}: {text[:300]}")
                resp.raise_for_status()
                data = json.loads(text)
                content = data["choices"][0]["message"]["content"]
                return str(content)
        except network_err_types as exc:
            last_err = exc
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise InfraNetworkError(f"llm_network_error: {type(last_err).__name__}: {last_err}")
        except asyncio.TimeoutError as exc:
            last_err = exc
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise InfraNetworkError(f"llm_timeout_error: {type(last_err).__name__}: {last_err}")
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM request failed after retries: {last_err}")

    raise RuntimeError("LLM request failed")


async def call_compile_server(
    session: aiohttp.ClientSession,
    compile_server_url: str,
    code: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    url = compile_server_url.rstrip("/") + "/run"
    payload = {
        "raw_text": code,
        "extract_code": False,
        "timeout": timeout,
        "auto_invoke": False,
    }

    network_err_types = (
        aiohttp.ClientConnectorError,
        aiohttp.ClientOSError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientConnectionError,
    )
    started = time.perf_counter()
    last_network_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req_timeout = aiohttp.ClientTimeout(total=timeout + 3)
            async with session.post(url, json=payload, timeout=req_timeout) as resp:
                text = await resp.text()
                if resp.status == 429:
                    return {
                        "status": "busy",
                        "error": "compile_server_busy",
                        "stdout": "",
                        "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    }
                resp.raise_for_status()
                data = json.loads(text)
                data["wall_time_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                return data
        except network_err_types as exc:
            last_network_err = exc
            if attempt < retries:
                await asyncio.sleep(0.3 * (attempt + 1))
                continue
            raise InfraNetworkError(
                f"compile_server_network_error: {type(last_network_err).__name__}: {last_network_err}"
            )
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "error": f"compile_server_timeout({timeout}s)",
                "stdout": "",
                "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        except Exception as exc:
            return {
                "status": "system_error",
                "error": f"compile_server_error: {type(exc).__name__}: {exc}",
                "stdout": "",
                "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }

    raise InfraNetworkError("compile_server_network_error: unknown")


def to_result_block(tool_response: dict[str, Any]) -> dict[str, Any]:
    status = str(tool_response.get("status", ""))
    ok = status == "success"
    error = str(tool_response.get("error", "") or "")
    exc_type, exc_msg = parse_exception(error)

    return {
        "ok": ok,
        "status": status,
        "exception_type": exc_type,
        "exception_msg": clip_text(exc_msg, 500),
        "stdout_snippet": clip_text(str(tool_response.get("stdout", "") or ""), 4096),
        "stderr_snippet": "",
        "wall_time_ms": float(tool_response.get("wall_time_ms", 0.0) or 0.0),
    }


def inject_results_after_codes(text: str, result_blocks: list[dict[str, Any]]) -> str:
    matches = list(CODE_TAG_RE.finditer(text))
    if not matches:
        return text

    parts: list[str] = []
    cursor = 0
    for idx, match in enumerate(matches):
        parts.append(text[cursor : match.end()])
        if idx < len(result_blocks):
            parts.append(f"<result>{json.dumps(result_blocks[idx], ensure_ascii=False)}</result>")
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts)


def build_final_judge_program(answer_code: str, test_cases: list[str]) -> str:
    tests_blob = "\n".join(test_cases)
    return f"{answer_code}\n\n{tests_blob}\n"


def build_canonical_trace(code_steps: list[dict[str, Any]], answer_think: str, answer_code: str) -> str:
    def _sanitize_think(x: str) -> str:
        text = str(x)
        for tag in ("think", "code", "result", "answer"):
            text = text.replace(f"<{tag}>", f"[{tag}]").replace(f"</{tag}>", f"[/{tag}]")
        return text

    chunks: list[str] = []
    for step in code_steps:
        think = _sanitize_think(step.get("think", "Check and refine the solution."))
        code = step.get("code", "")
        result_payload = step.get("result", {})
        chunks.append(
            f"<think>{think}</think>"
            f"<code>{code}</code>"
            f"<result>{json.dumps(result_payload, ensure_ascii=False)}</result>"
        )
    chunks.append(f"<think>{_sanitize_think(answer_think)}</think><answer>{answer_code}</answer>")
    return "\n".join(chunks)


async def generate_attempt(
    row: dict[str, Any],
    args: argparse.Namespace,
    llm_session: aiohttp.ClientSession,
    tool_session: aiohttp.ClientSession,
    retry_format: bool = False,
) -> AttemptResult:
    test_cases = normalize_test_cases(row.get("test_cases"))
    if not test_cases:
        return AttemptResult(
            ok=False,
            judge_pass=False,
            answer_code="",
            raw_trace="",
            canonical_trace="",
            raw_format_valid=False,
            num_tool_calls=0,
            failure_reason="missing_test_cases",
        )

    messages, _ = build_prompt_messages(row, retry_format=retry_format)

    raw_trace_parts: list[str] = []
    code_steps: list[dict[str, Any]] = []
    tool_status_counts: Counter = Counter()

    invalid_turns = 0
    final_answer_code = ""
    final_answer_think = "Finalize and provide the correct solution."

    for _ in range(args.max_steps):
        assistant_raw = await call_llm(
            session=llm_session,
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.llm_timeout,
            retries=args.llm_retries,
        )

        assistant_text = trim_for_protocol(assistant_raw)
        assistant_text = RESULT_TAG_RE.sub("", assistant_text).strip()

        think = extract_last_think(assistant_text)
        codes = [strip_code_fence(x.strip()) for x in CODE_TAG_RE.findall(assistant_text)]
        answers = [strip_code_fence(x.strip()) for x in ANSWER_TAG_RE.findall(assistant_text)]

        strict_turn = turn_strict_valid(assistant_text)

        if think is None or (not codes and not answers):
            invalid_turns += 1
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": INVALID_TURN_FEEDBACK})
            raw_trace_parts.append(assistant_text)
            if invalid_turns >= 3:
                return AttemptResult(
                    ok=False,
                    judge_pass=False,
                    answer_code="",
                    raw_trace="\n".join(raw_trace_parts),
                    canonical_trace="",
                    raw_format_valid=False,
                    num_tool_calls=len(code_steps),
                    failure_reason="invalid_format_no_think",
                    tool_status_counts=tool_status_counts,
                )
            continue

        if codes:
            # If both <code> and <answer> appear in one turn, force it into a code-turn first.
            assistant_no_answer = ANSWER_TAG_RE.sub("", assistant_text)
            result_blocks: list[dict[str, Any]] = []
            result_tags: list[str] = []

            for code in codes:
                tool_resp = await call_compile_server(
                    session=tool_session,
                    compile_server_url=args.compile_server_url,
                    code=code,
                    timeout=args.tool_timeout,
                    retries=args.compile_retries,
                )
                status = str(tool_resp.get("status", ""))
                tool_status_counts[status] += 1
                result_payload = to_result_block(tool_resp)
                result_blocks.append(result_payload)
                result_tags.append(f"<result>{json.dumps(result_payload, ensure_ascii=False)}</result>")
                code_steps.append({"think": think, "code": code, "result": result_payload})

            raw_piece = inject_results_after_codes(assistant_no_answer, result_blocks)
            raw_trace_parts.append(raw_piece)

            feedback = TOOL_CONTINUE_FEEDBACK
            if not strict_turn:
                feedback = f"{INVALID_TURN_FEEDBACK} {TOOL_CONTINUE_FEEDBACK}"
            messages.append({"role": "assistant", "content": assistant_no_answer})
            messages.append({"role": "user", "content": "\n".join(result_tags + [feedback])})
            continue

        if answers:
            final_answer_code = answers[-1]
            final_answer_think = think
            raw_trace_parts.append(assistant_text)
            break

    if not final_answer_code.strip():
        return AttemptResult(
            ok=False,
            judge_pass=False,
            answer_code="",
            raw_trace="\n".join(raw_trace_parts),
            canonical_trace="",
            raw_format_valid=False,
            num_tool_calls=len(code_steps),
            failure_reason="missing_final_answer",
            tool_status_counts=tool_status_counts,
        )

    judge_program = build_final_judge_program(final_answer_code, test_cases)
    judge_resp = await call_compile_server(
        session=tool_session,
        compile_server_url=args.compile_server_url,
        code=judge_program,
        timeout=args.judge_timeout,
        retries=args.compile_retries,
    )
    judge_status = str(judge_resp.get("status", ""))
    tool_status_counts[f"judge_{judge_status}"] += 1

    raw_trace = "\n".join(part for part in raw_trace_parts if part and part.strip())
    canonical_trace = build_canonical_trace(code_steps, final_answer_think, final_answer_code)

    if judge_status != "success":
        return AttemptResult(
            ok=False,
            judge_pass=False,
            answer_code=final_answer_code,
            raw_trace=raw_trace,
            canonical_trace=canonical_trace,
            raw_format_valid=trace_strict_valid(raw_trace),
            num_tool_calls=len(code_steps),
            failure_reason="judge_failed",
            tool_status_counts=tool_status_counts,
        )

    return AttemptResult(
        ok=True,
        judge_pass=True,
        answer_code=final_answer_code,
        raw_trace=raw_trace,
        canonical_trace=canonical_trace,
        raw_format_valid=trace_strict_valid(raw_trace),
        num_tool_calls=len(code_steps),
        failure_reason="",
        tool_status_counts=tool_status_counts,
    )


async def ensure_compile_server(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    base = args.compile_server_url.rstrip("/")
    health_url = base + "/healthz"

    async def _healthy() -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=3.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as resp:
                    return resp.status == 200
        except Exception:
            return False

    if await _healthy():
        return None

    if not args.auto_start_compile_server:
        raise RuntimeError(
            f"Compile server is not reachable at {health_url}. "
            "Start it manually or pass --auto-start-compile-server."
        )

    parsed = urlparse(args.compile_server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 18080
    script_path = Path(args.compile_server_script)

    env = os.environ.copy()
    env["RUNPY_HOST"] = host
    env["RUNPY_PORT"] = str(port)
    env["RUNPY_MAX_WORKERS"] = str(args.runpy_max_workers)
    env["RUNPY_MAX_INFLIGHT"] = str(args.runpy_max_inflight)
    env["RUNPY_QUEUE_WAIT_TIMEOUT"] = str(args.runpy_queue_wait_timeout)
    env["RUNPY_WORKER_MAX_TASKS"] = str(args.runpy_worker_max_tasks)
    env.setdefault("RUNPY_TIMEOUT", str(max(args.tool_timeout, args.judge_timeout)))

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        preexec_fn=os.setsid,
    )

    for _ in range(40):
        if await _healthy():
            return proc
        await asyncio.sleep(0.5)

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    raise RuntimeError("Auto-started compile server did not become healthy in time.")


async def ensure_llm_connectivity(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    probe_url = base + "/models"
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    req_timeout = aiohttp.ClientTimeout(total=min(20.0, max(8.0, float(args.llm_timeout))))
    max_probe_attempts = max(3, int(args.llm_retries) + 1)

    last_err: Exception | None = None
    for attempt in range(max_probe_attempts):
        try:
            async with aiohttp.ClientSession(timeout=req_timeout) as session:
                async with session.get(probe_url, headers=headers) as resp:
                    _ = await resp.text()
                    if resp.status < 500:
                        return
                    raise RuntimeError(f"llm_probe_http_{resp.status}")
        except Exception as exc:
            last_err = exc
            if attempt < max_probe_attempts - 1:
                await asyncio.sleep(0.8 * (attempt + 1))

    raise InfraNetworkError(f"llm_probe_failed: {type(last_err).__name__}: {last_err}")


def stop_compile_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        return

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except Exception:
            return
        time.sleep(1.0)
        if proc.poll() is not None:
            return


def save_dataset_copy(dataset: Dataset, target_dir: Path, dataset_name: str, split_name: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = target_dir / "hf_dataset"
    parquet_path = target_dir / f"{dataset_slug(dataset_name)}_{split_name}.parquet"

    if not hf_dir.exists():
        dataset.save_to_disk(str(hf_dir))
    if not parquet_path.exists():
        dataset.to_parquet(str(parquet_path))


def load_dataset_local_first(
    target_dir: Path,
    dataset_name: str,
    split_name: str,
    cache_dir: Path,
) -> tuple[Dataset, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = target_dir / "hf_dataset"
    parquet_path = target_dir / f"{dataset_slug(dataset_name)}_{split_name}.parquet"

    if hf_dir.exists():
        restored = load_from_disk(str(hf_dir))
        if isinstance(restored, Dataset):
            return restored, "local_hf_dataset"
        if split_name in restored:
            return restored[split_name], f"local_hf_dataset[{split_name}]"
        first_split = next(iter(restored.keys()))
        return restored[first_split], f"local_hf_dataset[{first_split}]"

    if parquet_path.exists():
        return Dataset.from_parquet(str(parquet_path)), "local_parquet"

    dataset = load_dataset(dataset_name, split=split_name, cache_dir=str(cache_dir))
    return dataset, "remote_hf"


async def process_one_sample(
    row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    llm_session: aiohttp.ClientSession,
    tool_session: aiohttp.ClientSession,
) -> DistillRecord:
    sample_id = str(row.get("id") or f"sample_{row_index}")
    source = str(row.get("source") or "")
    question = str(row.get("question") or "")
    test_cases = normalize_test_cases(row.get("test_cases"))

    if not test_cases:
        return DistillRecord(False, {"id": sample_id, "row_index": row_index}, failure_reason="missing_test_cases")

    _messages, prompt_input = build_prompt_messages(row, retry_format=False)

    attempt1 = await generate_attempt(
        row=row,
        args=args,
        llm_session=llm_session,
        tool_session=tool_session,
        retry_format=False,
    )

    if not attempt1.ok:
        out = {
            "id": sample_id,
            "row_index": row_index,
            "num_tool_calls": attempt1.num_tool_calls,
            "failure_reason": attempt1.failure_reason,
            "tool_status_counts": dict(attempt1.tool_status_counts),
        }
        return DistillRecord(False, out, failure_reason=attempt1.failure_reason)

    selected = attempt1
    format_fix_type = "none"
    format_fix_applied = False
    format_retry_used = False
    format_fix_marker = ""

    if not attempt1.raw_format_valid:
        format_retry_used = True
        attempt2 = await generate_attempt(
            row=row,
            args=args,
            llm_session=llm_session,
            tool_session=tool_session,
            retry_format=True,
        )

        if attempt2.ok and attempt2.raw_format_valid:
            selected = attempt2
            format_fix_type = "retry"
            format_fix_applied = True
            format_fix_marker = "[FORMAT_FIXED_RETRY]"
        else:
            selected = attempt2 if attempt2.ok else attempt1
            format_fix_type = "manual"
            format_fix_applied = True
            format_fix_marker = "[FORMAT_FIXED_MANUAL]"

    if format_fix_type == "manual":
        trace = selected.canonical_trace
    else:
        trace = selected.raw_trace

    if not trace_strict_valid(trace):
        # Final safety net: force canonical strict trace.
        trace = selected.canonical_trace
        format_fix_type = "manual"
        format_fix_applied = True
        format_fix_marker = "[FORMAT_FIXED_MANUAL]"

    if not trace_strict_valid(trace):
        out = {
            "id": sample_id,
            "row_index": row_index,
            "num_tool_calls": selected.num_tool_calls,
            "failure_reason": "invalid_trace_sequence",
            "tool_status_counts": dict(selected.tool_status_counts),
        }
        return DistillRecord(False, out, failure_reason="invalid_trace_sequence")

    test_hash = hashlib.sha256("\n".join(test_cases).encode("utf-8")).hexdigest()

    output = {
        "id": sample_id,
        "source": source,
        "question": question,
        "prompt_input": prompt_input,
        "trace": trace,
        "answer_code": selected.answer_code,
        "num_tool_calls": selected.num_tool_calls,
        "final_pass_all": True,
        "test_case_count": len(test_cases),
        "test_cases_hash": test_hash,
        "train_dataset": args.train_dataset,
        "train_split": args.train_split,
        "model_name": args.model_name,
        "compile_server_url": args.compile_server_url,
        "row_index": row_index,
        "format_fix_applied": format_fix_applied,
        "format_fix_type": format_fix_type,
        "format_retry_used": format_retry_used,
        "format_fix_marker": format_fix_marker,
        "raw_format_valid_initial": attempt1.raw_format_valid,
        "tool_status_counts": dict(selected.tool_status_counts),
    }
    return DistillRecord(True, output)


def choose_indices(total_rows: int, args: argparse.Namespace) -> list[int]:
    indices = list(range(total_rows))
    if args.shuffle:
        rng = random.Random(args.sample_seed)
        rng.shuffle(indices)

    if args.sample_offset > 0:
        indices = indices[args.sample_offset :]

    if args.max_samples and args.max_samples > 0:
        indices = indices[: args.max_samples]

    return indices


async def run_distillation(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_cache = train_dir / ".hf_cache"
    test_cache = test_dir / ".hf_cache"

    print(f"[distill] Loading train dataset: {args.train_dataset}[{args.train_split}]")
    train_dataset, train_source = load_dataset_local_first(
        target_dir=train_dir,
        dataset_name=args.train_dataset,
        split_name=args.train_split,
        cache_dir=train_cache,
    )
    print(f"[distill] Train rows: {len(train_dataset)} (source={train_source})")

    print(f"[distill] Loading test dataset: {args.test_dataset}[{args.test_split}]")
    test_dataset, test_source = load_dataset_local_first(
        target_dir=test_dir,
        dataset_name=args.test_dataset,
        split_name=args.test_split,
        cache_dir=test_cache,
    )
    print(f"[distill] Test rows: {len(test_dataset)} (source={test_source})")

    save_dataset_copy(train_dataset, train_dir, args.train_dataset, args.train_split)
    save_dataset_copy(test_dataset, test_dir, args.test_dataset, args.test_split)

    if args.download_only:
        print("[distill] Download-only mode finished.")
        return

    if not args.api_key:
        raise ValueError("Missing API key. Set JUDGE_SET_3_API_KEY in env or pass --api-key explicitly.")

    indices = choose_indices(len(train_dataset), args)
    target_total = len(indices)

    print(
        f"[distill] Distilling target={target_total} "
        f"concurrency={args.sample_concurrency} offset={args.sample_offset} "
        f"time_budget_sec={args.time_budget_sec}"
    )

    out_dir = train_dir / "cold_start"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = f"{args.output_prefix}_{max(args.max_samples, 0)}_{timestamp}"
    passed_jsonl = out_dir / f"{prefix}.passed.jsonl"
    failed_jsonl = out_dir / f"{prefix}.failed.jsonl"
    passed_parquet = out_dir / f"{prefix}.passed.parquet"
    metrics_json = out_dir / f"{prefix}.metrics.json"

    server_proc = await ensure_compile_server(args)
    await ensure_llm_connectivity(args)

    passed_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []

    counters = {
        "processed": 0,
        "passed": 0,
        "failed": 0,
        "correct_modified": 0,
        "correct_modified_manual": 0,
        "correct_modified_retry": 0,
        "infra_exceptions": 0,
        "infra_network_errors": 0,
        "network_contaminated_filtered": 0,
    }
    failure_reason_counts: Counter = Counter()
    tool_status_agg: Counter = Counter()

    cursor = 0
    cursor_lock = asyncio.Lock()
    start_ts = time.perf_counter()
    abort_due_network_error = False
    abort_reason = ""

    llm_timeout = aiohttp.ClientTimeout(total=args.llm_timeout + 5)
    tool_timeout = aiohttp.ClientTimeout(total=max(args.judge_timeout, args.tool_timeout) + 10)

    try:
        async with aiohttp.ClientSession(timeout=llm_timeout) as llm_session, aiohttp.ClientSession(
            timeout=tool_timeout
        ) as tool_session:

            async def worker() -> None:
                nonlocal cursor
                nonlocal abort_due_network_error, abort_reason
                while True:
                    if abort_due_network_error:
                        return
                    if args.time_budget_sec > 0 and (time.perf_counter() - start_ts) >= args.time_budget_sec:
                        return

                    async with cursor_lock:
                        if cursor >= len(indices):
                            return
                        row_index = indices[cursor]
                        cursor += 1

                    row = train_dataset[row_index]
                    try:
                        result = await asyncio.wait_for(
                            process_one_sample(
                                row=row,
                                row_index=row_index,
                                args=args,
                                llm_session=llm_session,
                                tool_session=tool_session,
                            ),
                            timeout=float(args.sample_timeout_sec),
                        )
                    except InfraNetworkError as exc:
                        abort_due_network_error = True
                        abort_reason = f"{type(exc).__name__}: {exc}"
                        counters["infra_network_errors"] += 1
                        return
                    except asyncio.TimeoutError:
                        counters["processed"] += 1
                        counters["failed"] += 1
                        failure_reason_counts["sample_timeout"] += 1
                        rec = {
                            "id": str(row.get("id") or f"sample_{row_index}"),
                            "row_index": row_index,
                            "failure_reason": "sample_timeout",
                        }
                        failed_records.append(rec)
                        if args.save_failures:
                            with failed_jsonl.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue
                    except Exception as exc:
                        counters["processed"] += 1
                        counters["failed"] += 1
                        counters["infra_exceptions"] += 1
                        failure_reason_counts["exception"] += 1
                        rec = {
                            "id": str(row.get("id") or f"sample_{row_index}"),
                            "row_index": row_index,
                            "failure_reason": "exception",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        rec = sanitize_network_error_text(rec)
                        failed_records.append(rec)
                        if args.save_failures:
                            with failed_jsonl.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue

                    counters["processed"] += 1
                    tool_counts = result.output.get("tool_status_counts", {})
                    if isinstance(tool_counts, dict):
                        tool_status_agg.update({k: int(v) for k, v in tool_counts.items()})

                    if result.passed:
                        if trace_has_network_error_text(
                            str(result.output.get("trace", "")),
                            str(result.output.get("answer_code", "")),
                        ):
                            counters["failed"] += 1
                            counters["network_contaminated_filtered"] += 1
                            failure_reason_counts["network_text_contaminated"] += 1
                            rec = {
                                "id": str(result.output.get("id") or f"sample_{row_index}"),
                                "row_index": row_index,
                                "failure_reason": "network_text_contaminated",
                                "num_tool_calls": int(result.output.get("num_tool_calls", 0) or 0),
                            }
                            failed_records.append(rec)
                            if args.save_failures:
                                with failed_jsonl.open("a", encoding="utf-8") as f:
                                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            continue

                        counters["passed"] += 1
                        clean_output = sanitize_network_error_text(result.output)
                        if result.output.get("format_fix_applied"):
                            counters["correct_modified"] += 1
                            if result.output.get("format_fix_type") == "manual":
                                counters["correct_modified_manual"] += 1
                            elif result.output.get("format_fix_type") == "retry":
                                counters["correct_modified_retry"] += 1

                        passed_records.append(clean_output)
                        with passed_jsonl.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(clean_output, ensure_ascii=False) + "\n")
                    else:
                        counters["failed"] += 1
                        reason = result.failure_reason or "unknown"
                        failure_reason_counts[reason] += 1
                        rec = sanitize_network_error_text(dict(result.output))
                        rec["failure_reason"] = reason
                        failed_records.append(rec)
                        if args.save_failures:
                            with failed_jsonl.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            workers = [asyncio.create_task(worker()) for _ in range(max(1, args.sample_concurrency))]
            await asyncio.gather(*workers)

        elapsed = max(1e-9, time.perf_counter() - start_ts)
        processed = counters["processed"]
        throughput = processed / elapsed

        remaining = max(0, target_total - processed)
        eta_finish = None
        eta_seconds = None
        if throughput > 0 and target_total > 0:
            eta_seconds = remaining / throughput
            eta_finish = datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)

        if passed_records:
            pd.DataFrame(passed_records).to_parquet(passed_parquet, index=False)
        else:
            pd.DataFrame(columns=[
                "id",
                "source",
                "question",
                "prompt_input",
                "trace",
                "answer_code",
                "num_tool_calls",
                "final_pass_all",
                "test_case_count",
                "test_cases_hash",
                "train_dataset",
                "train_split",
                "model_name",
                "compile_server_url",
                "row_index",
                "format_fix_applied",
                "format_fix_type",
                "format_retry_used",
                "format_fix_marker",
                "raw_format_valid_initial",
                "tool_status_counts",
            ]).to_parquet(passed_parquet, index=False)

        metrics = {
            "timestamp": timestamp,
            "train_dataset": args.train_dataset,
            "train_split": args.train_split,
            "target_total": target_total,
            "processed_total": processed,
            "correct_total": counters["passed"],
            "correct_modified_total": counters["correct_modified"],
            "correct_modified_manual": counters["correct_modified_manual"],
            "correct_modified_retry": counters["correct_modified_retry"],
            "wrong_total": counters["failed"],
            "sample_seed": args.sample_seed,
            "sample_offset": args.sample_offset,
            "max_samples": args.max_samples,
            "time_budget_sec": args.time_budget_sec,
            "model_name": args.model_name,
            "sample_concurrency": args.sample_concurrency,
            "elapsed_seconds": elapsed,
            "throughput_samples_per_sec": throughput,
            "infra_exception_count": counters["infra_exceptions"],
            "infra_network_error_count": counters["infra_network_errors"],
            "network_contaminated_filtered": counters["network_contaminated_filtered"],
            "aborted_due_network_error": abort_due_network_error,
            "abort_reason": abort_reason if abort_due_network_error else None,
            "failure_reason_counts": dict(failure_reason_counts),
            "tool_status_counts": dict(tool_status_agg),
            "estimated_remaining_seconds": eta_seconds,
            "estimated_finish_at_utc": eta_finish.isoformat() if eta_finish is not None else None,
            "output_passed_jsonl": str(passed_jsonl),
            "output_passed_parquet": str(passed_parquet),
            "output_failed_jsonl": str(failed_jsonl) if args.save_failures else None,
        }
        metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[distill] Finished")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        stop_compile_server(server_proc)


def main() -> None:
    args = parse_args()
    asyncio.run(run_distillation(args))


if __name__ == "__main__":
    main()
