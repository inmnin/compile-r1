#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
from datasets import Dataset, load_dataset

from acecode_tool_coldstart_common import (
    NO_TOOL_SYSTEM_PROMPT,
    PYTHON_EXEC_TOOL,
    StaticToolNeed,
    TOOL_SYSTEM_PROMPT,
    build_task_prompt,
    is_pure_python_solution,
    is_self_contained_tool_script,
    json_dumps_compact,
    normalize_test_cases,
    normalize_tool_code_for_compare,
    observation_from_run_result,
    score_static_tool_need,
    strip_single_markdown_fence,
)


DEFAULT_TRAIN_DATASET = "TIGER-Lab/AceCode-87K"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_MODEL_NAME = os.environ.get("JUDGE_SET_3_MODEL_NAME") or os.environ.get("JUDGE_SET_2_MODEL_NAME") or "deepseek-chat"
DEFAULT_BASE_URL = os.environ.get("JUDGE_SET_3_BASE_URL") or os.environ.get("JUDGE_SET_2_BASE_URL") or "https://api.deepseek.com/beta"
DEFAULT_API_KEY = os.environ.get("JUDGE_SET_3_API_KEY") or os.environ.get("JUDGE_SET_2_API_KEY") or ""


@dataclass
class ToolCallRecord:
    tool_call_id: str
    raw_arguments: str
    code: str
    valid_name: bool
    valid_json: bool
    self_contained: bool
    self_contained_reason: str
    execution_status: str
    execution_success: bool
    observation: dict[str, Any]


@dataclass
class RouteResult:
    path_name: str
    teacher_mode: str
    final_code: str = ""
    final_valid: bool = False
    final_invalid_reason: str = ""
    final_format_fix: str = ""
    judge: dict[str, Any] = field(default_factory=dict)
    passed_all: bool = False
    passed_count: int = 0
    total_count: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    conversation: list[dict[str, str]] = field(default_factory=list)
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    failed_reason: str = ""
    finish_reason: str = ""

    @property
    def valid_tool_calls(self) -> int:
        return sum(1 for x in self.tool_calls if x.valid_name and x.valid_json and x.self_contained)

    @property
    def successful_tool_calls(self) -> int:
        return sum(1 for x in self.tool_calls if x.execution_success)

    @property
    def tool_invocation_failures(self) -> int:
        return sum(
            1
            for x in self.tool_calls
            if (not x.valid_name)
            or (not x.valid_json)
            or (not x.self_contained)
            or x.execution_status in {"invalid_call", "invalid_script", "busy", "system_error"}
        )

    @property
    def unique_tool_codes(self) -> int:
        return len({normalize_tool_code_for_compare(x.code) for x in self.tool_calls if x.code.strip()})

    @property
    def observation_utilized(self) -> bool:
        return self.unique_tool_codes >= 2

    @property
    def has_tool_invocation_failure(self) -> bool:
        return self.tool_invocation_failures > 0


@dataclass
class DistillDecision:
    sample_id: str
    source: str
    question: str
    static_bucket: str
    static_score: int
    risk_flags: list[str]
    bucket: str
    bucket_reason: str
    selected_route: str
    raw_record: dict[str, Any] | None
    sharegpt_record: dict[str, Any] | None
    route_results: dict[str, Any]


class InfraNetworkError(RuntimeError):
    pass


class AsyncResultCache:
    def __init__(self) -> None:
        self._run_cache: dict[str, dict[str, Any]] = {}
        self._tests_cache: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get_run(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            value = self._run_cache.get(key)
            return None if value is None else dict(value)

    async def put_run(self, key: str, value: dict[str, Any]) -> None:
        async with self._lock:
            self._run_cache[key] = dict(value)

    async def get_tests(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            value = self._tests_cache.get(key)
            return None if value is None else dict(value)

    async def put_tests(self, key: str, value: dict[str, Any]) -> None:
        async with self._lock:
            self._tests_cache[key] = dict(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill tool-augmented ShareGPT cold-start data from AceCode with DeepSeek.")
    parser.add_argument("--data-root", type=str, default="examples/compile-r1/data")
    parser.add_argument("--train-dataset", type=str, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--train-split", type=str, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260310)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--direct-candidate-ratio", type=float, default=0.3)
    parser.add_argument("--tool-candidate-order", choices=["random", "hardfirst"], default="random")
    parser.add_argument("--time-budget-sec", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="examples/compile-r1/data/cold_start_v2")
    parser.add_argument("--output-prefix", type=str, default="acecode_tool_coldstart")
    parser.add_argument("--save-failures", action="store_true")

    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-turns", type=int, default=6)
    parser.add_argument("--sample-concurrency", type=int, default=24)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-retries", type=int, default=3)

    parser.add_argument("--compile-server-url", type=str, default="http://127.0.0.1:18080")
    parser.add_argument("--tool-timeout", type=int, default=12)
    parser.add_argument("--judge-timeout", type=int, default=20)
    parser.add_argument("--compile-retries", type=int, default=2)
    parser.add_argument("--auto-start-compile-server", action="store_true")
    parser.add_argument("--compile-server-script", type=str, default="tools/RunPythonTool.py")
    parser.add_argument("--runpy-max-workers", type=int, default=24)
    parser.add_argument("--runpy-max-inflight", type=int, default=96)
    parser.add_argument("--runpy-queue-wait-timeout", type=float, default=1.5)
    parser.add_argument("--runpy-worker-max-tasks", type=int, default=400)
    parser.add_argument("--sample-timeout-sec", type=int, default=600)
    return parser.parse_args()


def choose_indices(
    dataset: Dataset,
    args: argparse.Namespace,
) -> tuple[list[int], dict[int, StaticToolNeed], dict[str, int]]:
    static_cache: dict[int, StaticToolNeed] = {}
    direct_indices: list[int] = []
    tool_indices: list[int] = []
    for idx in range(len(dataset)):
        row = dataset[idx]
        static = score_static_tool_need(
            question=str(row.get("question") or ""),
            test_cases=normalize_test_cases(row.get("test_cases")),
            inferences=row.get("inferences"),
        )
        static_cache[idx] = static
        if static.bucket == "tool_candidate":
            tool_indices.append(idx)
        else:
            direct_indices.append(idx)
        if (idx + 1) % 10000 == 0:
            print(
                json.dumps(
                    {
                        "phase": "static_scoring",
                        "scored_rows": idx + 1,
                        "direct_candidate_count": len(direct_indices),
                        "tool_candidate_count": len(tool_indices),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    rng = random.Random(args.sample_seed)
    if args.shuffle:
        rng.shuffle(direct_indices)
        rng.shuffle(tool_indices)

    if args.tool_candidate_order == "hardfirst":
        tool_indices.sort(
            key=lambda idx: (
                static_cache[idx].score,
                -static_cache[idx].max_inference_pass_rate,
                static_cache[idx].num_tests,
            ),
            reverse=True,
        )
        direct_indices.sort(
            key=lambda idx: (
                static_cache[idx].score,
                static_cache[idx].max_inference_pass_rate,
                static_cache[idx].num_tests,
            )
        )

    direct_ratio = min(0.95, max(0.05, float(args.direct_candidate_ratio)))
    tool_slots = max(1, round((1.0 - direct_ratio) * 10))
    direct_slots = max(1, round(direct_ratio * 10))
    schedule = ["tool"] * tool_slots + ["direct"] * direct_slots
    rng.shuffle(schedule)

    mixed: list[int] = []
    while direct_indices or tool_indices:
        progressed = False
        for slot in schedule:
            if slot == "tool" and tool_indices:
                mixed.append(tool_indices.pop())
                progressed = True
            elif slot == "direct" and direct_indices:
                mixed.append(direct_indices.pop())
                progressed = True
        if not progressed:
            break

    if args.sample_offset > 0:
        mixed = mixed[args.sample_offset :]
    if args.max_samples and args.max_samples > 0:
        mixed = mixed[: args.max_samples]

    summary = {
        "direct_candidate_total": sum(1 for x in static_cache.values() if x.bucket == "direct_candidate"),
        "tool_candidate_total": sum(1 for x in static_cache.values() if x.bucket == "tool_candidate"),
        "direct_candidate_ratio": direct_ratio,
    }
    print(json.dumps({"phase": "static_scoring_done", **summary}, ensure_ascii=False), flush=True)
    return mixed, static_cache, summary


def dataset_slug(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def load_acecode_dataset(data_root: Path, dataset_name: str, split_name: str) -> Dataset:
    local_pattern = str((data_root / "train" / "data" / "*.parquet").resolve())
    train_parquet = data_root / "train.parquet"
    if (data_root / "train" / "data").exists():
        return load_dataset("parquet", data_files=local_pattern, split="train")
    if train_parquet.exists():
        return load_dataset("parquet", data_files=str(train_parquet.resolve()), split="train")
    cache_dir = data_root / "train" / ".cache" / "huggingface"
    return load_dataset(dataset_name, split=split_name, cache_dir=str(cache_dir))


async def call_chat_completion(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_completion_tokens: int,
    timeout: float,
    retries: int,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_completion_tokens,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
        payload["parallel_tool_calls"] = False

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/chat/completions"
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
                    raise RuntimeError(f"llm_http_{resp.status}: {text[:500]}")
                resp.raise_for_status()
                data = json.loads(text)
                choice = data["choices"][0]
                return {
                    "message": choice["message"],
                    "finish_reason": choice.get("finish_reason", ""),
                    "usage": data.get("usage", {}),
                }
        except network_err_types as exc:
            last_err = exc
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise InfraNetworkError(f"llm_network_error: {type(exc).__name__}: {exc}")
        except asyncio.TimeoutError as exc:
            last_err = exc
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise InfraNetworkError(f"llm_timeout_error: {type(exc).__name__}: {exc}")
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise RuntimeError(f"llm_request_failed: {type(last_err).__name__}: {last_err}")
    raise RuntimeError("llm_request_failed")


async def call_compile_server_run(
    session: aiohttp.ClientSession,
    cache: AsyncResultCache,
    *,
    compile_server_url: str,
    code: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    cache_key = hashlib.sha256(code.encode("utf-8")).hexdigest()
    cached = await cache.get_run(cache_key)
    if cached is not None:
        return cached

    url = compile_server_url.rstrip("/") + "/run"
    payload = {
        "raw_text": code,
        "extract_code": False,
        "timeout": timeout,
        "auto_invoke": False,
    }
    started = time.perf_counter()
    network_err_types = (
        aiohttp.ClientConnectorError,
        aiohttp.ClientOSError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientConnectionError,
    )

    for attempt in range(retries + 1):
        try:
            req_timeout = aiohttp.ClientTimeout(total=timeout + 4)
            async with session.post(url, json=payload, timeout=req_timeout) as resp:
                text = await resp.text()
                if resp.status == 429:
                    result = {
                        "status": "busy",
                        "error": "compile_server_busy",
                        "stdout": "",
                        "return_value": None,
                    }
                else:
                    resp.raise_for_status()
                    result = json.loads(text)
                result["wall_time_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                await cache.put_run(cache_key, result)
                return result
        except network_err_types as exc:
            if attempt < retries:
                await asyncio.sleep(0.4 * (attempt + 1))
                continue
            raise InfraNetworkError(f"compile_server_network_error: {type(exc).__name__}: {exc}")
        except asyncio.TimeoutError:
            result = {
                "status": "timeout",
                "error": f"compile_server_timeout({timeout}s)",
                "stdout": "",
                "return_value": None,
                "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            await cache.put_run(cache_key, result)
            return result
        except Exception as exc:
            result = {
                "status": "system_error",
                "error": f"compile_server_error: {type(exc).__name__}: {exc}",
                "stdout": "",
                "return_value": None,
                "wall_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            await cache.put_run(cache_key, result)
            return result
    raise RuntimeError("compile_server_run_failed")


async def call_compile_server_tests(
    session: aiohttp.ClientSession,
    cache: AsyncResultCache,
    *,
    compile_server_url: str,
    code: str,
    tests: list[str],
    timeout: int,
    retries: int,
    max_workers: int,
) -> dict[str, Any]:
    cache_material = code + "\n__TESTS__\n" + "\n".join(tests)
    cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()
    cached = await cache.get_tests(cache_key)
    if cached is not None:
        return cached

    url = compile_server_url.rstrip("/") + "/run_tests"
    payload = {"code": code, "tests": tests, "timeout": timeout, "max_workers": max_workers}
    network_err_types = (
        aiohttp.ClientConnectorError,
        aiohttp.ClientOSError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientConnectionError,
    )

    for attempt in range(retries + 1):
        try:
            req_timeout = aiohttp.ClientTimeout(total=timeout + 6)
            async with session.post(url, json=payload, timeout=req_timeout) as resp:
                text = await resp.text()
                resp.raise_for_status()
                result = json.loads(text)
                await cache.put_tests(cache_key, result)
                return result
        except network_err_types as exc:
            if attempt < retries:
                await asyncio.sleep(0.4 * (attempt + 1))
                continue
            raise InfraNetworkError(f"compile_server_network_error: {type(exc).__name__}: {exc}")
        except asyncio.TimeoutError:
            result = {
                "passed_all": False,
                "passed_count": 0,
                "total_count": len(tests),
                "status": "timeout",
                "error": f"compile_server_timeout({timeout}s)",
                "traceback": "",
                "stdout": "",
                "stderr": "",
                "failed_test_index": None,
            }
            await cache.put_tests(cache_key, result)
            return result
        except Exception as exc:
            result = {
                "passed_all": False,
                "passed_count": 0,
                "total_count": len(tests),
                "status": "runtime_error",
                "error": f"compile_server_error: {type(exc).__name__}: {exc}",
                "traceback": "",
                "stdout": "",
                "stderr": "",
                "failed_test_index": None,
            }
            await cache.put_tests(cache_key, result)
            return result
    raise RuntimeError("compile_server_tests_failed")


def route_payload_to_sharegpt(sample_id: str, source: str, question: str, bucket: str, route: RouteResult) -> dict[str, Any]:
    return {
        "id": sample_id,
        "source": source,
        "bucket": bucket,
        "system": TOOL_SYSTEM_PROMPT,
        "tools": [PYTHON_EXEC_TOOL],
        "messages": route.raw_messages,
        "conversations": route.conversation,
    }


def route_pass_ratio(route: RouteResult | None) -> float | None:
    if route is None:
        return None
    total = max(0, int(route.total_count or 0))
    if total <= 0:
        return None
    return float(route.passed_count) / float(total)


def route_payload_to_raw_record(
    sample_id: str,
    source: str,
    question: str,
    bucket: str,
    selected_route: str,
    direct: RouteResult,
    auto: RouteResult,
    required: RouteResult | None,
) -> dict[str, Any]:
    selected_obj = {"direct": direct, "auto": auto, "required": required}.get(selected_route)
    messages = list((selected_obj.raw_messages if selected_obj is not None else []) or [])
    return {
        "id": sample_id,
        "bucket": bucket,
        "question": question,
        "source": source,
        "tools": [PYTHON_EXEC_TOOL],
        "messages": messages,
        "hidden_eval": {
            "pass_direct": route_pass_ratio(direct),
            "pass_auto": route_pass_ratio(auto),
            "pass_required": route_pass_ratio(required),
        },
    }


def parse_tool_call(tool_call: dict[str, Any]) -> tuple[bool, bool, str, dict[str, Any]]:
    name = str(tool_call.get("function", {}).get("name", ""))
    raw_arguments = tool_call.get("function", {}).get("arguments", "")
    valid_name = name == "python_exec"
    parsed_args: dict[str, Any] = {}
    valid_json = False
    try:
        if isinstance(raw_arguments, str):
            parsed_args = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            parsed_args = raw_arguments
        valid_json = isinstance(parsed_args, dict)
    except json.JSONDecodeError:
        valid_json = False
    code = str(parsed_args.get("code", "") if isinstance(parsed_args, dict) else "")
    return valid_name, valid_json, code, parsed_args


async def run_direct_route(
    *,
    question: str,
    tests: list[str],
    args: argparse.Namespace,
    llm_session: aiohttp.ClientSession,
    tool_session: aiohttp.ClientSession,
    cache: AsyncResultCache,
) -> RouteResult:
    route = RouteResult(path_name="direct", teacher_mode="none")
    user_prompt = build_task_prompt(question)
    route.conversation = [{"from": "human", "value": user_prompt}]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": NO_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    route.raw_messages = list(messages)

    response = await call_chat_completion(
        llm_session,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        messages=messages,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        timeout=args.llm_timeout,
        retries=args.llm_retries,
        tools=[PYTHON_EXEC_TOOL],
        tool_choice="none",
    )
    message = response["message"]
    route.finish_reason = str(response.get("finish_reason", ""))
    content = str(message.get("content", "") or "").strip()
    if message.get("tool_calls"):
        route.failed_reason = "tool_call_returned_in_none_mode"
        return route

    cleaned_code, stripped_fence = strip_single_markdown_fence(content)
    route.final_code = cleaned_code
    route.final_format_fix = "strip_markdown_fence" if stripped_fence else ""
    route.conversation.append({"from": "gpt", "value": cleaned_code})
    route.raw_messages.append({"role": "assistant", "content": cleaned_code})
    valid, reason = is_pure_python_solution(cleaned_code)
    route.final_valid = valid
    route.final_invalid_reason = reason
    if not valid:
        route.failed_reason = reason
        return route

    judge = await call_compile_server_tests(
        tool_session,
        cache,
        compile_server_url=args.compile_server_url,
        code=cleaned_code,
        tests=tests,
        timeout=args.judge_timeout,
        retries=args.compile_retries,
        max_workers=args.runpy_max_workers,
    )
    route.judge = judge
    route.passed_all = bool(judge.get("passed_all", False))
    route.passed_count = int(judge.get("passed_count", 0) or 0)
    route.total_count = int(judge.get("total_count", len(tests)) or len(tests))
    if not route.passed_all:
        route.failed_reason = "hidden_tests_failed"
    return route


async def run_tool_route(
    *,
    question: str,
    tests: list[str],
    args: argparse.Namespace,
    llm_session: aiohttp.ClientSession,
    tool_session: aiohttp.ClientSession,
    cache: AsyncResultCache,
    path_name: str,
    initial_tool_choice: Any,
) -> RouteResult:
    route = RouteResult(path_name=path_name, teacher_mode=str(initial_tool_choice))
    user_prompt = build_task_prompt(question)
    route.conversation = [{"from": "human", "value": user_prompt}]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    route.raw_messages = list(messages)

    current_tool_choice = initial_tool_choice
    for _turn in range(args.max_tool_turns):
        response = await call_chat_completion(
            llm_session,
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            messages=messages,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            timeout=args.llm_timeout,
            retries=args.llm_retries,
            tools=[PYTHON_EXEC_TOOL],
            tool_choice=current_tool_choice,
        )
        current_tool_choice = "auto"
        route.finish_reason = str(response.get("finish_reason", ""))
        message = response["message"]
        content = str(message.get("content", "") or "")
        tool_calls = message.get("tool_calls") or []
        route.raw_messages.append(message)

        if tool_calls:
            if content.strip():
                route.failed_reason = "tool_call_turn_contains_text"
                return route
            assistant_tool_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_tool_message)
            route.raw_messages.append(dict(assistant_tool_message))
            for tool_call in tool_calls:
                valid_name, valid_json, code, _parsed_args = parse_tool_call(tool_call)
                self_contained, self_contained_reason = is_self_contained_tool_script(code)
                if not valid_name or not valid_json:
                    route.tool_calls.append(
                        ToolCallRecord(
                            tool_call_id=str(tool_call.get("id", "")),
                            raw_arguments=str(tool_call.get("function", {}).get("arguments", "")),
                            code=code,
                            valid_name=valid_name,
                            valid_json=valid_json,
                            self_contained=self_contained,
                            self_contained_reason=self_contained_reason,
                            execution_status="invalid_call",
                            execution_success=False,
                            observation={"status": "invalid_call", "stdout": "", "error": "invalid function call", "return_value": None},
                        )
                    )
                    route.failed_reason = "invalid_tool_call_payload"
                    return route
                if not self_contained:
                    route.tool_calls.append(
                        ToolCallRecord(
                            tool_call_id=str(tool_call.get("id", "")),
                            raw_arguments=str(tool_call.get("function", {}).get("arguments", "")),
                            code=code,
                            valid_name=valid_name,
                            valid_json=valid_json,
                            self_contained=self_contained,
                            self_contained_reason=self_contained_reason,
                            execution_status="invalid_script",
                            execution_success=False,
                            observation={"status": "invalid_script", "stdout": "", "error": self_contained_reason, "return_value": None},
                        )
                    )
                    route.failed_reason = "tool_code_not_self_contained"
                    return route

                tool_result = await call_compile_server_run(
                    tool_session,
                    cache,
                    compile_server_url=args.compile_server_url,
                    code=code,
                    timeout=args.tool_timeout,
                    retries=args.compile_retries,
                )
                observation = observation_from_run_result(tool_result)
                record = ToolCallRecord(
                    tool_call_id=str(tool_call.get("id", "")),
                    raw_arguments=str(tool_call.get("function", {}).get("arguments", "")),
                    code=code,
                    valid_name=valid_name,
                    valid_json=valid_json,
                    self_contained=self_contained,
                    self_contained_reason=self_contained_reason,
                    execution_status=str(tool_result.get("status", "")),
                    execution_success=str(tool_result.get("status", "")) == "success",
                    observation=observation,
                )
                route.tool_calls.append(record)
                route.conversation.append(
                    {
                        "from": "function_call",
                        "value": json_dumps_compact({"name": "python_exec", "arguments": {"code": code}}),
                    }
                )
                route.conversation.append({"from": "observation", "value": json_dumps_compact(observation)})
                tool_message = {
                    "role": "tool",
                    "tool_call_id": str(tool_call.get("id", "")),
                    "content": json_dumps_compact(observation),
                }
                messages.append(tool_message)
                route.raw_messages.append(dict(tool_message))
            continue

        final_code, stripped_fence = strip_single_markdown_fence(content.strip())
        route.final_code = final_code
        route.final_format_fix = "strip_markdown_fence" if stripped_fence else ""
        route.conversation.append({"from": "gpt", "value": final_code})
        route.raw_messages.append({"role": "assistant", "content": final_code})
        valid, reason = is_pure_python_solution(final_code)
        route.final_valid = valid
        route.final_invalid_reason = reason
        if not valid:
            route.failed_reason = reason
            return route

        judge = await call_compile_server_tests(
            tool_session,
            cache,
            compile_server_url=args.compile_server_url,
            code=final_code,
            tests=tests,
            timeout=args.judge_timeout,
            retries=args.compile_retries,
            max_workers=args.runpy_max_workers,
        )
        route.judge = judge
        route.passed_all = bool(judge.get("passed_all", False))
        route.passed_count = int(judge.get("passed_count", 0) or 0)
        route.total_count = int(judge.get("total_count", len(tests)) or len(tests))
        if not route.passed_all:
            route.failed_reason = "hidden_tests_failed"
        return route

    route.failed_reason = "max_tool_turns_exhausted"
    return route


def choose_bucket(static_bucket: str, direct: RouteResult, auto: RouteResult, required: RouteResult | None) -> tuple[str, str, RouteResult | None]:
    candidate_tool_routes = [
        route
        for route in (auto, required)
        if route is not None and route.passed_all and route.valid_tool_calls >= 1 and not route.has_tool_invocation_failure
    ]
    best_tool_route = None
    if candidate_tool_routes:
        best_tool_route = max(
            candidate_tool_routes,
            key=lambda x: (x.passed_count, x.valid_tool_calls, x.unique_tool_codes, x.successful_tool_calls),
        )

    if direct.passed_all and (best_tool_route is None or best_tool_route.passed_count <= direct.passed_count):
        return "direct", "direct_passed_and_tool_not_better", direct

    if best_tool_route is None:
        return "discard", "no_valid_passing_route", None

    if best_tool_route.valid_tool_calls >= 2 and best_tool_route.observation_utilized:
        return "repair", "multi_turn_tool_route_passed_hidden_tests", best_tool_route

    if best_tool_route.valid_tool_calls >= 1:
        if static_bucket == "direct_candidate" and direct.passed_all:
            return "direct", "tool_route_not_needed_for_direct_candidate", direct
        return "single_tool", "tool_route_improved_or_rescued_sample", best_tool_route

    return "discard", "tool_route_missing_valid_tool_use", None


def route_to_dict(route: RouteResult | None) -> dict[str, Any] | None:
    if route is None:
        return None
    payload = asdict(route)
    payload["observation_utilized"] = route.observation_utilized
    payload["valid_tool_calls"] = route.valid_tool_calls
    payload["successful_tool_calls"] = route.successful_tool_calls
    payload["tool_invocation_failures"] = route.tool_invocation_failures
    payload["has_tool_invocation_failure"] = route.has_tool_invocation_failure
    payload["unique_tool_codes"] = route.unique_tool_codes
    return payload


async def process_one_sample(
    *,
    row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    llm_session: aiohttp.ClientSession,
    tool_session: aiohttp.ClientSession,
    cache: AsyncResultCache,
    static: StaticToolNeed,
) -> DistillDecision:
    sample_id = str(row.get("id") or f"row_{row_index}")
    source = str(row.get("source") or "")
    question = str(row.get("question") or "").strip()
    tests = normalize_test_cases(row.get("test_cases"))
    if not question:
        raise RuntimeError("missing_question")
    if not tests:
        raise RuntimeError("missing_test_cases")

    direct = await run_direct_route(
        question=question,
        tests=tests,
        args=args,
        llm_session=llm_session,
        tool_session=tool_session,
        cache=cache,
    )
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

    required: RouteResult | None = None
    if static.bucket == "tool_candidate" and (auto.valid_tool_calls == 0 or auto.has_tool_invocation_failure):
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

    bucket, bucket_reason, selected = choose_bucket(static.bucket, direct, auto, required)
    sharegpt_record = None
    raw_record = None
    if selected is not None and bucket != "discard":
        raw_record = route_payload_to_raw_record(sample_id, source, question, bucket, selected.path_name, direct, auto, required)
        sharegpt_record = route_payload_to_sharegpt(sample_id, source, question, bucket, selected)

    return DistillDecision(
        sample_id=sample_id,
        source=source,
        question=question,
        static_bucket=static.bucket,
        static_score=static.score,
        risk_flags=static.risk_flags,
        bucket=bucket,
        bucket_reason=bucket_reason,
        selected_route=selected.path_name if selected is not None else "",
        raw_record=raw_record,
        sharegpt_record=sharegpt_record,
        route_results={
            "direct": route_to_dict(direct),
            "auto": route_to_dict(auto),
            "required": route_to_dict(required),
        },
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
        raise RuntimeError(f"Compile server is not reachable at {health_url}. Start it or pass --auto-start-compile-server.")

    parsed = urlparse(args.compile_server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 18080
    env = os.environ.copy()
    env["RUNPY_HOST"] = host
    env["RUNPY_PORT"] = str(port)
    env["RUNPY_MAX_WORKERS"] = str(args.runpy_max_workers)
    env["RUNPY_MAX_INFLIGHT"] = str(args.runpy_max_inflight)
    env["RUNPY_QUEUE_WAIT_TIMEOUT"] = str(args.runpy_queue_wait_timeout)
    env["RUNPY_WORKER_MAX_TASKS"] = str(args.runpy_worker_max_tasks)
    env["RUNPY_TIMEOUT"] = str(max(args.tool_timeout, args.judge_timeout))

    proc = subprocess.Popen(
        [sys.executable, str(Path(args.compile_server_script))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        preexec_fn=os.setsid,
    )
    for _ in range(60):
        if await _healthy():
            return proc
        await asyncio.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    raise RuntimeError("auto-started compile server did not become healthy in time")


async def ensure_llm_connectivity(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    probe_url = base + "/models"
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=20.0)
    last_err: Exception | None = None
    for attempt in range(max(3, args.llm_retries + 1)):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(probe_url, headers=headers) as resp:
                    _ = await resp.text()
                    if resp.status < 500:
                        return
                    raise RuntimeError(f"llm_probe_http_{resp.status}")
        except Exception as exc:
            last_err = exc
            if attempt < max(2, args.llm_retries):
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


async def run_distillation(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("Missing API key. Set JUDGE_SET_3_API_KEY / JUDGE_SET_2_API_KEY or pass --api-key.")

    data_root = Path(args.data_root)
    dataset = load_acecode_dataset(data_root, args.train_dataset, args.train_split)
    indices, static_cache, static_summary = choose_indices(dataset, args)
    target_total = len(indices)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = f"{args.output_prefix}_{target_total}_{timestamp}"
    passed_jsonl = output_dir / f"{prefix}.passed.jsonl"
    failed_jsonl = output_dir / f"{prefix}.failed.jsonl"
    metrics_json = output_dir / f"{prefix}.metrics.json"

    server_proc = await ensure_compile_server(args)
    await ensure_llm_connectivity(args)

    counters = Counter()
    bucket_counts = Counter()
    failure_reasons = Counter()
    start_ts = time.perf_counter()
    cursor = 0
    cursor_lock = asyncio.Lock()
    print_lock = asyncio.Lock()
    cache = AsyncResultCache()

    llm_timeout = aiohttp.ClientTimeout(total=args.llm_timeout + 5)
    tool_timeout = aiohttp.ClientTimeout(total=max(args.tool_timeout, args.judge_timeout) + 10)

    try:
        async with aiohttp.ClientSession(timeout=llm_timeout) as llm_session, aiohttp.ClientSession(timeout=tool_timeout) as tool_session:

            async def log_progress(force: bool = False) -> None:
                processed = counters["processed"]
                if not force and processed % 20 != 0:
                    return
                elapsed = max(1e-9, time.perf_counter() - start_ts)
                throughput = processed / elapsed if processed else 0.0
                remaining = max(0, target_total - processed)
                eta = datetime.now(timezone.utc) + timedelta(seconds=remaining / throughput) if throughput > 0 else None
                async with print_lock:
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "target_total": target_total,
                                "passed": counters["passed"],
                                "failed": counters["failed"],
                                "discarded": bucket_counts["discard"],
                                "bucket_counts": dict(bucket_counts),
                                "throughput_samples_per_sec": round(throughput, 4),
                                "eta_utc": eta.isoformat() if eta else None,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

            async def worker() -> None:
                nonlocal cursor
                while True:
                    if args.time_budget_sec > 0 and (time.perf_counter() - start_ts) >= args.time_budget_sec:
                        return
                    async with cursor_lock:
                        if cursor >= len(indices):
                            return
                        row_index = indices[cursor]
                        cursor += 1
                    row = dataset[row_index]
                    sample_id = str(row.get("id") or f"row_{row_index}")
                    try:
                        result = await asyncio.wait_for(
                            process_one_sample(
                                row=row,
                                row_index=row_index,
                                args=args,
                                llm_session=llm_session,
                                tool_session=tool_session,
                                cache=cache,
                                static=static_cache[row_index],
                            ),
                            timeout=float(args.sample_timeout_sec),
                        )
                        counters["processed"] += 1
                        counters["passed"] += 1 if result.bucket != "discard" else 0
                        bucket_counts[result.bucket] += 1
                        payload = {
                            "id": result.sample_id,
                            "source": result.source,
                            "question": result.question,
                            "static_bucket": result.static_bucket,
                            "static_score": result.static_score,
                            "risk_flags": result.risk_flags,
                            "bucket": result.bucket,
                            "bucket_reason": result.bucket_reason,
                            "selected_route": result.selected_route,
                            "raw_record": result.raw_record,
                            "sharegpt_record": result.sharegpt_record,
                            "route_results": result.route_results,
                        }
                        with passed_jsonl.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        await log_progress()
                    except Exception as exc:
                        counters["processed"] += 1
                        counters["failed"] += 1
                        failure_reasons[type(exc).__name__] += 1
                        payload = {"id": sample_id, "row_index": row_index, "error": f"{type(exc).__name__}: {exc}"}
                        if args.save_failures:
                            with failed_jsonl.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        await log_progress()

            workers = [asyncio.create_task(worker()) for _ in range(max(1, args.sample_concurrency))]
            await asyncio.gather(*workers)

        elapsed = max(1e-9, time.perf_counter() - start_ts)
        throughput = counters["processed"] / elapsed if counters["processed"] else 0.0
        remaining = max(0, target_total - counters["processed"])
        eta_finish = datetime.now(timezone.utc) + timedelta(seconds=remaining / throughput) if throughput > 0 else None
        metrics = {
            "timestamp": timestamp,
            "target_total": target_total,
            "processed_total": counters["processed"],
            "passed_total": counters["passed"],
            "failed_total": counters["failed"],
            "bucket_counts": dict(bucket_counts),
            "failure_reasons": dict(failure_reasons),
            "sample_concurrency": args.sample_concurrency,
            "runpy_max_workers": args.runpy_max_workers,
            "runpy_max_inflight": args.runpy_max_inflight,
            "elapsed_seconds": elapsed,
            "throughput_samples_per_sec": throughput,
            "estimated_remaining_seconds": remaining / throughput if throughput > 0 else None,
            "estimated_finish_at_utc": eta_finish.isoformat() if eta_finish else None,
            "output_passed_jsonl": str(passed_jsonl),
            "output_failed_jsonl": str(failed_jsonl) if args.save_failures else None,
            "static_summary": static_summary,
        }
        metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    finally:
        stop_compile_server(server_proc)


def main() -> None:
    args = parse_args()
    asyncio.run(run_distillation(args))


if __name__ == "__main__":
    main()
