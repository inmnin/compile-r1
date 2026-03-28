import asyncio
import contextlib
import io
import inspect
import linecache
import logging
import multiprocessing
import os
import pickle
import re
import signal
import sys
import threading
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

current_path = os.getcwd()
if current_path not in sys.path:
    sys.path.append(current_path)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid int env %s=%r, fallback=%s", name, raw, default)
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        logger.warning("Invalid float env %s=%r, fallback=%s", name, raw, default)
        return default


def _detect_available_vcpu() -> int:
    candidates = []

    try:
        if hasattr(os, "sched_getaffinity"):
            candidates.append(len(os.sched_getaffinity(0)))
    except Exception:
        pass

    try:
        nproc_output = os.popen("nproc").read().strip()
        if nproc_output:
            candidates.append(int(nproc_output))
    except Exception:
        pass

    if os.cpu_count():
        candidates.append(os.cpu_count() or 1)

    positive = [x for x in candidates if isinstance(x, int) and x > 0]
    if not positive:
        return 1
    return min(positive)


AVAILABLE_VCPU = _detect_available_vcpu()
DEFAULT_POOL_WORKERS = max(1, min(32, AVAILABLE_VCPU // 2))
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_QUEUE_WAIT_SECONDS = 1.0
DEFAULT_MAX_INFLIGHT = max(DEFAULT_POOL_WORKERS * 3, DEFAULT_POOL_WORKERS + 8)
DEFAULT_WORKER_MAX_TASKS = 200


class CodeExtractor:
    @staticmethod
    def extract(text: str) -> str:
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match_generic = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if match_generic:
            return match_generic.group(1).strip()
        return text.strip()


class _WallTimeLimitExceeded(Exception):
    pass


def _build_error_details(exc: BaseException | None, tb_text: str, code_str: str) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "error_type": "",
        "error_message": "",
        "error_line": None,
        "error_snippet": "",
        "traceback": tb_text or "",
    }
    if exc is None:
        return details

    details["error_type"] = type(exc).__name__
    details["error_message"] = str(exc)

    if isinstance(exc, SyntaxError):
        lineno = getattr(exc, "lineno", None)
        if isinstance(lineno, int) and lineno > 0:
            details["error_line"] = lineno
            snippet = getattr(exc, "text", "") or ""
            details["error_snippet"] = snippet.rstrip()
        return details

    tb = exc.__traceback__
    preferred_tb = None
    last_tb = None
    while tb is not None:
        last_tb = tb
        frame_filename = getattr(tb.tb_frame.f_code, "co_filename", "")
        if frame_filename == "<string>":
            preferred_tb = tb
        tb = tb.tb_next
    target_tb = preferred_tb or last_tb
    if target_tb is None:
        return details

    lineno = getattr(target_tb, "tb_lineno", None)
    if isinstance(lineno, int) and lineno > 0:
        details["error_line"] = lineno
        code_lines = code_str.splitlines()
        if lineno <= len(code_lines):
            details["error_snippet"] = code_lines[lineno - 1].rstrip()
        else:
            cached = linecache.getline("<string>", lineno).rstrip()
            if cached:
                details["error_snippet"] = cached
    return details


def _with_error_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("error_type", "")
    payload.setdefault("error_message", "")
    payload.setdefault("error_line", None)
    payload.setdefault("error_snippet", "")
    payload.setdefault("traceback", "")
    return payload


def _sandbox_worker_task(code_str: str, wall_timeout: int, auto_invoke: bool) -> Dict[str, Any]:
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    sandbox_scope = {"__builtins__": __builtins__}

    status = "success"
    result_value = None
    error_message = None
    error_details = _with_error_fields({})
    function_executed = False

    old_handler = None
    if hasattr(signal, "SIGALRM"):

        def _timeout_handler(_signum, _frame):
            raise _WallTimeLimitExceeded(f"Execution reached wall timeout ({wall_timeout}s)")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, max(1, wall_timeout))

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code_str, sandbox_scope, sandbox_scope)

            skipped_functions = []
            if auto_invoke:
                target_func = None
                priority_names = ["main", "run", "solution", "solve", "execute"]
                for name in priority_names:
                    if name in sandbox_scope and callable(sandbox_scope[name]):
                        target_func = sandbox_scope[name]
                        break

                if not target_func:
                    candidates = []
                    for name, obj in sandbox_scope.items():
                        if name.startswith("__") or name == "__builtins__":
                            continue
                        if callable(obj):
                            try:
                                sig = inspect.signature(obj)
                                required_params = [
                                    p
                                    for p in sig.parameters.values()
                                    if p.default == inspect.Parameter.empty
                                    and p.kind
                                    in (
                                        inspect.Parameter.POSITIONAL_ONLY,
                                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                    )
                                ]
                                if len(required_params) == 0:
                                    candidates.append(obj)
                                else:
                                    skipped_functions.append(name)
                            except (ValueError, TypeError):
                                continue
                    if candidates:
                        target_func = candidates[-1]

                if target_func:
                    try:
                        result_value = target_func()
                        function_executed = True
                    except TypeError:
                        pass

                if not function_executed or result_value is None:
                    heuristic_vars = ["result", "ret", "answer", "output", "solution", "data"]
                    found_heuristic = False
                    for var_name in heuristic_vars:
                        if var_name in sandbox_scope and not callable(sandbox_scope[var_name]):
                            result_value = sandbox_scope[var_name]
                            found_heuristic = True
                            break

                    if not found_heuristic:
                        keys = list(sandbox_scope.keys())
                        for key in reversed(keys):
                            val = sandbox_scope[key]
                            if key.startswith("__"):
                                continue
                            if inspect.ismodule(val):
                                continue
                            if callable(val):
                                continue
                            result_value = val
                            break

                if result_value is None and not stdout_capture.getvalue() and skipped_functions and not function_executed:
                    print(
                        f"[System Warning]: Code defined functions {skipped_functions} but did not call them. "
                        f"These functions require arguments, so the sandbox could not auto-execute them. "
                        f"Please explicitly call the function in your code (e.g., '{skipped_functions[0]}(...)' )."
                    )
            else:
                # In script-only mode we do not auto-call functions, but still surface
                # explicit `result` payloads produced by harness programs.
                if "result" in sandbox_scope and not callable(sandbox_scope["result"]):
                    result_value = sandbox_scope["result"]

            if isinstance(result_value, types.GeneratorType):
                result_value = list(result_value)

            try:
                if result_value is not None:
                    pickle.dumps(result_value)
            except (TypeError, pickle.PicklingError):
                result_value = f"<Unpicklable Object>: {str(result_value)}"

    except _WallTimeLimitExceeded as exc:
        status = "timeout"
        error_message = str(exc)
        error_details = _with_error_fields(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    except SystemExit as exc:
        status = "error"
        error_message = f"Program exited with code: {exc.code}"
        error_details = _with_error_fields(
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    except Exception as exc:
        status = "error"
        tb_text = traceback.format_exc()
        error_message = tb_text
        error_details = _build_error_details(exc, tb_text=tb_text, code_str=code_str)
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    output_logs = stdout_capture.getvalue()
    if stderr_capture.getvalue():
        output_logs += f"\n[STDERR]: {stderr_capture.getvalue()}"

    return {
        "status": status,
        "return_value": result_value,
        "stdout": output_logs.strip(),
        "error": error_message,
        **error_details,
    }


class ToolSandbox:
    def __init__(
        self,
        max_workers: int = DEFAULT_POOL_WORKERS,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
        queue_wait_timeout: float = DEFAULT_QUEUE_WAIT_SECONDS,
        worker_max_tasks: int = DEFAULT_WORKER_MAX_TASKS,
    ):
        self.max_workers = max(1, int(max_workers))
        self.timeout = max(1, int(timeout))
        self.max_inflight = max(self.max_workers, int(max_inflight))
        self.queue_wait_timeout = max(0.0, float(queue_wait_timeout))
        self.worker_max_tasks = max(1, int(worker_max_tasks))

        self._executor_lock = threading.Lock()
        self._metric_lock = threading.Lock()
        self._slots = asyncio.Semaphore(self.max_inflight)

        self._executor = self._create_executor()

        self._total_requests = 0
        self._success_requests = 0
        self._busy_requests = 0
        self._timeout_requests = 0
        self._error_requests = 0
        self._pool_restarts = 0
        self._inflight = 0
        self._max_inflight_seen = 0
        self._latency_ms_sum = 0.0

    def _create_executor(self) -> ProcessPoolExecutor:
        default_method = "spawn" if sys.platform == "win32" else "fork"
        start_method = os.environ.get("RUNPY_TOOL_MP_START_METHOD", default_method)
        try:
            ctx = multiprocessing.get_context(start_method)
            try:
                return ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    mp_context=ctx,
                    max_tasks_per_child=self.worker_max_tasks,
                )
            except TypeError:
                return ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)
        except (TypeError, ValueError):
            return ProcessPoolExecutor(max_workers=self.max_workers)

    def _submit(self, code: str, timeout: int, auto_invoke: bool):
        with self._executor_lock:
            return self._executor.submit(_sandbox_worker_task, code, timeout, auto_invoke)

    def _restart_pool(self, reason: str) -> None:
        with self._executor_lock:
            old_executor = self._executor
            self._executor = self._create_executor()
        with self._metric_lock:
            self._pool_restarts += 1
        logger.warning("Process pool restarted due to: %s", reason)
        old_executor.shutdown(wait=False, cancel_futures=True)

    def _record_start(self) -> float:
        now = time.perf_counter()
        with self._metric_lock:
            self._total_requests += 1
            self._inflight += 1
            if self._inflight > self._max_inflight_seen:
                self._max_inflight_seen = self._inflight
        return now

    def _record_end(self, started: float, status: str) -> None:
        latency_ms = (time.perf_counter() - started) * 1000.0
        with self._metric_lock:
            self._inflight = max(0, self._inflight - 1)
            if status == "success":
                self._success_requests += 1
                self._latency_ms_sum += latency_ms
            elif status == "busy":
                self._busy_requests += 1
            elif status == "timeout":
                self._timeout_requests += 1
            else:
                self._error_requests += 1

    async def run_async(
        self,
        raw_text: str,
        extract_code: bool = True,
        timeout: int | None = None,
        auto_invoke: bool = True,
    ) -> Dict[str, Any]:
        timeout_value = self.timeout if timeout is None else max(1, int(timeout))
        code = CodeExtractor.extract(raw_text) if extract_code else raw_text
        if not code.strip():
            return _with_error_fields(
                {"status": "error", "error": "No python code found.", "stdout": "", "return_value": None}
            )

        acquired = False
        started = time.perf_counter()
        try:
            try:
                await asyncio.wait_for(self._slots.acquire(), timeout=self.queue_wait_timeout)
                acquired = True
            except asyncio.TimeoutError:
                with self._metric_lock:
                    self._total_requests += 1
                    self._busy_requests += 1
                return _with_error_fields({
                    "status": "busy",
                    "error": "Server is overloaded. Request queue is full, please retry.",
                    "stdout": "",
                    "return_value": None,
                })

            started = self._record_start()
            future = self._submit(code, timeout_value, auto_invoke)
            wrapped = asyncio.wrap_future(future)
            try:
                result = await asyncio.wait_for(wrapped, timeout=timeout_value + 1)
                status = str(result.get("status", "success"))
                self._record_end(started, "success" if status == "success" else status)
                return result
            except asyncio.TimeoutError:
                future.cancel()
                self._record_end(started, "timeout")
                self._restart_pool("outer-timeout")
                return _with_error_fields({
                    "status": "timeout",
                    "error": f"Execution timed out ({timeout_value}s).",
                    "stdout": "",
                    "return_value": None,
                    "error_type": "TimeoutError",
                    "error_message": f"Execution timed out ({timeout_value}s).",
                })
            except BrokenProcessPool:
                self._record_end(started, "error")
                self._restart_pool("broken-process-pool")
                return _with_error_fields({
                    "status": "system_error",
                    "error": "Process pool crashed and has been restarted. Please retry.",
                    "stdout": "",
                    "return_value": None,
                    "error_type": "BrokenProcessPool",
                    "error_message": "Process pool crashed and has been restarted. Please retry.",
                })
            except Exception as exc:
                self._record_end(started, "error")
                return _with_error_fields(
                    {
                        "status": "system_error",
                        "error": str(exc),
                        "stdout": "",
                        "return_value": None,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
        finally:
            if acquired:
                self._slots.release()

    def run(
        self,
        raw_text: str,
        extract_code: bool = True,
        timeout: int | None = None,
        auto_invoke: bool = True,
    ) -> Dict[str, Any]:
        timeout_value = self.timeout if timeout is None else max(1, int(timeout))
        code = CodeExtractor.extract(raw_text) if extract_code else raw_text
        if not code.strip():
            return _with_error_fields(
                {"status": "error", "error": "No python code found.", "stdout": "", "return_value": None}
            )

        started = self._record_start()
        future = None
        try:
            future = self._submit(code, timeout_value, auto_invoke)
            result = future.result(timeout=timeout_value + 1)
            status = str(result.get("status", "success"))
            self._record_end(started, "success" if status == "success" else status)
            return result
        except TimeoutError:
            if future is not None:
                future.cancel()
            self._record_end(started, "timeout")
            self._restart_pool("sync-timeout")
            return _with_error_fields({
                "status": "timeout",
                "error": f"Execution timed out ({timeout_value}s).",
                "stdout": "",
                "return_value": None,
                "error_type": "TimeoutError",
                "error_message": f"Execution timed out ({timeout_value}s).",
            })
        except BrokenProcessPool:
            self._record_end(started, "error")
            self._restart_pool("broken-process-pool")
            return _with_error_fields({
                "status": "system_error",
                "error": "Process pool crashed and has been restarted. Please retry.",
                "stdout": "",
                "return_value": None,
                "error_type": "BrokenProcessPool",
                "error_message": "Process pool crashed and has been restarted. Please retry.",
            })
        except Exception as exc:
            self._record_end(started, "error")
            return _with_error_fields(
                {
                    "status": "system_error",
                    "error": str(exc),
                    "stdout": "",
                    "return_value": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    def stats(self) -> Dict[str, Any]:
        with self._metric_lock:
            avg_latency = self._latency_ms_sum / self._success_requests if self._success_requests else 0.0
            return {
                "status": "ok",
                "available_vcpu": AVAILABLE_VCPU,
                "max_workers": self.max_workers,
                "max_inflight": self.max_inflight,
                "timeout_seconds": self.timeout,
                "queue_wait_seconds": self.queue_wait_timeout,
                "worker_max_tasks": self.worker_max_tasks,
                "total_requests": self._total_requests,
                "success_requests": self._success_requests,
                "busy_requests": self._busy_requests,
                "timeout_requests": self._timeout_requests,
                "error_requests": self._error_requests,
                "inflight_requests": self._inflight,
                "max_inflight_seen": self._max_inflight_seen,
                "pool_restarts": self._pool_restarts,
                "avg_success_latency_ms": round(avg_latency, 3),
            }

    def shutdown(self) -> None:
        with self._executor_lock:
            self._executor.shutdown(wait=False, cancel_futures=True)


def _build_humaneval_program(code: str, prompt: str, test: str, entry_point: str) -> tuple[str, bool]:
    used_prompt_prefix = f"def {entry_point}" not in code
    candidate_program = f"{prompt}\n{code}" if used_prompt_prefix else code

    harness = f"""
import traceback

__slime_he_result = {{
    "passed": False,
    "status": "runtime_error",
    "error": "",
    "traceback": "",
    "error_type": "",
    "error_message": "",
    "error_line": None,
    "error_snippet": "",
    "used_prompt_prefix": {used_prompt_prefix},
}}

def __slime_he_run():
    candidate = globals().get({entry_point!r})
    if not callable(candidate):
        raise NameError(f"Entry point '{{{entry_point}}}' is missing or not callable")
    check_fn = globals().get("check")
    if not callable(check_fn):
        raise NameError("Function 'check' is missing in test code")
    check_fn(candidate)

try:
    __slime_he_run()
    __slime_he_result["passed"] = True
    __slime_he_result["status"] = "passed"
except Exception as _exc:
    __slime_he_result["error"] = f"{{type(_exc).__name__}}: {{_exc}}"
    __slime_he_result["traceback"] = traceback.format_exc()
    __slime_he_result["error_type"] = type(_exc).__name__
    __slime_he_result["error_message"] = str(_exc)
    _tb = _exc.__traceback__
    while _tb is not None and _tb.tb_next is not None:
        _tb = _tb.tb_next
    if _tb is not None:
        __slime_he_result["error_line"] = _tb.tb_lineno
        try:
            _code_lines = {repr(code)}.splitlines()
            if 1 <= _tb.tb_lineno <= len(_code_lines):
                __slime_he_result["error_snippet"] = _code_lines[_tb.tb_lineno - 1].rstrip()
        except Exception:
            pass

result = __slime_he_result
"""

    program = f"{candidate_program}\n\n{test}\n\n{harness}"
    return program, used_prompt_prefix


_SANDBOX_CACHE: dict[tuple[int, int, int], ToolSandbox] = {}
_SANDBOX_CACHE_LOCK = threading.Lock()


def _get_cached_sandbox(max_workers: int, timeout: int, worker_max_tasks: int = DEFAULT_WORKER_MAX_TASKS) -> ToolSandbox:
    key = (max_workers, timeout, worker_max_tasks)
    with _SANDBOX_CACHE_LOCK:
        sandbox = _SANDBOX_CACHE.get(key)
        if sandbox is None:
            sandbox = ToolSandbox(
                max_workers=max_workers,
                timeout=timeout,
                max_inflight=max(max_workers * 3, max_workers + 8),
                queue_wait_timeout=DEFAULT_QUEUE_WAIT_SECONDS,
                worker_max_tasks=worker_max_tasks,
            )
            _SANDBOX_CACHE[key] = sandbox
    return sandbox


def run_humaneval_case(
    code: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_workers: int = DEFAULT_POOL_WORKERS,
) -> Dict[str, Any]:
    code = CodeExtractor.extract(str(code))
    prompt = str(prompt or "")
    test = str(test or "")
    entry_point = str(entry_point or "")

    if not entry_point:
        return _with_error_fields({
            "passed": False,
            "status": "runtime_error",
            "error": "entry_point is empty",
            "traceback": "",
            "stdout": "",
            "stderr": "",
            "used_prompt_prefix": False,
        })

    sandbox = _get_cached_sandbox(max_workers=max_workers, timeout=timeout)
    program, used_prompt_prefix = _build_humaneval_program(
        code=code,
        prompt=prompt,
        test=test,
        entry_point=entry_point,
    )

    tool_result = sandbox.run(program, extract_code=False, timeout=timeout, auto_invoke=False)
    normalized: Dict[str, Any] = _with_error_fields({
        "passed": False,
        "status": "runtime_error",
        "error": "",
        "traceback": "",
        "stdout": tool_result.get("stdout", "") or "",
        "stderr": "",
        "used_prompt_prefix": used_prompt_prefix,
    })

    status = str(tool_result.get("status", ""))
    if status == "timeout":
        normalized["status"] = "timeout"
        normalized["error"] = tool_result.get("error", "") or f"Execution timed out ({timeout}s)."
        normalized["error_type"] = tool_result.get("error_type", "") or normalized.get("error_type", "")
        normalized["error_message"] = tool_result.get("error_message", "") or normalized.get("error_message", "")
        normalized["error_line"] = tool_result.get("error_line")
        normalized["error_snippet"] = tool_result.get("error_snippet", "") or normalized.get("error_snippet", "")
        normalized["traceback"] = tool_result.get("traceback", "") or normalized.get("traceback", "")
        return normalized
    if status in ("error", "system_error", "busy"):
        normalized["status"] = "runtime_error"
        normalized["error"] = tool_result.get("error", "") or "Sandbox execution failed."
        normalized["error_type"] = tool_result.get("error_type", "") or normalized.get("error_type", "")
        normalized["error_message"] = tool_result.get("error_message", "") or normalized.get("error_message", "")
        normalized["error_line"] = tool_result.get("error_line")
        normalized["error_snippet"] = tool_result.get("error_snippet", "") or normalized.get("error_snippet", "")
        normalized["traceback"] = tool_result.get("traceback", "") or normalized.get("traceback", "")
        return normalized

    rv = tool_result.get("return_value")
    if isinstance(rv, dict) and "passed" in rv:
        normalized.update(rv)
        normalized["stdout"] = tool_result.get("stdout", "") or normalized.get("stdout", "")
        normalized.setdefault("stderr", "")
        _with_error_fields(normalized)
        return normalized

    normalized["status"] = "runtime_error"
    normalized["error"] = "HumanEval harness did not produce expected result payload."
    return normalized


class RunRequest(BaseModel):
    raw_text: str = Field(..., description="Python code or markdown text containing Python code.")
    extract_code: bool = Field(default=True, description="Whether to extract code blocks from markdown.")
    timeout: int | None = Field(default=None, ge=1, le=180)
    auto_invoke: bool = Field(
        default=True,
        description="Whether to auto-call zero-arg functions / infer return value after exec.",
    )


class RunResponse(BaseModel):
    status: str
    return_value: Optional[Any] = None
    stdout: str = ""
    error: Optional[str] = None
    error_type: str = ""
    error_message: str = ""
    error_line: Optional[int] = None
    error_snippet: str = ""
    traceback: str = ""


class HumanEvalRequest(BaseModel):
    code: str
    prompt: str
    test: str
    entry_point: str
    timeout: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=180)
    max_workers: int = Field(default=DEFAULT_POOL_WORKERS, ge=1, le=256)


class RunTestsRequest(BaseModel):
    code: str
    tests: list[str]
    timeout: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=180)
    max_workers: int = Field(default=DEFAULT_POOL_WORKERS, ge=1, le=256)


def _build_run_tests_program(code: str, tests: list[str]) -> str:
    test_list_literal = repr([str(t) for t in tests if str(t).strip()])
    harness = f"""
import traceback

__slime_tests = {test_list_literal}
__slime_test_result = {{
    "passed_all": False,
    "passed_count": 0,
    "total_count": len(__slime_tests),
    "status": "runtime_error",
    "error": "",
    "traceback": "",
    "error_type": "",
    "error_message": "",
    "error_line": None,
    "error_snippet": "",
    "failed_test_index": None,
}}

try:
    for __slime_idx, __slime_case in enumerate(__slime_tests):
        exec(__slime_case, globals(), globals())
        __slime_test_result["passed_count"] += 1
    __slime_test_result["passed_all"] = True
    __slime_test_result["status"] = "passed"
except Exception as _exc:
    __slime_test_result["failed_test_index"] = __slime_idx if "__slime_idx" in globals() else None
    __slime_test_result["error"] = f"{{type(_exc).__name__}}: {{_exc}}"
    __slime_test_result["traceback"] = traceback.format_exc()
    __slime_test_result["error_type"] = type(_exc).__name__
    __slime_test_result["error_message"] = str(_exc)
    _tb = _exc.__traceback__
    while _tb is not None and _tb.tb_next is not None:
        _tb = _tb.tb_next
    if _tb is not None:
        __slime_test_result["error_line"] = _tb.tb_lineno
        try:
            _code_lines = {repr(code)}.splitlines()
            if 1 <= _tb.tb_lineno <= len(_code_lines):
                __slime_test_result["error_snippet"] = _code_lines[_tb.tb_lineno - 1].rstrip()
        except Exception:
            pass
    __slime_test_result["status"] = "failed"

result = __slime_test_result
"""
    return f"{code}\n\n{harness}"


def run_python_tests(
    code: str,
    tests: list[str],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_workers: int = DEFAULT_POOL_WORKERS,
) -> Dict[str, Any]:
    code = CodeExtractor.extract(str(code))
    normalized_tests = [str(t).strip() for t in tests if str(t).strip()]
    if not code.strip():
        return _with_error_fields({
            "passed_all": False,
            "passed_count": 0,
            "total_count": len(normalized_tests),
            "status": "runtime_error",
            "error": "candidate code is empty",
            "traceback": "",
            "stdout": "",
            "stderr": "",
            "failed_test_index": None,
        })

    sandbox = _get_cached_sandbox(max_workers=max_workers, timeout=timeout)
    program = _build_run_tests_program(code=code, tests=normalized_tests)
    tool_result = sandbox.run(program, extract_code=False, timeout=timeout, auto_invoke=False)

    normalized: Dict[str, Any] = _with_error_fields({
        "passed_all": False,
        "passed_count": 0,
        "total_count": len(normalized_tests),
        "status": "runtime_error",
        "error": "",
        "traceback": "",
        "stdout": tool_result.get("stdout", "") or "",
        "stderr": "",
        "failed_test_index": None,
    })

    status = str(tool_result.get("status", ""))
    if status == "timeout":
        normalized["status"] = "timeout"
        normalized["error"] = tool_result.get("error", "") or f"Execution timed out ({timeout}s)."
        normalized["error_type"] = tool_result.get("error_type", "") or normalized.get("error_type", "")
        normalized["error_message"] = tool_result.get("error_message", "") or normalized.get("error_message", "")
        normalized["error_line"] = tool_result.get("error_line")
        normalized["error_snippet"] = tool_result.get("error_snippet", "") or normalized.get("error_snippet", "")
        normalized["traceback"] = tool_result.get("traceback", "") or normalized.get("traceback", "")
        return normalized
    if status in ("error", "system_error", "busy"):
        normalized["status"] = "runtime_error"
        normalized["error"] = tool_result.get("error", "") or "Sandbox execution failed."
        normalized["error_type"] = tool_result.get("error_type", "") or normalized.get("error_type", "")
        normalized["error_message"] = tool_result.get("error_message", "") or normalized.get("error_message", "")
        normalized["error_line"] = tool_result.get("error_line")
        normalized["error_snippet"] = tool_result.get("error_snippet", "") or normalized.get("error_snippet", "")
        normalized["traceback"] = tool_result.get("traceback", "") or normalized.get("traceback", "")
        return normalized

    rv = tool_result.get("return_value")
    if isinstance(rv, dict) and "passed_all" in rv:
        normalized.update(rv)
        normalized["stdout"] = tool_result.get("stdout", "") or normalized.get("stdout", "")
        normalized.setdefault("stderr", "")
        _with_error_fields(normalized)
        return normalized

    normalized["status"] = "runtime_error"
    normalized["error"] = "run_tests harness did not produce expected result payload."
    return normalized


MAX_WORKERS = _env_int("RUNPY_MAX_WORKERS", DEFAULT_POOL_WORKERS)
TIMEOUT_SECONDS = _env_int("RUNPY_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
MAX_INFLIGHT = _env_int("RUNPY_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT)
QUEUE_WAIT_SECONDS = _env_float("RUNPY_QUEUE_WAIT_TIMEOUT", DEFAULT_QUEUE_WAIT_SECONDS, minimum=0.0)
WORKER_MAX_TASKS = _env_int("RUNPY_WORKER_MAX_TASKS", DEFAULT_WORKER_MAX_TASKS)

SERVICE_SANDBOX = ToolSandbox(
    max_workers=MAX_WORKERS,
    timeout=TIMEOUT_SECONDS,
    max_inflight=MAX_INFLIGHT,
    queue_wait_timeout=QUEUE_WAIT_SECONDS,
    worker_max_tasks=WORKER_MAX_TASKS,
)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    SERVICE_SANDBOX.shutdown()


app = FastAPI(
    title="RunPythonTool Service",
    version="1.1.0",
    description="High-throughput Python execution service with process-isolated workers.",
    lifespan=_lifespan,
)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/stats")
async def stats() -> Dict[str, Any]:
    return SERVICE_SANDBOX.stats()


@app.post("/run", response_model=RunResponse)
async def run_python(request: RunRequest) -> Dict[str, Any]:
    result = await SERVICE_SANDBOX.run_async(
        request.raw_text,
        extract_code=request.extract_code,
        timeout=request.timeout,
        auto_invoke=request.auto_invoke,
    )
    if result.get("status") == "busy":
        raise HTTPException(status_code=429, detail=result)
    return result


@app.post("/run_humaneval")
async def run_humaneval_api(request: HumanEvalRequest) -> Dict[str, Any]:
    result = await asyncio.to_thread(
        run_humaneval_case,
        code=request.code,
        prompt=request.prompt,
        test=request.test,
        entry_point=request.entry_point,
        timeout=request.timeout,
        max_workers=request.max_workers,
    )
    return result


@app.post("/run_tests")
async def run_tests_api(request: RunTestsRequest) -> Dict[str, Any]:
    result = await asyncio.to_thread(
        run_python_tests,
        code=request.code,
        tests=request.tests,
        timeout=request.timeout,
        max_workers=request.max_workers,
    )
    return result


def main() -> None:
    host = os.getenv("RUNPY_HOST", "0.0.0.0")
    port = _env_int("RUNPY_PORT", 18080)
    workers = _env_int("RUNPY_API_WORKERS", 1)
    log_level = os.getenv("RUNPY_LOG_LEVEL", "info")

    if workers == 1:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
        return

    uvicorn.run("RunPythonTool:app", host=host, port=port, workers=workers, log_level=log_level)


if __name__ == "__main__":
    main()


__all__ = [
    "CodeExtractor",
    "ToolSandbox",
    "run_humaneval_case",
    "run_python_tests",
    "app",
]
