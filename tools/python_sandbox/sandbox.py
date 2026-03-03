#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

RESULT_MARKER = "__SLIME_PY_SANDBOX_RESULT__"

DEFAULT_TIMEOUT_SECONDS = 6
DEFAULT_MEMORY_MB = 768
DEFAULT_MAX_OUTPUT_CHARS = 8000


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    return int(value) if value is not None and value != "" else default


DEFAULT_BACKEND = os.environ.get("SLIME_PY_SANDBOX_BACKEND", "runpytool").strip().lower()
RUNPY_TOOL_MAX_WORKERS = _env_int("RUNPY_TOOL_MAX_WORKERS", 4)

DISALLOWED_IMPORT_ROOTS = {
    "asyncio",
    "ctypes",
    "multiprocessing",
    "os",
    "pathlib",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "threading",
}

DISALLOWED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}

DISALLOWED_ATTR_CALLS = {
    "chmod",
    "chown",
    "kill",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "rmtree",
    "system",
    "unlink",
}


def _clip_text(text: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    extra = len(text) - max_chars
    return f"{text[:max_chars]}\n...<truncated {extra} chars>"


def _extract_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def validate_code_safety(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in DISALLOWED_IMPORT_ROOTS:
                    return False, f"Import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in DISALLOWED_IMPORT_ROOTS:
                    return False, f"Import not allowed: {node.module}"
        elif isinstance(node, ast.Call):
            call_name = _extract_call_name(node.func)
            if call_name in DISALLOWED_CALL_NAMES or call_name in DISALLOWED_ATTR_CALLS:
                return False, f"Call not allowed: {call_name}"

    return True, ""


def _build_runner_script(
    code: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout_seconds: int,
    memory_mb: int,
) -> str:
    return textwrap.dedent(
        f"""
        import json
        import resource
        import signal
        import time
        import traceback

        RESULT_MARKER = {RESULT_MARKER!r}
        TIMEOUT_SECONDS = {int(timeout_seconds)}
        MEMORY_BYTES = {int(memory_mb)} * 1024 * 1024

        USER_CODE = {code!r}
        TASK_PROMPT = {prompt!r}
        TEST_CODE = {test!r}
        ENTRY_POINT = {entry_point!r}


        def emit(payload):
            print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False))


        def set_limits():
            # Memory limit to keep execution bounded.
            try:
                resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
            except Exception:
                pass

            # CPU time hard cap.
            cpu_limit = max(1, int(TIMEOUT_SECONDS))
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
            except Exception:
                pass


        def timeout_handler(_signum, _frame):
            raise TimeoutError(f"Execution timed out after {{TIMEOUT_SECONDS}} seconds")


        start_ts = time.time()
        used_prompt_prefix = False
        set_limits()
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, int(TIMEOUT_SECONDS)))

        try:
            namespace = {{}}
            program = USER_CODE
            if f"def {{ENTRY_POINT}}" not in USER_CODE:
                # If model outputs only completion/body, prepend original prompt.
                program = TASK_PROMPT + "\\n" + USER_CODE
                used_prompt_prefix = True

            exec(program, namespace)
            exec(TEST_CODE, namespace)

            candidate = namespace.get(ENTRY_POINT)
            if not callable(candidate):
                raise NameError(f"Entry point '{{ENTRY_POINT}}' is missing or not callable")

            check_fn = namespace.get("check")
            if not callable(check_fn):
                raise NameError("Function 'check' is missing in test code")

            check_fn(candidate)
            signal.alarm(0)
            emit(
                {{
                    "passed": True,
                    "status": "passed",
                    "error": "",
                    "used_prompt_prefix": used_prompt_prefix,
                    "execution_seconds": round(time.time() - start_ts, 6),
                }}
            )
        except TimeoutError as exc:
            signal.alarm(0)
            emit(
                {{
                    "passed": False,
                    "status": "timeout",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "used_prompt_prefix": used_prompt_prefix,
                    "execution_seconds": round(time.time() - start_ts, 6),
                }}
            )
        except Exception as exc:
            signal.alarm(0)
            emit(
                {{
                    "passed": False,
                    "status": "runtime_error",
                    "error": f"{{type(exc).__name__}}: {{exc}}",
                    "traceback": traceback.format_exc(),
                    "used_prompt_prefix": used_prompt_prefix,
                    "execution_seconds": round(time.time() - start_ts, 6),
                }}
            )
        """
    ).strip()


def _parse_runner_output(stdout: str) -> tuple[dict[str, Any] | None, str]:
    payload: dict[str, Any] | None = None
    visible_lines: list[str] = []

    for line in stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            raw = line[len(RESULT_MARKER) :]
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {
                    "passed": False,
                    "status": "sandbox_error",
                    "error": "Runner returned invalid JSON payload",
                }
        else:
            visible_lines.append(line)

    return payload, "\n".join(visible_lines).strip()


def _load_runpytool_backend():
    slime_root = str(Path(__file__).resolve().parents[2])
    if slime_root not in sys.path:
        sys.path.insert(0, slime_root)
    try:
        module = importlib.import_module("tools.RunPythonTool")
        return getattr(module, "run_humaneval_case", None)
    except Exception:
        return None


RUN_HUMANEVAL_CASE = _load_runpytool_backend()


def _run_with_runpytool(
    code: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout_seconds: int,
    max_output_chars: int,
    start_ts: float,
) -> dict[str, Any]:
    if RUN_HUMANEVAL_CASE is None:
        return {
            "passed": False,
            "status": "sandbox_error",
            "error": "RunPythonTool backend is unavailable",
            "stdout": "",
            "stderr": "",
            "execution_seconds": round(time.time() - start_ts, 6),
            "backend": "runpytool",
        }

    try:
        payload = RUN_HUMANEVAL_CASE(
            code=code,
            prompt=prompt,
            test=test,
            entry_point=entry_point,
            timeout=int(timeout_seconds),
            max_workers=RUNPY_TOOL_MAX_WORKERS,
        )
    except Exception as exc:
        return {
            "passed": False,
            "status": "sandbox_error",
            "error": f"RunPythonTool invocation failed: {type(exc).__name__}: {exc}",
            "stdout": "",
            "stderr": "",
            "execution_seconds": round(time.time() - start_ts, 6),
            "backend": "runpytool",
        }

    payload["stdout"] = _clip_text(str(payload.get("stdout", "")), max_output_chars)
    payload["stderr"] = _clip_text(str(payload.get("stderr", "")), max_output_chars)
    payload["execution_seconds"] = round(
        max(float(payload.get("execution_seconds", 0.0)), time.time() - start_ts),
        6,
    )
    payload["backend"] = "runpytool"
    return payload


def run_humaneval_in_sandbox(
    code: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    start_ts = time.time()
    safe, message = validate_code_safety(code)
    if not safe:
        return {
            "passed": False,
            "status": "safety_reject",
            "error": message,
            "stdout": "",
            "stderr": "",
            "execution_seconds": round(time.time() - start_ts, 6),
        }

    backend = DEFAULT_BACKEND
    if backend in ("runpytool", "auto"):
        runpy_payload = _run_with_runpytool(
            code=code,
            prompt=prompt,
            test=test,
            entry_point=entry_point,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
            start_ts=start_ts,
        )
        if backend == "runpytool" or runpy_payload.get("status") != "sandbox_error":
            return runpy_payload

    with tempfile.TemporaryDirectory(prefix="slime_py_sandbox_") as temp_dir:
        runner_path = Path(temp_dir) / "runner.py"
        runner_code = _build_runner_script(
            code=code,
            prompt=prompt,
            test=test,
            entry_point=entry_point,
            timeout_seconds=timeout_seconds,
            memory_mb=memory_mb,
        )
        runner_path.write_text(runner_code, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(runner_path)],
                capture_output=True,
                text=True,
                cwd=temp_dir,
                timeout=max(1, int(timeout_seconds) + 1),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "status": "timeout",
                "error": f"Sandbox process timeout after {timeout_seconds} seconds",
                "stdout": "",
                "stderr": "",
                "execution_seconds": round(time.time() - start_ts, 6),
            }

    payload, visible_stdout = _parse_runner_output(completed.stdout or "")
    if payload is None:
        payload = {
            "passed": False,
            "status": "sandbox_error",
            "error": f"Sandbox exited without result marker (returncode={completed.returncode})",
        }

    payload["stdout"] = _clip_text(visible_stdout, max_output_chars)
    payload["stderr"] = _clip_text((completed.stderr or "").strip(), max_output_chars)
    payload["returncode"] = int(completed.returncode)
    payload["execution_seconds"] = round(
        max(float(payload.get("execution_seconds", 0.0)), time.time() - start_ts),
        6,
    )
    payload["backend"] = "subprocess"
    return payload


def _load_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        code = Path(args.code_file).read_text(encoding="utf-8") if args.code_file else (args.code or "")
        prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else (args.prompt or "")
        test = Path(args.test_file).read_text(encoding="utf-8") if args.test_file else (args.test or "")
        payload = {
            "code": code,
            "prompt": prompt,
            "test": test,
            "entry_point": args.entry_point,
        }

    missing = [key for key in ("code", "prompt", "test", "entry_point") if not payload.get(key)]
    if missing:
        raise ValueError(f"Missing required input fields: {', '.join(missing)}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate Python code against HumanEval-style tests in sandbox.")
    parser.add_argument("--input-json", type=str, default="", help="Path to JSON file containing tool input.")
    parser.add_argument("--code-file", type=str, default="", help="Path to candidate code file.")
    parser.add_argument("--prompt-file", type=str, default="", help="Path to task prompt file.")
    parser.add_argument("--test-file", type=str, default="", help="Path to test code file.")
    parser.add_argument("--entry-point", type=str, default="", help="Function entry point name.")
    parser.add_argument("--code", type=str, default="", help="Inline candidate code.")
    parser.add_argument("--prompt", type=str, default="", help="Inline task prompt.")
    parser.add_argument("--test", type=str, default="", help="Inline test code.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    parser.add_argument("--max-output-chars", type=int, default=DEFAULT_MAX_OUTPUT_CHARS)
    args = parser.parse_args()

    payload = _load_payload_from_args(args)
    result = run_humaneval_in_sandbox(
        code=str(payload["code"]),
        prompt=str(payload["prompt"]),
        test=str(payload["test"]),
        entry_point=str(payload["entry_point"]),
        timeout_seconds=args.timeout_seconds,
        memory_mb=args.memory_mb,
        max_output_chars=args.max_output_chars,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
