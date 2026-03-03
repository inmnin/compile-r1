import sys
import os
import io
import re
import contextlib
import traceback
import logging
import inspect
import pickle
import types
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Dict, Any

# 0. 确保路径正确
current_path = os.getcwd()
if current_path not in sys.path:
    sys.path.append(current_path)

logger = logging.getLogger(__name__)

class CodeExtractor:
    @staticmethod
    def extract(text: str) -> str:
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if match: return match.group(1).strip()
        match_generic = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if match_generic: return match_generic.group(1).strip()
        return text.strip()

def _sandbox_worker_task(code_str: str) -> Dict[str, Any]:
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # 统一作用域
    sandbox_scope = {"__builtins__": __builtins__}
    
    status = "success"
    result_value = None
    error_message = None
    
    function_executed = False

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code_str, sandbox_scope, sandbox_scope)
            
            target_func = None

            priority_names = ['main', 'run', 'solution', 'solve', 'execute']
            for name in priority_names:
                if name in sandbox_scope and callable(sandbox_scope[name]):
                    target_func = sandbox_scope[name]
                    break

            skipped_functions = [] 
            if not target_func:
                candidates = []
                for name, obj in sandbox_scope.items():
                    if name.startswith("__") or name == "__builtins__": continue
                    
                    if callable(obj):
                        try:
                            sig = inspect.signature(obj)
                            required_params = [
                                p for p in sig.parameters.values() 
                                if p.default == inspect.Parameter.empty and 
                                p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
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
                
                heuristic_vars = ['result', 'ret', 'answer', 'output', 'solution', 'data']
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
                        
                        if key.startswith("__"): continue
                        if inspect.ismodule(val): continue
                        if callable(val): continue
                        
                        # 🎉 找到了！这是最后一个被定义的“数据”
                        result_value = val
                        break


            if result_value is None and not stdout_capture.getvalue() and skipped_functions and not function_executed:
                print(f"[System Warning]: Code defined functions {skipped_functions} but did not call them. "
                      f"These functions require arguments, so the sandbox could not auto-execute them. "
                      f"Please explicitly call the function in your code (e.g., '{skipped_functions[0]}(...)' ).")

            # F. 生成器处理
            if isinstance(result_value, types.GeneratorType):
                result_value = list(result_value)
            
            # G. 序列化检查
            try:
                if result_value is not None:
                    pickle.dumps(result_value)
            except (TypeError, pickle.PicklingError):
                result_value = f"<Unpicklable Object>: {str(result_value)}"

    except SystemExit as e:
        status = "error"
        error_message = f"Program exited with code: {e.code}"
    except Exception:
        status = "error"
        error_message = traceback.format_exc()

    output_logs = stdout_capture.getvalue()
    if stderr_capture.getvalue():
        output_logs += f"\n[STDERR]: {stderr_capture.getvalue()}"

    return {
        "status": status,
        "return_value": result_value,
        "stdout": output_logs.strip(),
        "error": error_message
    }


class ToolSandbox:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ToolSandbox, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 2, timeout: int = 15):
        if not hasattr(self, 'initialized'):
            self.timeout = timeout
            try:
                default_method = "spawn" if sys.platform == "win32" else "fork"
                start_method = os.environ.get("RUNPY_TOOL_MP_START_METHOD", default_method)
                ctx = multiprocessing.get_context(start_method)
                self._executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
            except (TypeError, ValueError):
                self._executor = ProcessPoolExecutor(max_workers=max_workers)
            self.initialized = True

    def run(self, raw_text: str, extract_code: bool = True, timeout: int | None = None) -> Dict[str, Any]:
        if extract_code:
            code = CodeExtractor.extract(raw_text)
        else:
            code = raw_text

        if not code.strip():
            return {"status": "error", "error": "No python code found."}

        try:
            future = self._executor.submit(_sandbox_worker_task, code)
            timeout_value = self.timeout if timeout is None else timeout
            return future.result(timeout=timeout_value)
        except TimeoutError:
            future.cancel()
            timeout_value = self.timeout if timeout is None else timeout
            return {
                "status": "timeout",
                "error": f"Execution timed out ({timeout_value}s).",
                "stdout": "",
                "return_value": None,
            }
        except Exception as e:
            return {"status": "system_error", "error": str(e), "stdout": "", "return_value": None}

    def shutdown(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    def run_humaneval(
        self,
        code: str,
        prompt: str,
        test: str,
        entry_point: str,
        timeout: int | None = None,
    ) -> Dict[str, Any]:
        timeout_value = self.timeout if timeout is None else timeout
        return run_humaneval_case(
            code=code,
            prompt=prompt,
            test=test,
            entry_point=entry_point,
            timeout=timeout_value,
        )


def _build_humaneval_program(code: str, prompt: str, test: str, entry_point: str) -> tuple[str, bool]:
    """
    Build a deterministic HumanEval execution program.
    Returns (program, used_prompt_prefix).
    """
    used_prompt_prefix = f"def {entry_point}" not in code
    candidate_program = f"{prompt}\n{code}" if used_prompt_prefix else code

    harness = f"""
import traceback

__slime_he_result = {{
    "passed": False,
    "status": "runtime_error",
    "error": "",
    "traceback": "",
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

result = __slime_he_result
"""

    program = f"{candidate_program}\n\n{test}\n\n{harness}"
    return program, used_prompt_prefix


def run_humaneval_case(
    code: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout: int = 15,
    max_workers: int = 2,
) -> Dict[str, Any]:
    """
    Library API: run HumanEval-style candidate code in multi-process sandbox.
    Returns a normalized dict used by compile-r1 tool calling.
    """
    code = CodeExtractor.extract(str(code))
    prompt = str(prompt or "")
    test = str(test or "")
    entry_point = str(entry_point or "")

    if not entry_point:
        return {
            "passed": False,
            "status": "runtime_error",
            "error": "entry_point is empty",
            "traceback": "",
            "stdout": "",
            "stderr": "",
            "used_prompt_prefix": False,
        }

    sandbox = ToolSandbox(max_workers=max_workers, timeout=timeout)
    program, used_prompt_prefix = _build_humaneval_program(
        code=code,
        prompt=prompt,
        test=test,
        entry_point=entry_point,
    )

    tool_result = sandbox.run(program, extract_code=False, timeout=timeout)
    normalized: Dict[str, Any] = {
        "passed": False,
        "status": "runtime_error",
        "error": "",
        "traceback": "",
        "stdout": tool_result.get("stdout", "") or "",
        "stderr": "",
        "used_prompt_prefix": used_prompt_prefix,
    }

    status = str(tool_result.get("status", ""))
    if status == "timeout":
        normalized["status"] = "timeout"
        normalized["error"] = tool_result.get("error", "") or f"Execution timed out ({timeout}s)."
        return normalized
    if status in ("error", "system_error"):
        normalized["status"] = "runtime_error"
        normalized["error"] = tool_result.get("error", "") or "Sandbox execution failed."
        return normalized

    rv = tool_result.get("return_value")
    if isinstance(rv, dict) and "passed" in rv:
        normalized.update(rv)
        # keep stdout captured by executor, do not overwrite with harness payload
        normalized["stdout"] = tool_result.get("stdout", "") or normalized.get("stdout", "")
        normalized.setdefault("stderr", "")
        return normalized

    normalized["status"] = "runtime_error"
    normalized["error"] = "HumanEval harness did not produce expected result payload."
    return normalized


__all__ = [
    "CodeExtractor",
    "ToolSandbox",
    "run_humaneval_case",
]
