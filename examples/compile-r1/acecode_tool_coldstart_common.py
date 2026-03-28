from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


PYTHON_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "strict": True,
        "description": (
            "Execute a short self-contained Python verification script to reduce uncertainty on a coding task. "
            "Use it only when execution can materially improve correctness, such as checking tricky edge cases, "
            "validating a candidate implementation on small examples, debugging suspected syntax/runtime issues, "
            "or comparing candidates when uncertain. Do not use it for trivial tasks when execution would not "
            "change the final answer. The code must be a complete runnable Python script. If you define helper "
            "functions, call them explicitly in the same script. Do not include Markdown fences. Do not use files, "
            "network access, subprocesses, or third-party packages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "A complete self-contained Python script to execute. It may define helper functions, "
                        "run assertions, and print diagnostic outputs. No Markdown fences."
                    ),
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

TOOL_SYSTEM_PROMPT = (
    "You are a Python coding assistant solving function-implementation tasks.\n"
    "You may use one tool:\n"
    "python_exec(code: string)\n"
    "Use the tool only when execution can materially reduce uncertainty, especially for:\n"
    "checking tricky edge cases,\n"
    "validating a candidate implementation on small examples,\n"
    "debugging suspected syntax or runtime issues,\n"
    "comparing candidate implementations when you are unsure.\n"
    "Do not use the tool for trivial tasks when execution would not change your final answer.\n"
    "When you call the tool:\n"
    "produce exactly one function call at a time,\n"
    "the assistant message for a tool call must contain no natural-language text and only the tool_calls field,\n"
    "set the assistant content to empty/null for tool-call turns,\n"
    "put only a complete self-contained Python script in the code argument,\n"
    "include any helper functions, assertions, or prints needed for verification,\n"
    "if you define helper functions, call them explicitly in the same script,\n"
    "do not send placeholders, stubs, pseudocode, or partial function definitions,\n"
    "do not include Markdown fences,\n"
    "do not rely on files, network access, subprocesses, or third-party packages.\n"
    "After receiving a tool result, either make another grounded tool call or provide the final answer.\n"
    "Only the last assistant message may contain the final answer.\n"
    "Your final answer must contain only the final Python solution implementing the required function or class.\n"
    "Do not include tests.\n"
    "Do not include explanation.\n"
    "Do not include Markdown fences."
)

NO_TOOL_SYSTEM_PROMPT = TOOL_SYSTEM_PROMPT

TASK_PROMPT_TEMPLATE = (
    "Solve the following Python coding task.\n\n"
    "Return only the final Python solution in your last assistant message.\n\n"
    "Task:\n{question}"
)

RISK_KEYWORDS = {
    "recursion": 2,
    "recursive": 2,
    "regex": 2,
    "regular expression": 2,
    "parse": 2,
    "parser": 2,
    "decode": 2,
    "escape": 2,
    "unicode": 2,
    "preserve punctuation": 2,
    "punctuation": 1,
    "preserve spaces": 2,
    "exact formatting": 2,
    "bitwise": 2,
    "without using +": 3,
    "without using built": 2,
    "without sorting": 2,
    "version": 2,
    "timedelta": 2,
    "date": 1,
    "maze": 3,
    "graph": 2,
    "path": 2,
    "search": 2,
    "edge case": 1,
    "invalid input": 1,
    "mixed input types": 1,
}

DIRECT_HINT_KEYWORDS = (
    "list of",
    "dictionary",
    "mapping",
    "replace",
    "filter",
    "count",
    "boolean",
    "return true",
    "return false",
    "sum of",
    "format",
    "concatenate",
    "append",
)

TAG_RE = re.compile(r"</?(think|code|result|answer)>", re.IGNORECASE)
FENCED_CODE_RE = re.compile(r"^\s*```(?:python)?\s*(.*?)```\s*$", re.IGNORECASE | re.DOTALL)


@dataclass
class StaticToolNeed:
    bucket: str
    score: int
    num_tests: int
    max_inference_pass_rate: float
    mean_inference_pass_rate: float
    risk_flags: list[str]


def build_task_prompt(question: str) -> str:
    return TASK_PROMPT_TEMPLATE.format(question=str(question).strip())


def normalize_test_cases(raw: Any) -> list[str]:
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        return [text]
    return [str(raw).strip()] if str(raw).strip() else []


def normalize_context_messages(raw: Any) -> list[dict[str, str]]:
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return []
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(item.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def inference_pass_rates(raw: Any) -> list[float]:
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = item.get("pass_rate")
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def score_static_tool_need(question: str, test_cases: list[str], inferences: Any) -> StaticToolNeed:
    text = str(question or "").lower()
    pass_rates = inference_pass_rates(inferences)
    risk_flags: list[str] = []
    score = 0

    for keyword, weight in RISK_KEYWORDS.items():
        if keyword in text:
            score += weight
            risk_flags.append(keyword)

    if len(test_cases) >= 14:
        score += 1
        risk_flags.append("many_test_cases")
    if len(test_cases) >= 20:
        score += 1
        risk_flags.append("very_many_test_cases")

    if pass_rates:
        max_pass = max(pass_rates)
        mean_pass = sum(pass_rates) / len(pass_rates)
        if max_pass < 0.8:
            score += 1
            risk_flags.append("low_max_inference_pass")
        if mean_pass < 0.5:
            score += 1
            risk_flags.append("low_mean_inference_pass")
    else:
        max_pass = 0.0
        mean_pass = 0.0

    if score == 0 and any(hint in text for hint in DIRECT_HINT_KEYWORDS):
        risk_flags.append("direct_hint")

    bucket = "tool_candidate" if score >= 2 else "direct_candidate"
    return StaticToolNeed(
        bucket=bucket,
        score=score,
        num_tests=len(test_cases),
        max_inference_pass_rate=max_pass,
        mean_inference_pass_rate=mean_pass,
        risk_flags=risk_flags,
    )


def contains_markdown_fence(text: str) -> bool:
    return "```" in str(text or "")


def strip_single_markdown_fence(text: str) -> tuple[str, bool]:
    raw = str(text or "").strip()
    match = FENCED_CODE_RE.match(raw)
    if not match:
        return raw, False
    return match.group(1).strip(), True


def parse_python_module(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def is_pure_python_solution(code: str) -> tuple[bool, str]:
    text = str(code or "").strip()
    if not text:
        return False, "empty_final_answer"
    if contains_markdown_fence(text):
        return False, "markdown_fence_in_final_answer"
    if TAG_RE.search(text):
        return False, "old_tag_protocol_in_final_answer"
    tree = parse_python_module(text)
    if tree is None:
        return False, "final_answer_not_parseable_python"
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        if isinstance(tree.body[0].value.value, str):
            return False, "final_answer_contains_free_text"
    for node in tree.body:
        if isinstance(node, ast.Assert):
            return False, "final_answer_contains_tests"
    return True, ""


def is_self_contained_tool_script(code: str) -> tuple[bool, str]:
    text = str(code or "").strip()
    if not text:
        return False, "empty_tool_code"
    if contains_markdown_fence(text):
        return False, "markdown_fence_in_tool_code"
    tree = parse_python_module(text)
    if tree is None:
        return False, "tool_code_not_parseable_python"

    executable_nodes = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "result" in targets:
                executable_nodes += 1
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            executable_nodes += 1
            continue
        if isinstance(node, (ast.Assert, ast.If, ast.For, ast.While, ast.With, ast.Try)):
            executable_nodes += 1
            continue

    if executable_nodes == 0:
        return False, "tool_code_not_self_contained"
    return True, ""


def normalize_tool_code_for_compare(code: str) -> str:
    return "\n".join(line.rstrip() for line in str(code or "").strip().splitlines() if line.strip())


def observation_from_run_result(result: dict[str, Any], max_stdout_chars: int = 4000, max_error_chars: int = 1500) -> dict[str, Any]:
    stdout = str(result.get("stdout", "") or "")
    error = str(result.get("error", "") or "")
    return {
        "status": str(result.get("status", "")),
        "stdout": stdout[:max_stdout_chars],
        "error": error[:max_error_chars],
        "return_value": result.get("return_value"),
    }


def json_dumps_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
