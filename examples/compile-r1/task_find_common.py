#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import ast
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "strict": True,
        "description": (
            "Execute a short self-contained Python script to compute, verify, or disambiguate a numeric or table reasoning answer. "
            "Use it only when execution can materially improve correctness, especially for arithmetic, scale conversion, table lookup, "
            "percentage/rate questions, or multi-step combination reasoning. "
            "The code argument must be real executable Python, not prose, not a plan, not comments-only text, and not a partial snippet. "
            "Do not use the tool just to restate the question, quote the context, or print an English sentence. "
            "The script should do actual computation or explicit checking on concrete values from the input. "
            "A valid tool call usually contains assignments, arithmetic, and at least one print(...) or assert ... statement so the result is observable. "
            "Good example: revenue = 9896 / 0.236\nprint(revenue)\n"
            "Bad example: # compute 9896 / 0.236\n"
            "Bad example: print('The context says interest expense changes by $3.8 million')\n"
            "Do not include Markdown fences. Do not use files, network access, subprocesses, or third-party packages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "A complete self-contained Python script. "
                        "It must contain executable Python statements and should print or assert observable results. "
                        "No Markdown fences. No natural-language explanation. No comments-only content."
                    ),
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


HARD_TOOL_SUFFIX = (
    "For this task, if you use the tool, the tool call should target the hardest or most ambiguous part of the problem.\n"
    "A good tool call should usually do at least one of the following:\n"
    "- verify a nontrivial ratio, percentage, or scale conversion,\n"
    "- compare two candidate formulas or interpretations,\n"
    "- check a likely failure mode with an assert,\n"
    "- validate that the chosen values from the table/context are the correct ones.\n"
    "Avoid trivial print-only checks unless they directly resolve the main uncertainty.\n"
)


NO_TOOL_SYSTEM_PROMPT = (
    "You solve numeric, table, and mixed table-text reasoning questions.\n"
    "Return only the final answer.\n"
    "Output a single normalized answer span only.\n"
    "Do not output an equation, derivation, explanation, or sentence.\n"
    "Do not output reasoning.\n"
    "Do not output code.\n"
    "Do not output Markdown.\n"
    "If the answer is numeric, preserve the dataset's scale convention exactly.\n"
    "If the answer is a percentage, include the percent sign only if the dataset answer uses it.\n"
)


TOOL_SYSTEM_PROMPT = (
    "You solve numeric, table, and mixed table-text reasoning questions.\n"
    "You may use one tool: python_exec(code: string).\n"
    "Use the tool only when it can materially improve correctness, especially for arithmetic, table lookup, scale conversion, or multi-step combination reasoning.\n"
    "Tool-call quality rules are strict.\n"
    "When you call the tool:\n"
    "- make exactly one tool call at a time,\n"
    "- the assistant tool-call message must contain no natural-language text,\n"
    "- set assistant content to empty/null for tool-call turns,\n"
    "- put only a complete self-contained Python script in the code argument,\n"
    "- include prints or assertions needed for verification,\n"
    "- do not put comments-only text or natural language in code,\n"
    "- the script must contain executable Python statements and should print observable results,\n"
    "- if the code argument is not executable Python, the tool call is invalid,\n"
    "- if the code argument is only comments or only an English explanation, the tool call is invalid,\n"
    "- do not include Markdown fences,\n"
    "- do not use files, network access, subprocesses, or third-party packages.\n"
    "Bad tool call code: # compute 9896 / 0.236\n"
    "Bad tool call code: print('The context says the change is $3.8 million')\n"
    "Good tool call code: value = 9896 / 0.236\nprint(value)\n"
    "Good tool call code: old = 991.1\nnew = 959.2\nprint((new - old) / old * 100)\n"
    "Keep it concise; for simple checks aim for <= 8 lines, but for hard tasks you may use up to 16 lines if needed to verify the main uncertainty.\n"
    "Do not use the tool only for the final one-line arithmetic step if the main uncertainty lies earlier in scale interpretation, value selection, or formula composition.\n"
    "Use the tool only to resolve the single most important uncertainty in the task.\n"
    "Prefer checking the most error-prone part of the reasoning over checking an obvious final arithmetic operation.\n"
    "Prefer assertions, candidate comparison, or minimal falsification tests over generic print-only checks whenever possible.\n"
    "If two formulas or interpretations are plausible, use the tool to disambiguate them.\n"
    "If you define helper functions, call them explicitly in the same script.\n"
    "If you use the tool, the tool code must do real computation and expose an observable result.\n"
    "After receiving a tool result, either make another grounded tool call or give the final answer.\n"
    "Only the last assistant message may contain the final answer.\n"
    "Return only the final answer.\n"
    "Output a single normalized answer span only.\n"
    "Do not output an equation, derivation, explanation, or sentence.\n"
    "Do not output reasoning.\n"
    "Do not output Markdown.\n"
    "If the answer is numeric, preserve the dataset's scale convention exactly.\n"
)


def _explicit_tool_system_prompt(*, require_tool: bool) -> str:
    base = (
        "You solve numeric, table, and mixed table-text reasoning questions.\n"
        "You may use one external tool named python_exec.\n"
        "You are NOT using API-native function calling in this conversation.\n"
        "Instead, if you want to use the tool, you must output exactly one XML block in the assistant message and nothing else:\n"
        "<tool_call>\n"
        "{\"name\":\"python_exec\",\"arguments\":{\"code\":\"...executable python...\"}}\n"
        "</tool_call>\n"
        "Rules for the code field:\n"
        "- it must be executable Python,\n"
        "- it must be a complete self-contained script,\n"
        "- it must not contain comments,\n"
        "- it must not be comments-only,\n"
        "- it must not be natural-language prose,\n"
        "- it must not merely print or restate a sentence from the question, table, or context,\n"
        "- it should perform actual arithmetic, comparison, or explicit checking on concrete values,\n"
        "- keep it concise; for simple checks aim for <= 8 lines, but for hard tasks you may use up to 16 lines if needed to verify the main uncertainty,\n"
        "- it should usually print(...) or assert ... so the result is observable,\n"
        "- it must not contain Markdown fences,\n"
        "- it must not use files, network access, subprocesses, or third-party packages.\n"
        "Do not use the tool only for the final one-line arithmetic step if the main uncertainty lies earlier in scale interpretation, value selection, or formula composition.\n"
        "Use the tool only to resolve the single most important uncertainty in the task.\n"
        "Prefer checking the most error-prone part of the reasoning over checking an obvious final arithmetic operation.\n"
        "Prefer assertions, candidate comparison, or minimal falsification tests over generic print-only checks whenever possible.\n"
        "If two formulas or interpretations are plausible, use the tool to disambiguate them.\n"
        "If you define helper functions, call them explicitly in the same script.\n"
        "After a tool call, you will receive a user message wrapped in <tool_response>...</tool_response>.\n"
        "Then you may either output another single <tool_call>...</tool_call> block or output the final answer.\n"
        "Only the final assistant message may contain the final answer.\n"
        "Return only the final answer in the last assistant message.\n"
        "Output a single normalized answer span only.\n"
        "Do not output an equation, derivation, explanation, or sentence.\n"
        "Do not output reasoning.\n"
        "Do not output Markdown.\n"
        "If the answer is numeric, preserve the dataset's scale convention exactly.\n"
        "Bad tool call example:\n"
        "<tool_call>\n"
        "{\"name\":\"python_exec\",\"arguments\":{\"code\":\"# compute 9896 / 0.236\"}}\n"
        "</tool_call>\n"
        "Bad tool call example:\n"
        "<tool_call>\n"
        "{\"name\":\"python_exec\",\"arguments\":{\"code\":\"print('The context says the change is $3.8 million')\"}}\n"
        "</tool_call>\n"
        "Good tool call example:\n"
        "<tool_call>\n"
        "{\"name\":\"python_exec\",\"arguments\":{\"code\":\"value = 9896 / 0.236\\nprint(value)\"}}\n"
        "</tool_call>\n"
    )
    if require_tool:
        base += "For this run, you must make a tool call before giving the final answer.\n"
    return base


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def render_question(row: dict[str, Any]) -> str:
    blocks = [f"Dataset: {row['dataset']}"]
    if row.get("table"):
        blocks.append("Table:\n" + str(row["table"]).strip())
    if row.get("context"):
        blocks.append("Context:\n" + str(row["context"]).strip())
    blocks.append("Question:\n" + str(row["question"]).strip())
    blocks.append("Return only the final answer.")
    return "\n\n".join(blocks)


def build_no_tool_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": NO_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": render_question(row)},
    ]


def build_tool_messages(row: dict[str, Any], *, hard_mode: bool = False) -> list[dict[str, str]]:
    user_content = (
        render_question(row)
        + "\n\nTool-use contract:\n"
        + "- If you call python_exec, the code argument must be executable Python.\n"
        + "- The code must not be comments-only, prose, or a plan.\n"
        + "- The code should usually end with print(...) or assert ... so the result is visible.\n"
        + "- The assistant tool-call turn must contain no normal text.\n"
        + "- The final assistant message must contain only the final answer.\n"
    )
    if hard_mode:
        user_content += "\n" + HARD_TOOL_SUFFIX
    return [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_explicit_tool_messages(row: dict[str, Any], *, require_tool: bool, hard_mode: bool = False) -> list[dict[str, str]]:
    user_content = (
        render_question(row)
        + "\n\nExplicit tool protocol:\n"
        + "- If you use the tool, output exactly one <tool_call>...</tool_call> block and nothing else.\n"
        + "- The JSON inside <tool_call> must have name=python_exec and arguments.code=<python script>.\n"
        + "- The code must be executable Python, must not contain comments, must not be comments-only, must not be prose, and should usually print or assert something.\n"
        + "- The code must do actual computation or explicit checking on concrete values; it must not just print or restate an English sentence from the prompt.\n"
        + "- Keep the code concise; for simple checks aim for <= 8 lines, but for hard tasks you may use up to 16 lines if needed to verify the main uncertainty.\n"
        + "- After you receive <tool_response>...</tool_response>, either output another single <tool_call>...</tool_call> block or output only the final answer.\n"
        + "- The last assistant message must contain only the final answer.\n"
    )
    if hard_mode:
        user_content += "\n" + HARD_TOOL_SUFFIX
    return [
        {"role": "system", "content": _explicit_tool_system_prompt(require_tool=require_tool)},
        {"role": "user", "content": user_content},
    ]


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?")


def strip_answer_text(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    boxed = _BOXED_RE.findall(s)
    if boxed:
        s = boxed[-1].strip()
    s = s.replace("<answer>", "").replace("</answer>", "").strip()
    s = s.splitlines()[-1].strip() if "\n" in s else s
    return s


def parse_numeric_value(text: str, *, scale_hint: str = "") -> float | None:
    s = strip_answer_text(text)
    if not s:
        return None
    candidate = s.replace("−", "-").strip()
    if "=" in candidate:
        candidate = candidate.rsplit("=", 1)[-1].strip()
    if " or " in candidate.lower():
        candidate = re.split(r"\bor\b", candidate, flags=re.I)[-1].strip()
    matches = list(_NUMBER_RE.finditer(candidate))
    if not matches:
        matches = list(_NUMBER_RE.finditer(s.replace("−", "-")))
    if not matches:
        return None
    token = matches[-1].group(0).replace(",", "").strip()
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    try:
        if "/" in token and token.count("/") == 1:
            value = float(Fraction(token))
        else:
            value = float(token)
    except Exception:
        return None
    if is_percent or str(scale_hint).strip().lower() == "percent":
        return value
    return value


def normalize_text_answer(text: str) -> str:
    return re.sub(r"\\s+", " ", strip_answer_text(text)).strip().lower()


def grade_prediction(row: dict[str, Any], pred_text: str) -> tuple[bool, dict[str, Any]]:
    gold = str(row.get("answer", "")).strip()
    dataset = str(row.get("dataset", ""))
    meta = row.get("meta") or {}
    scale = str(meta.get("scale", "") or "")

    gold_num = parse_numeric_value(gold, scale_hint=scale)
    pred_num = parse_numeric_value(pred_text, scale_hint=scale)
    if gold_num is not None and pred_num is not None:
        ok = math.isclose(gold_num, pred_num, rel_tol=1e-5, abs_tol=1e-5)
        return ok, {
            "type": "numeric",
            "gold": gold,
            "pred": strip_answer_text(pred_text),
            "gold_num": gold_num,
            "pred_num": pred_num,
            "dataset": dataset,
            "scale": scale,
        }

    gold_norm = normalize_text_answer(gold)
    pred_norm = normalize_text_answer(pred_text)
    ok = gold_norm == pred_norm
    return ok, {
        "type": "exact",
        "gold": gold,
        "pred": strip_answer_text(pred_text),
        "gold_norm": gold_norm,
        "pred_norm": pred_norm,
        "dataset": dataset,
    }


def classify_error_type(row: dict[str, Any]) -> str:
    reasons = set((row.get("meta") or {}).get("candidate_hard_reason") or [])
    if "hybrid" in reasons:
        return "table_text_hybrid"
    if "scale" in reasons:
        return "scale"
    if "table" in reasons:
        return "table_lookup"
    if "multi_step" in reasons:
        return "multi_step_numeric"
    return "other"


def is_executable_python_script(code: str) -> tuple[bool, str]:
    text = str(code or "")
    if not text.strip():
        return False, "empty_script"
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False, "syntax_error"
    if not tree.body:
        return False, "empty_script"

    def _is_docstring_expr(stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(getattr(stmt, "value", None), ast.Constant)
            and isinstance(getattr(stmt.value, "value", None), str)
        )

    effectful = False
    for stmt in tree.body:
        if _is_docstring_expr(stmt):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Pass)):
            continue
        effectful = True
        break
    if not effectful:
        return False, "no_top_level_execution"
    return True, ""


def has_informative_tool_result(messages: list[dict[str, Any]]) -> tuple[bool, str]:
    for message in messages:
        if str(message.get("role", "")) != "tool":
            continue
        content = message.get("content", "")
        try:
            payload = json.loads(content) if isinstance(content, str) else content
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status", "")) != "success":
            continue
        stdout = str(payload.get("stdout", "") or "").strip()
        return_value = payload.get("return_value")
        if stdout:
            return True, ""
        if return_value not in (None, ""):
            return True, ""
    return False, "tool_success_not_informative"


@dataclass
class KeepDecision:
    keep: bool
    reason: str
    error_type: str


def _question_blob(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("question", "") or ""),
            str(row.get("table", "") or ""),
            str(row.get("context", "") or ""),
        ]
    ).lower()


def _iter_tool_codes(messages: list[dict[str, Any]]) -> list[str]:
    codes: list[str] = []
    for message in messages:
        if str(message.get("role", "")) != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            try:
                parsed = json.loads(tool_call.get("function", {}).get("arguments", "") or "{}")
            except Exception:
                continue
            if isinstance(parsed, dict):
                code = str(parsed.get("code", "") or "")
                if code.strip():
                    codes.append(code)
    return codes


def analyze_tool_code_utility(code: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
    text = str(code or "")
    lowered = text.lower()
    tags: set[str] = set()
    reasons: list[str] = []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {
            "tags": [],
            "low_utility": True,
            "reasons": ["syntax_error"],
            "assert_count": 0,
            "comparison_count": 0,
            "print_count": 0,
            "line_count": len(lines),
        }

    assert_count = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    comparison_count = sum(isinstance(node, ast.Compare) for node in ast.walk(tree))
    if_count = sum(isinstance(node, ast.If) for node in ast.walk(tree))
    print_count = 0
    assign_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            assign_count += 1
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                print_count += 1

    if assert_count:
        tags.add("assert_check")
    if comparison_count and (
        assert_count
        or if_count
        or any(token in lowered for token in ["candidate", "hypothesis", "formula", "option", "compare", "vs"])
    ):
        tags.add("comparison")

    blob = _question_blob(row or {})
    if any(token in blob for token in ["percent", "percentage", "ratio", "margin", "growth", "share", "basis points", "bps"]):
        if any(token in lowered for token in ["/", "* 100", "100 *", "percent", "ratio", "margin", "growth", "share"]):
            tags.add("scale_ratio_check")

    if (row or {}).get("meta", {}).get("has_table") and (row or {}).get("meta", {}).get("has_text_context"):
        if assign_count >= 2 or any(token in lowered for token in ["table", "context", "selected", "chosen", "value"]):
            tags.add("value_selection_check")

    if print_count >= 1 and assert_count == 0 and if_count == 0:
        tags.add("print_only")

    if len(lines) <= 4 and print_count == 1 and assert_count == 0 and comparison_count == 0:
        reasons.append("single_short_arithmetic")

    high_value = bool(tags & {"assert_check", "comparison", "scale_ratio_check", "value_selection_check"})
    low_utility = False
    if "print_only" in tags and not (tags & {"assert_check", "comparison", "value_selection_check"}):
        low_utility = True
        reasons.append("print_only_no_verification")
    if not high_value:
        low_utility = True
        reasons.append("no_high_value_signal")

    return {
        "tags": sorted(tags),
        "low_utility": low_utility,
        "reasons": reasons,
        "assert_count": assert_count,
        "comparison_count": comparison_count,
        "print_count": print_count,
        "line_count": len(lines),
    }


def summarize_tool_utility(messages: list[dict[str, Any]], row: dict[str, Any] | None = None) -> dict[str, Any]:
    codes = _iter_tool_codes(messages)
    call_summaries = [analyze_tool_code_utility(code, row=row) for code in codes]
    distinct_codes = len({code.strip() for code in codes if code.strip()})
    assert_calls = sum(1 for item in call_summaries if item["assert_count"] > 0)
    comparison_calls = sum(1 for item in call_summaries if "comparison" in item["tags"])
    low_utility_calls = sum(1 for item in call_summaries if item["low_utility"])
    high_value_calls = sum(1 for item in call_summaries if not item["low_utility"])
    return {
        "num_calls": len(codes),
        "assert_calls": assert_calls,
        "comparison_calls": comparison_calls,
        "low_utility_calls": low_utility_calls,
        "high_value_calls": high_value_calls,
        "all_low_utility": bool(codes) and low_utility_calls == len(codes),
        "any_low_utility": low_utility_calls > 0,
        "repair_like": len(codes) >= 2 and distinct_codes >= 2,
        "call_summaries": call_summaries,
    }


def keep_tool_positive(
    row: dict[str, Any],
    no_tool_ok: bool,
    tool_ok: bool,
    tool_num_calls: int,
    tool_success_calls: int,
    messages: list[dict[str, Any]] | None = None,
) -> KeepDecision:
    error_type = classify_error_type(row)
    if no_tool_ok:
        return KeepDecision(False, "no_tool_already_correct", error_type)
    if not tool_ok:
        return KeepDecision(False, "tool_not_correct", error_type)
    if tool_num_calls < 1:
        return KeepDecision(False, "missing_tool_call", error_type)
    if tool_success_calls < 1:
        return KeepDecision(False, "tool_never_succeeded", error_type)
    if messages:
        informative, reason = has_informative_tool_result(messages)
        if not informative:
            return KeepDecision(False, reason, error_type)
        utility = summarize_tool_utility(messages, row=row)
        if utility["all_low_utility"]:
            return KeepDecision(False, "tool_call_low_utility", error_type)
    if error_type not in {"table_text_hybrid", "scale", "table_lookup", "multi_step_numeric"}:
        return KeepDecision(False, "error_type_not_target", error_type)
    return KeepDecision(True, "tool_positive", error_type)
