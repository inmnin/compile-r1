#!/usr/bin/env python3

from __future__ import annotations

PROMPT_VERSION = "compile_r1_natural_unified_v1"

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


def build_problem_block(task_prompt: str) -> str:
    return "Problem:\n```python\n" + task_prompt.rstrip() + "\n```"


def build_user_prompt_text(task_prompt: str) -> str:
    return (
        "Solve the Python programming task below. "
        "If uncertain, use <code> first and wait for <result>.\n\n"
        + build_problem_block(task_prompt)
    )


def build_user_prompt_messages(task_prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_user_prompt_text(task_prompt)}]


def build_alpaca_instruction() -> str:
    return SYSTEM_PROMPT.strip()


def build_alpaca_input(task_prompt: str) -> str:
    return build_user_prompt_text(task_prompt)

