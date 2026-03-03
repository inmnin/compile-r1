import json
import re

ACTION_PATTERN = re.compile(r"<(code|answer)>(.*?)</\1>", re.DOTALL)
TAG_PATTERN = re.compile(r"</?(think|code|result|answer)>")
RESULT_PATTERN = re.compile(r"<result>(.*?)</result>", re.DOTALL)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def extract_last_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_last_action(text: str) -> tuple[str | None, str]:
    matches = list(ACTION_PATTERN.finditer(text))
    if not matches:
        return None, ""
    last = matches[-1]
    action = last.group(1)
    content = strip_code_fence(last.group(2).strip())
    return action, content


def extract_answer_code(text: str) -> str | None:
    answer = extract_last_tag(text, "answer")
    if answer is None:
        return None
    return strip_code_fence(answer)


def has_successful_result(text: str) -> bool:
    for result_block in RESULT_PATTERN.findall(text):
        raw = result_block.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if payload.get("passed") is True:
            return True
    return False


def is_valid_sequence(text: str) -> bool:
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
                # Nested tags are not supported in this protocol.
                return False
            stack.append(tag)

    if stack:
        return False
    if not closed_order:
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
