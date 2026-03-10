# Custom multi-turn rollout for compile-r1.
# The model follows <think></think><code></code><result></result><answer></answer> format.

import asyncio
import atexit
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from humaneval_format import extract_answer_code, extract_last_action

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

TOOL_DIR = Path(__file__).resolve().parents[2] / "tools" / "python_sandbox"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from sandbox import run_humaneval_in_sandbox  # noqa: E402


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    return int(value) if value is not None and value != "" else default


def _env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    return float(value) if value is not None and value != "" else default


COMPILE_R1_CONFIGS = {
    "max_turns": _env_int("COMPILE_R1_MAX_TURNS", 3),
    "tool_concurrency": _env_int("COMPILE_R1_TOOL_CONCURRENCY", 32),
    "async_tool_workers": _env_int("COMPILE_R1_ASYNC_TOOL_WORKERS", _env_int("COMPILE_R1_TOOL_CONCURRENCY", 32)),
    "async_tool_timeout_buffer": _env_int("COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER", 3),
    "timeout_seconds": _env_int("COMPILE_R1_TIMEOUT_SECONDS", 6),
    "memory_mb": _env_int("COMPILE_R1_MEMORY_MB", 768),
    "return_logprob": _env_bool("COMPILE_R1_RETURN_LOGPROB", True),
    "format_score": _env_float("COMPILE_R1_FORMAT_SCORE", 0.02),
    "tool_bonus": _env_float("COMPILE_R1_TOOL_BONUS", 0.12),
    "tool_bonus_cap": _env_int("COMPILE_R1_TOOL_BONUS_CAP", 3),
    "tool_success_bonus": _env_float("COMPILE_R1_TOOL_SUCCESS_BONUS", 0.20),
    "tool_success_cap": _env_int("COMPILE_R1_TOOL_SUCCESS_CAP", 3),
    "tool_invalid_penalty": _env_float("COMPILE_R1_TOOL_INVALID_PENALTY", 0.0),
    "tool_overcall_penalty": _env_float("COMPILE_R1_TOOL_OVERCALL_PENALTY", 0.0),
    "tool_overcall_free_calls": _env_int("COMPILE_R1_TOOL_OVERCALL_FREE_CALLS", 3),
    "reward_pass_only": _env_bool("COMPILE_R1_REWARD_PASS_ONLY", True),
    "reward_clip_min": _env_float("COMPILE_R1_REWARD_CLIP_MIN", -1.0),
    "reward_clip_max": _env_float("COMPILE_R1_REWARD_CLIP_MAX", 2.0),
}

SEMAPHORE = asyncio.Semaphore(COMPILE_R1_CONFIGS["tool_concurrency"])
TOOL_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, COMPILE_R1_CONFIGS["async_tool_workers"]),
    thread_name_prefix="compile_r1_tool",
)
ACTION_STOP_TAGS = ["</code>", "</answer>"]
INVALID_TOOL_STATUSES = {"timeout", "runtime_error", "invalid_code", "sandbox_error", "parse_error"}
TOOL_PROMPT_GUIDANCE = (
    "\n\n[Tool Usage Reminder]\n"
    "When needed, you may write executable intermediate code or sub-code inside <code>...</code>. "
    "The environment will execute it and return detailed feedback inside <result>...</result>. "
    "You may call the tool for multiple rounds before the final <answer>...</answer>. "
    "Please follow the structured tags and keep your final answer inside <answer>."
)


def _shutdown_tool_executor() -> None:
    TOOL_EXECUTOR.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_tool_executor)


def _resolve_label(sample: Sample) -> tuple[str, dict[str, Any]]:
    label = sample.label
    target_answer = ""
    ground_truth: dict[str, Any] = {}

    if isinstance(label, dict):
        if isinstance(label.get("ground_truth"), dict):
            ground_truth = label["ground_truth"]

        if isinstance(label.get("target"), str):
            target_answer = label["target"]
        elif isinstance(label.get("answer"), str):
            target_answer = label["answer"]
        elif isinstance(label.get("canonical_solution"), str):
            target_answer = label["canonical_solution"]

        if not target_answer and isinstance(ground_truth.get("canonical_solution"), str):
            target_answer = ground_truth["canonical_solution"]
    elif isinstance(label, str):
        target_answer = label

    return target_answer.strip(), ground_truth


def _extract_test_cases(ground_truth: dict[str, Any]) -> list[str]:
    raw_cases = ground_truth.get("test_cases")
    cases: list[str] = []

    if isinstance(raw_cases, list):
        for item in raw_cases:
            text = str(item).strip()
            if text:
                cases.append(text)
    elif isinstance(raw_cases, str):
        for line in raw_cases.splitlines():
            text = line.strip()
            if text:
                cases.append(text)

    if not cases:
        test_code = str(ground_truth.get("test", "") or "")
        for line in test_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert "):
                cases.append(stripped)

    return cases


def _normalize_case_for_candidate(case: str, entry_point: str) -> str:
    text = str(case).strip()
    if not text:
        return text
    if entry_point:
        text = re.sub(rf"\b{re.escape(entry_point)}\s*\(", "candidate(", text)
    return text


def _single_case_test_code(case: str, entry_point: str) -> str:
    normalized = _normalize_case_for_candidate(case, entry_point)
    lines = [ln.rstrip() for ln in normalized.splitlines() if ln.strip()]
    if not lines:
        return "def check(candidate):\n    assert True"
    body = "\n".join(f"    {ln}" for ln in lines)
    return f"def check(candidate):\n{body}"


async def _answer_pass_rate(answer_code: str, ground_truth: dict[str, Any]) -> tuple[float, int, int] | None:
    entry_point = str(ground_truth.get("entry_point") or "").strip()
    prompt = str(ground_truth.get("prompt") or "").strip()
    cases = _extract_test_cases(ground_truth)
    if not entry_point or not prompt or not cases:
        return None

    tasks = []
    for case in cases:
        case_ground_truth = dict(ground_truth)
        case_ground_truth["test"] = _single_case_test_code(case, entry_point)
        tasks.append(_execute_code_action(answer_code, case_ground_truth))

    results = await asyncio.gather(*tasks)
    passed_cases = sum(1 for item in results if item.get("passed") is True)
    total_cases = len(results)
    if total_cases == 0:
        return None
    return passed_cases / total_cases, passed_cases, total_cases


def postprocess_responses(resp: str) -> str:
    if "</answer>" in resp:
        idx = resp.rfind("</answer>") + len("</answer>")
        return resp[:idx]
    if "</code>" in resp:
        idx = resp.rfind("</code>") + len("</code>")
        return resp[:idx]
    return resp


def _build_action_sampling_params(sampling_params: dict[str, Any], remaining_new_tokens: int | None) -> dict[str, Any]:
    """Inject action-closing stop tags so generation pauses as soon as an action closes."""
    params = dict(sampling_params)

    existing_stop = params.get("stop")
    if existing_stop is None:
        stops: list[str] = []
    elif isinstance(existing_stop, list):
        stops = list(existing_stop)
    else:
        stops = [str(existing_stop)]

    for tag in ACTION_STOP_TAGS:
        if tag not in stops:
            stops.append(tag)
    params["stop"] = stops

    if remaining_new_tokens is not None:
        params["max_new_tokens"] = max(0, int(remaining_new_tokens))

    return params


async def _execute_code_action(code: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    hard_timeout = max(
        1,
        int(COMPILE_R1_CONFIGS["timeout_seconds"]) + int(COMPILE_R1_CONFIGS["async_tool_timeout_buffer"]),
    )
    start_ts = time.time()
    async with SEMAPHORE:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            TOOL_EXECUTOR,
            run_humaneval_in_sandbox,
            code,
            str(ground_truth.get("prompt", "")),
            str(ground_truth.get("test", "")),
            str(ground_truth.get("entry_point", "")),
            int(COMPILE_R1_CONFIGS["timeout_seconds"]),
            int(COMPILE_R1_CONFIGS["memory_mb"]),
        )
        try:
            return await asyncio.wait_for(future, timeout=hard_timeout)
        except asyncio.TimeoutError:
            return {
                "passed": False,
                "status": "timeout",
                "error": f"Async tool wrapper timeout after {hard_timeout} seconds",
                "stdout": "",
                "stderr": "",
                "execution_seconds": round(time.time() - start_ts, 6),
                "backend": "async_wrapper",
            }
        except Exception as exc:
            return {
                "passed": False,
                "status": "sandbox_error",
                "error": f"Async tool execution failed: {type(exc).__name__}: {exc}",
                "stdout": "",
                "stderr": "",
                "execution_seconds": round(time.time() - start_ts, 6),
                "backend": "async_wrapper",
            }


def _is_valid_tool_result(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False

    status = str(result.get("status") or "").strip().lower()
    if status in INVALID_TOOL_STATUSES:
        return False
    if result.get("error"):
        return False

    if "passed" in result:
        return True

    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    return bool(status) and (stdout != "" or stderr != "")


def _append_tool_guidance(prompt_text: str) -> str:
    text = str(prompt_text or "")
    if "[Tool Usage Reminder]" in text:
        return text
    return text + TOOL_PROMPT_GUIDANCE


def _tail_text(text: str, max_chars: int = 1200) -> str:
    content = str(text or "")
    if len(content) <= max_chars:
        return content
    return content[-max_chars:]


def _extract_error_type(error_text: str) -> str:
    text = str(error_text or "").strip()
    if not text:
        return ""
    match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]+)\s*:", text)
    if match:
        return match.group(1)
    return text.split()[0]


def _extract_trace_line(traceback_text: str) -> int | None:
    text = str(traceback_text or "")
    line_matches = re.findall(r"line\s+(\d+)", text)
    if not line_matches:
        return None
    try:
        return int(line_matches[-1])
    except Exception:
        return None


def _code_excerpt_with_lineno(code: str, focus_line: int | None = None, radius: int = 3, max_lines: int = 20) -> str:
    lines = str(code or "").splitlines()
    if not lines:
        return ""
    if focus_line is None:
        selected = list(enumerate(lines[:max_lines], start=1))
    else:
        start = max(1, focus_line - radius)
        end = min(len(lines), focus_line + radius)
        selected = list(enumerate(lines[start - 1 : end], start=start))
    return "\n".join(f"{idx:>4}: {line}" for idx, line in selected)


def _build_tool_diagnostics(code: str, result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status") or "").strip()
    error_text = str(result.get("error") or "").strip()
    traceback_text = str(result.get("traceback") or "").strip()
    stdout_text = str(result.get("stdout") or "")
    stderr_text = str(result.get("stderr") or "")
    suspected_line = _extract_trace_line(traceback_text)
    error_type = _extract_error_type(error_text)

    timeout_detail = ""
    status_lower = status.lower()
    if status_lower == "timeout":
        timeout_detail = (
            f"Tool timed out. timeout_seconds={COMPILE_R1_CONFIGS['timeout_seconds']}, "
            f"async_timeout_seconds={int(COMPILE_R1_CONFIGS['timeout_seconds']) + int(COMPILE_R1_CONFIGS['async_tool_timeout_buffer'])}"
        )

    hint = ""
    if status_lower == "timeout":
        hint = "Your code likely exceeds the time limit. Try reducing complexity or fixing infinite loops."
    elif error_type:
        hint = f"Fix the {error_type} first, then call <code> again to verify."
    elif error_text:
        hint = "Inspect traceback/stdout/stderr and revise the code before next tool call."

    return {
        "status": status,
        "passed": bool(result.get("passed") is True),
        "error_type": error_type,
        "error_message": error_text,
        "timeout_detail": timeout_detail,
        "used_prompt_prefix": bool(result.get("used_prompt_prefix", False)),
        "execution_seconds": float(result.get("execution_seconds") or 0.0),
        "suspected_line": suspected_line,
        "code_excerpt": _code_excerpt_with_lineno(code, focus_line=suspected_line),
        "traceback_tail": _tail_text(traceback_text, 1600),
        "stdout_tail": _tail_text(stdout_text, 800),
        "stderr_tail": _tail_text(stderr_text, 800),
        "hint": hint,
    }


def _enrich_tool_result(code: str, result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result or {})
    payload["diagnosis"] = _build_tool_diagnostics(code, payload)
    return payload


def _has_strict_tag_format(response_text: str) -> bool:
    if not isinstance(response_text, str):
        return False

    think_open = response_text.find("<think>")
    think_close = response_text.find("</think>")
    code_open = response_text.find("<code>")
    code_close = response_text.find("</code>")
    answer_open = response_text.find("<answer>")
    answer_close = response_text.find("</answer>")

    if min(think_open, think_close, code_open, code_close, answer_open, answer_close) < 0:
        return False

    # Require strict order: think -> code -> answer.
    if not (think_open < think_close < code_open < code_close < answer_open < answer_close):
        return False

    return True


async def execute_predictions(prediction: str, ground_truth: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
    action, content = extract_last_action(prediction)

    if action == "code":
        if not content.strip():
            result = {
                "passed": False,
                "status": "invalid_code",
                "error": "Empty <code> block",
                "stdout": "",
                "stderr": "",
            }
        else:
            result = await _execute_code_action(content, ground_truth)
        result = _enrich_tool_result(content, result)
        next_obs = f"\n\n<result>{json.dumps(result, ensure_ascii=False)}</result>\n\n"
        return next_obs, False, result

    if action == "answer":
        return "", True, None

    next_obs = (
        "\nMy previous action is invalid. "
        "I must output either <code>...</code> to execute Python code or "
        "<answer>...</answer> to provide final code.\n"
    )
    return next_obs, False, None


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."
    if not isinstance(sample.prompt, str):
        raise TypeError("compile-r1 expects string prompt. Please enable --apply-chat-template.")

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text = _append_tool_guidance(sample.prompt)
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs = [] if COMPILE_R1_CONFIGS["return_logprob"] else None
    remaining_new_tokens = sampling_params.get("max_new_tokens")
    if remaining_new_tokens is not None:
        remaining_new_tokens = int(remaining_new_tokens)

    tool_call_count = 0
    valid_tool_call_count = 0
    invalid_tool_call_count = 0
    last_tool_result: dict[str, Any] | None = None
    finish_reason_type = "stop"
    _target_answer, ground_truth = _resolve_label(sample)

    for _turn_idx in range(COMPILE_R1_CONFIGS["max_turns"]):
        if remaining_new_tokens is not None and remaining_new_tokens <= 0:
            finish_reason_type = "length"
            break

        turn_sampling_params = _build_action_sampling_params(sampling_params, remaining_new_tokens)
        payload = {
            "text": prompt_text + response,
            "sampling_params": turn_sampling_params,
        }
        if COMPILE_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)
        finish_reason_type = output["meta_info"]["finish_reason"]["type"]

        if finish_reason_type == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        if COMPILE_R1_CONFIGS["return_logprob"]:
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure payload sets return_logprob=True."
                )
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            cur_response = postprocess_responses(cur_response)
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
            cur_response_log_probs = []

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)
        if remaining_new_tokens is not None:
            remaining_new_tokens -= len(cur_response_token_ids)

        if COMPILE_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        if finish_reason_type == "length":
            break

        next_obs, done, tool_result = await execute_predictions(cur_response, ground_truth)
        if tool_result is not None:
            tool_call_count += 1
            if _is_valid_tool_result(tool_result):
                valid_tool_call_count += 1
            else:
                invalid_tool_call_count += 1
            last_tool_result = tool_result

        if done:
            break

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        # Environment/tool feedback must not contribute to policy gradient updates.
        obs_loss_mask = [0] * len(obs_tokens_ids)
        loss_mask += obs_loss_mask
        assert all(v == 0 for v in obs_loss_mask), "Backfilled observation tokens must be fully masked."

        if COMPILE_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    sample.metadata["tool_call_count"] = tool_call_count
    sample.metadata["valid_tool_call_count"] = valid_tool_call_count
    sample.metadata["invalid_tool_call_count"] = invalid_tool_call_count
    if last_tool_result is not None:
        sample.metadata["last_tool_result"] = last_tool_result

    if COMPILE_R1_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    match finish_reason_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED
        case _:
            sample.status = Sample.Status.COMPLETED

    return sample


def _compose_reward(sample: Sample, pass_rate: float) -> float:
    # Strict reward-only mode: reward = 1 only when pass_rate == 1, else 0.
    reward = 1.0 if pass_rate >= 1.0 - 1e-12 else 0.0
    reward = float(min(max(reward, float(COMPILE_R1_CONFIGS["reward_clip_min"])), float(COMPILE_R1_CONFIGS["reward_clip_max"])))

    # Keep only the single reward channel in metadata.
    sample.metadata["reward/total"] = reward
    sample.metadata["reward/pass_only"] = reward
    return reward


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    response_text = sample.response
    answer_code = extract_answer_code(response_text)
    _target_answer, ground_truth = _resolve_label(sample)

    if not answer_code:
        sample.metadata["answer_pass_rate"] = 0.0
        sample.metadata["passed_cases"] = 0
        sample.metadata["total_cases"] = 0
        return _compose_reward(sample, pass_rate=0.0)

    case_score = await _answer_pass_rate(answer_code, ground_truth)
    if case_score is not None:
        pass_rate, passed_cases, total_cases = case_score
        sample.metadata["answer_pass_rate"] = float(pass_rate)
        sample.metadata["passed_cases"] = int(passed_cases)
        sample.metadata["total_cases"] = int(total_cases)
        return _compose_reward(sample, pass_rate=float(pass_rate))

    if all(k in ground_truth for k in ("prompt", "test", "entry_point")):
        result = await _execute_code_action(answer_code, ground_truth)
        pass_rate = 1.0 if result.get("passed") is True else 0.0
        sample.metadata["answer_pass_rate"] = pass_rate
        sample.metadata["passed_cases"] = int(pass_rate)
        sample.metadata["total_cases"] = 1
        return _compose_reward(sample, pass_rate=float(pass_rate))

    sample.metadata["answer_pass_rate"] = 0.0
    sample.metadata["passed_cases"] = 0
    sample.metadata["total_cases"] = 0
    return _compose_reward(sample, pass_rate=0.0)
