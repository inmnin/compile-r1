# Custom multi-turn rollout for compile-r1.
# The model follows <think></think><code></code><result></result><answer></answer> format.

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from humaneval_format import extract_answer_code, extract_last_action, has_successful_result, is_valid_sequence

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
    "timeout_seconds": _env_int("COMPILE_R1_TIMEOUT_SECONDS", 6),
    "memory_mb": _env_int("COMPILE_R1_MEMORY_MB", 768),
    "return_logprob": _env_bool("COMPILE_R1_RETURN_LOGPROB", True),
    "format_score": _env_float("COMPILE_R1_FORMAT_SCORE", 0.15),
    "tool_bonus": _env_float("COMPILE_R1_TOOL_BONUS", 0.05),
}

SEMAPHORE = asyncio.Semaphore(COMPILE_R1_CONFIGS["tool_concurrency"])


def postprocess_responses(resp: str) -> str:
    if "</answer>" in resp:
        idx = resp.rfind("</answer>") + len("</answer>")
        return resp[:idx]
    if "</code>" in resp:
        idx = resp.rfind("</code>") + len("</code>")
        return resp[:idx]
    return resp


async def _execute_code_action(code: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    async with SEMAPHORE:
        return await asyncio.to_thread(
            run_humaneval_in_sandbox,
            code=code,
            prompt=str(ground_truth.get("prompt", "")),
            test=str(ground_truth.get("test", "")),
            entry_point=str(ground_truth.get("entry_point", "")),
            timeout_seconds=COMPILE_R1_CONFIGS["timeout_seconds"],
            memory_mb=COMPILE_R1_CONFIGS["memory_mb"],
        )


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

    prompt_text = sample.prompt
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs = [] if COMPILE_R1_CONFIGS["return_logprob"] else None

    tool_call_count = 0
    last_tool_result: dict[str, Any] | None = None
    finish_reason_type = "stop"
    ground_truth = sample.label["ground_truth"]

    for _turn_idx in range(COMPILE_R1_CONFIGS["max_turns"]):
        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
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

        if COMPILE_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        if finish_reason_type == "length":
            break

        next_obs, done, tool_result = await execute_predictions(cur_response, ground_truth)
        if tool_result is not None:
            tool_call_count += 1
            last_tool_result = tool_result

        if done:
            break

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

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


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    response_text = sample.response
    format_ok = is_valid_sequence(response_text)
    answer_code = extract_answer_code(response_text)
    ground_truth = sample.label["ground_truth"]

    if answer_code:
        result = await _execute_code_action(answer_code, ground_truth)
        if result.get("passed") is True:
            return 1.0

    score = COMPILE_R1_CONFIGS["format_score"] if format_ok else 0.0
    if has_successful_result(response_text):
        score += COMPILE_R1_CONFIGS["tool_bonus"]

    return min(score, 0.99)
