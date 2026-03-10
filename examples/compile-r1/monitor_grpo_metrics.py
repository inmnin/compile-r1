#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class TrainState:
    step: int | None = None
    train_loss: float | None = None
    entropy: float | None = None
    truncated_ratio: float | None = None
    rollout_id: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor compile-r1 GRPO metrics from logs and rollout_data.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--interval-sec", type=int, default=300)
    p.add_argument("--duration-sec", type=int, default=7200)
    p.add_argument("--output-log", type=Path, default=None)
    return p.parse_args()


def _sample_attr(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _sample_metadata(sample: Any) -> dict[str, Any]:
    md = _sample_attr(sample, "metadata", {})
    return md if isinstance(md, dict) else {}


def _latest_numeric_pt(rollout_dir: Path) -> Path | None:
    files = [Path(p) for p in glob.glob(str(rollout_dir / "*.pt")) if Path(p).stem.isdigit()]
    if not files:
        return None
    return max(files, key=lambda p: int(p.stem))


def _latest_eval_pt(rollout_dir: Path) -> Path | None:
    files = [Path(p) for p in glob.glob(str(rollout_dir / "eval_*.pt"))]
    if not files:
        return None
    def _k(path: Path) -> int:
        name = path.stem
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return -1
    return max(files, key=_k)


def _load_rollout_metrics(path: Path) -> dict[str, float]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    samples = []
    if isinstance(obj, dict):
        if isinstance(obj.get("samples"), list):
            samples = obj["samples"]
        elif isinstance(obj.get("rollout_samples"), list):
            samples = obj["rollout_samples"]
        elif isinstance(obj.get("data"), list):
            samples = obj["data"]

    if not samples:
        return {}

    total_reward = []
    answer_pass_rate = []
    reward_base_pass = []
    reward_tool_success_bonus = []
    reward_format_bonus = []
    reward_invalid_tool_penalty = []
    reward_overcall_penalty = []
    tool_call_count = []
    valid_tool_call_count = []
    invalid_tool_call_count = []

    for s in samples:
        md = _sample_metadata(s)
        reward = _sample_attr(s, "reward", None)
        if isinstance(reward, (int, float)):
            total_reward.append(float(reward))
        if isinstance(md.get("answer_pass_rate"), (int, float)):
            answer_pass_rate.append(float(md["answer_pass_rate"]))
        if isinstance(md.get("reward/base_pass"), (int, float)):
            reward_base_pass.append(float(md["reward/base_pass"]))
        if isinstance(md.get("reward/tool_success_bonus"), (int, float)):
            reward_tool_success_bonus.append(float(md["reward/tool_success_bonus"]))
        if isinstance(md.get("reward/format_bonus"), (int, float)):
            reward_format_bonus.append(float(md["reward/format_bonus"]))
        if isinstance(md.get("reward/invalid_tool_penalty"), (int, float)):
            reward_invalid_tool_penalty.append(float(md["reward/invalid_tool_penalty"]))
        if isinstance(md.get("reward/overcall_penalty"), (int, float)):
            reward_overcall_penalty.append(float(md["reward/overcall_penalty"]))
        if isinstance(md.get("tool_call_count"), (int, float)):
            tool_call_count.append(float(md["tool_call_count"]))
        if isinstance(md.get("valid_tool_call_count"), (int, float)):
            valid_tool_call_count.append(float(md["valid_tool_call_count"]))
        if isinstance(md.get("invalid_tool_call_count"), (int, float)):
            invalid_tool_call_count.append(float(md["invalid_tool_call_count"]))

    out: dict[str, float] = {}
    if total_reward:
        out["reward_total_mean"] = float(np.mean(total_reward))
    if answer_pass_rate:
        out["answer_pass_rate_mean"] = float(np.mean(answer_pass_rate))
    if reward_base_pass:
        out["reward_base_pass_mean"] = float(np.mean(reward_base_pass))
    if reward_tool_success_bonus:
        out["reward_tool_success_bonus_mean"] = float(np.mean(reward_tool_success_bonus))
    if reward_format_bonus:
        out["reward_format_bonus_mean"] = float(np.mean(reward_format_bonus))
    if reward_invalid_tool_penalty:
        out["reward_invalid_tool_penalty_mean"] = float(np.mean(reward_invalid_tool_penalty))
    if reward_overcall_penalty:
        out["reward_overcall_penalty_mean"] = float(np.mean(reward_overcall_penalty))
    if tool_call_count:
        out["tool_call_count_mean"] = float(np.mean(tool_call_count))
    if valid_tool_call_count:
        out["valid_tool_call_count_mean"] = float(np.mean(valid_tool_call_count))
    if invalid_tool_call_count:
        out["invalid_tool_call_count_mean"] = float(np.mean(invalid_tool_call_count))
    return out


def _load_eval_metrics(path: Path) -> dict[str, float]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    samples = obj.get("samples", []) if isinstance(obj, dict) else []
    if not isinstance(samples, list) or not samples:
        return {}

    rewards = []
    for s in samples:
        r = _sample_attr(s, "reward", None)
        if isinstance(r, (int, float)):
            rewards.append(float(r))

    if not rewards:
        return {}

    arr = np.array(rewards, dtype=float)
    out = {
        "eval_reward_mean": float(np.mean(arr)),
        "eval_pass_rate_mean": float(np.mean(arr)),
        "eval_acc_strict": float(np.mean(arr >= 1.0 - 1e-12)),
    }
    return out


def _parse_train_state(log_path: Path) -> TrainState:
    state = TrainState()
    if not log_path.exists():
        return state

    ansi_pat = re.compile(r"\x1b\[[0-9;]*m")
    step_pat = re.compile(
        r"step\s+(\d+):\s+\{.*?'train/loss':\s*([-+]?[\d.]+(?:e[-+]?\d+)?)"
        r".*?'train/entropy_loss':\s*([-+]?[\d.]+(?:e[-+]?\d+)?)"
    )
    trunc_pat = re.compile(
        r"perf\s+(\d+):\s+\{.*?'rollout/truncated_ratio':\s*([-+]?[\d.]+(?:e[-+]?\d+)?)"
    )

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean_line = ansi_pat.sub("", line)
            m = step_pat.search(clean_line)
            if m:
                state.step = int(m.group(1))
                state.train_loss = float(m.group(2))
                state.entropy = float(m.group(3))
            m2 = trunc_pat.search(clean_line)
            if m2:
                state.rollout_id = int(m2.group(1))
                state.truncated_ratio = float(m2.group(2))
    return state


def _fmt(v: float | int | None) -> str:
    if v is None:
        return "na"
    if isinstance(v, int):
        return str(v)
    return f"{v:.6f}"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    log_path = run_dir / "logs" / "train_console.log"
    rollout_dir = run_dir / "debug" / "rollout_data"

    out_path = args.output_log or (run_dir / "logs" / "metrics_monitor.log")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    end_ts = time.time() + max(1, int(args.duration_sec))
    while time.time() <= end_ts:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        state = _parse_train_state(log_path)

        rollout_pt = _latest_numeric_pt(rollout_dir)
        rollout_metrics = _load_rollout_metrics(rollout_pt) if rollout_pt else {}
        rollout_id = int(rollout_pt.stem) if rollout_pt else None

        eval_pt = _latest_eval_pt(rollout_dir)
        eval_metrics = _load_eval_metrics(eval_pt) if eval_pt else {}
        eval_id = None
        if eval_pt:
            try:
                eval_id = int(eval_pt.stem.split("_", 1)[1])
            except Exception:
                eval_id = None

        line = (
            f"[{ts}] step={_fmt(state.step)} rollout={_fmt(rollout_id)} "
            f"reward_total={_fmt(rollout_metrics.get('reward_total_mean'))} "
            f"reward_base={_fmt(rollout_metrics.get('reward_base_pass_mean'))} "
            f"reward_tool_bonus={_fmt(rollout_metrics.get('reward_tool_success_bonus_mean'))} "
            f"reward_format_bonus={_fmt(rollout_metrics.get('reward_format_bonus_mean'))} "
            f"reward_invalid_penalty={_fmt(rollout_metrics.get('reward_invalid_tool_penalty_mean'))} "
            f"reward_overcall_penalty={_fmt(rollout_metrics.get('reward_overcall_penalty_mean'))} "
            f"answer_pass_rate={_fmt(rollout_metrics.get('answer_pass_rate_mean'))} "
            f"loss={_fmt(state.train_loss)} entropy={_fmt(state.entropy)} "
            f"truncated_ratio={_fmt(state.truncated_ratio)} "
            f"tool_calls={_fmt(rollout_metrics.get('tool_call_count_mean'))} "
            f"tool_calls_valid={_fmt(rollout_metrics.get('valid_tool_call_count_mean'))} "
            f"tool_calls_invalid={_fmt(rollout_metrics.get('invalid_tool_call_count_mean'))} "
            f"eval_id={_fmt(eval_id)} eval_pass_rate={_fmt(eval_metrics.get('eval_pass_rate_mean'))} "
            f"eval_acc_strict={_fmt(eval_metrics.get('eval_acc_strict'))}"
        )

        print(line, flush=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        time.sleep(max(1, int(args.interval_sec)))


if __name__ == "__main__":
    main()
