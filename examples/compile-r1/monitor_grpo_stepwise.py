#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor GRPO metrics and print every N steps.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--step-interval", type=int, default=5)
    p.add_argument("--duration-sec", type=int, default=10800)
    p.add_argument("--poll-sec", type=int, default=20)
    p.add_argument("--output-log", type=Path, default=None)
    return p.parse_args()


def _sample_attr(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _sample_metadata(sample: Any) -> dict[str, Any]:
    md = _sample_attr(sample, "metadata", {})
    return md if isinstance(md, dict) else {}


def _fmt(v: float | int | None) -> str:
    if v is None:
        return "na"
    if isinstance(v, int):
        return str(v)
    return f"{v:.6f}"


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _load_rollout_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
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

    reward_total = []
    answer_pass = []
    reward_base = []
    reward_tool = []
    reward_tool_call = []
    reward_tool_perfect = []
    reward_format = []
    reward_invalid = []
    reward_overcall = []
    tool_calls = []
    valid_tool_calls = []
    invalid_tool_calls = []

    for s in samples:
        r = _sample_attr(s, "reward", None)
        if isinstance(r, (int, float)):
            reward_total.append(float(r))
        md = _sample_metadata(s)
        for key, store in (
            ("answer_pass_rate", answer_pass),
            ("reward/base_pass", reward_base),
            ("reward/tool_success_bonus", reward_tool),
            ("reward/tool_call_bonus", reward_tool_call),
            ("reward/tool_perfect_bonus", reward_tool_perfect),
            ("reward/format_bonus", reward_format),
            ("reward/invalid_tool_penalty", reward_invalid),
            ("reward/overcall_penalty", reward_overcall),
            ("tool_call_count", tool_calls),
            ("valid_tool_call_count", valid_tool_calls),
            ("invalid_tool_call_count", invalid_tool_calls),
        ):
            v = md.get(key)
            if isinstance(v, (int, float)):
                store.append(float(v))

    return {
        "reward_total": _mean(reward_total),
        "answer_pass_rate": _mean(answer_pass),
        "reward_base": _mean(reward_base),
        "reward_tool_bonus": _mean(reward_tool),
        "reward_tool_call_bonus": _mean(reward_tool_call),
        "reward_tool_perfect_bonus": _mean(reward_tool_perfect),
        "reward_format_bonus": _mean(reward_format),
        "reward_invalid_penalty": _mean(reward_invalid),
        "reward_overcall_penalty": _mean(reward_overcall),
        "tool_calls": _mean(tool_calls),
        "tool_calls_valid": _mean(valid_tool_calls),
        "tool_calls_invalid": _mean(invalid_tool_calls),
    }


def _load_eval_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    obj = torch.load(path, map_location="cpu", weights_only=False)
    samples = obj.get("samples", []) if isinstance(obj, dict) else []
    rewards = []
    for s in samples:
        r = _sample_attr(s, "reward", None)
        if isinstance(r, (int, float)):
            rewards.append(float(r))
    if not rewards:
        return {}
    pass_rate = sum(rewards) / len(rewards)
    acc = sum(1.0 for x in rewards if x >= (1.0 - 1e-12)) / len(rewards)
    return {"eval_pass_rate": pass_rate, "eval_acc_strict": acc}


def _available_rollout_ids(rollout_dir: Path) -> list[int]:
    files = [Path(p) for p in glob.glob(str(rollout_dir / "*.pt"))]
    ids: list[int] = []
    for p in files:
        stem = p.stem
        if stem.isdigit():
            ids.append(int(stem))
    ids.sort()
    return ids


def _pick_rollout_file(rollout_dir: Path, target_step: int) -> tuple[int | None, Path | None]:
    ids = _available_rollout_ids(rollout_dir)
    if not ids:
        return None, None
    chosen = None
    for rid in ids:
        if rid <= target_step:
            chosen = rid
        else:
            break
    if chosen is None:
        chosen = ids[0]
    return chosen, rollout_dir / f"{chosen}.pt"


def _pick_eval_file(rollout_dir: Path, target_step: int) -> tuple[int | None, Path | None]:
    files = [Path(p) for p in glob.glob(str(rollout_dir / "eval_*.pt"))]
    ids: list[int] = []
    for p in files:
        try:
            ids.append(int(p.stem.split("_", 1)[1]))
        except Exception:
            pass
    ids.sort()
    if not ids:
        return None, None
    chosen = None
    for eid in ids:
        if eid <= target_step:
            chosen = eid
        else:
            break
    if chosen is None:
        chosen = ids[0]
    return chosen, rollout_dir / f"eval_{chosen}.pt"


def _parse_train_log(log_path: Path) -> tuple[int | None, dict[int, tuple[float, float]], dict[int, float]]:
    if not log_path.exists():
        return None, {}, {}
    ansi_pat = re.compile(r"\x1b\[[0-9;]*m")
    step_pat = re.compile(
        r"step\s+(\d+):\s+\{.*?'train/loss':\s*([-+]?[\d.]+(?:e[-+]?\d+)?)"
        r".*?'train/entropy_loss':\s*([-+]?[\d.]+(?:e[-+]?\d+)?)"
    )
    trunc_pat = re.compile(
        r"perf\s+(\d+):\s+\{.*?'rollout/truncated_ratio':\s*([-+]?[\d.]+(?:e[-+]?\d+)?)"
    )

    latest_step: int | None = None
    step_map: dict[int, tuple[float, float]] = {}
    trunc_map: dict[int, float] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = ansi_pat.sub("", line)
            m = step_pat.search(line)
            if m:
                step = int(m.group(1))
                step_map[step] = (float(m.group(2)), float(m.group(3)))
                latest_step = step if latest_step is None else max(latest_step, step)
            m2 = trunc_pat.search(line)
            if m2:
                trunc_map[int(m2.group(1))] = float(m2.group(2))
    return latest_step, step_map, trunc_map


def _lookup_last_leq(mapping: dict[int, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    candidates = [k for k in mapping if k <= key]
    if not candidates:
        return None
    return mapping[max(candidates)]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    log_path = run_dir / "logs" / "train_console.log"
    rollout_dir = run_dir / "debug" / "rollout_data"
    out_path = args.output_log or (run_dir / "logs" / "metrics_monitor_stepwise.log")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    end_ts = time.time() + max(1, int(args.duration_sec))
    step_interval = max(1, int(args.step_interval))
    poll_sec = max(1, int(args.poll_sec))

    latest_step, _, _ = _parse_train_log(log_path)
    if latest_step is None:
        next_step = step_interval
    else:
        next_step = ((latest_step // step_interval) + 1) * step_interval
    head = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] monitor_started latest_step={latest_step if latest_step is not None else 'na'} next_emit_step={next_step}"
    print(head, flush=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(head + "\n")

    while time.time() <= end_ts:
        latest_step, step_map, trunc_map = _parse_train_log(log_path)
        if latest_step is None or latest_step < next_step:
            time.sleep(poll_sec)
            continue

        while latest_step is not None and latest_step >= next_step:
            step = next_step
            step_pair = _lookup_last_leq(step_map, step)
            loss = step_pair[0] if step_pair else None
            entropy = step_pair[1] if step_pair else None
            truncated = _lookup_last_leq(trunc_map, step)

            rollout_id, rollout_file = _pick_rollout_file(rollout_dir, step)
            eval_id, eval_file = _pick_eval_file(rollout_dir, step)
            rollout_metrics = _load_rollout_metrics(rollout_file)
            eval_metrics = _load_eval_metrics(eval_file)

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = (
                f"[{ts}] step={step} "
                f"rollout={_fmt(rollout_id)} "
                f"loss={_fmt(loss)} entropy={_fmt(entropy)} truncated_ratio={_fmt(truncated)} "
                f"reward_total={_fmt(rollout_metrics.get('reward_total'))} "
                f"reward_base={_fmt(rollout_metrics.get('reward_base'))} "
                f"reward_tool_bonus={_fmt(rollout_metrics.get('reward_tool_bonus'))} "
                f"reward_tool_call_bonus={_fmt(rollout_metrics.get('reward_tool_call_bonus'))} "
                f"reward_tool_perfect_bonus={_fmt(rollout_metrics.get('reward_tool_perfect_bonus'))} "
                f"reward_format_bonus={_fmt(rollout_metrics.get('reward_format_bonus'))} "
                f"reward_invalid_penalty={_fmt(rollout_metrics.get('reward_invalid_penalty'))} "
                f"reward_overcall_penalty={_fmt(rollout_metrics.get('reward_overcall_penalty'))} "
                f"answer_pass_rate={_fmt(rollout_metrics.get('answer_pass_rate'))} "
                f"tool_calls={_fmt(rollout_metrics.get('tool_calls'))} "
                f"tool_calls_valid={_fmt(rollout_metrics.get('tool_calls_valid'))} "
                f"tool_calls_invalid={_fmt(rollout_metrics.get('tool_calls_invalid'))} "
                f"eval_id={_fmt(eval_id)} "
                f"eval_avg_pass_rate={_fmt(eval_metrics.get('eval_pass_rate'))} "
                f"eval_avg_acc={_fmt(eval_metrics.get('eval_acc_strict'))}"
            )
            print(line, flush=True)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            next_step += step_interval

        time.sleep(poll_sec)


if __name__ == "__main__":
    main()
