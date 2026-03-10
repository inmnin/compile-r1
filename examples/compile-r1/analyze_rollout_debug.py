#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def _to_float_reward(reward: Any) -> float:
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        for key in ("reward", "total", "score", "value"):
            if key in reward and isinstance(reward[key], (int, float)):
                return float(reward[key])
        for value in reward.values():
            if isinstance(value, (int, float)):
                return float(value)
    return 0.0


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _stats(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def _iter_rollout_files(rollout_dir: Path) -> list[Path]:
    files = [p for p in rollout_dir.glob("*.pt") if p.is_file()]

    def _key(p: Path):
        stem = p.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    return sorted(files, key=_key)


def _choose_threshold(
    index_df: pd.DataFrame,
    max_threshold: int,
    min_threshold: int,
    quantile_positive: float,
    quantile_fallback: float,
    round_to: int,
) -> int:
    lengths = index_df["max_response_length"].to_numpy(dtype=np.int64)
    pos = index_df[index_df["max_reward"] > 0]["max_response_length"].to_numpy(dtype=np.int64)
    base = pos if pos.size > 0 else lengths
    q = quantile_positive if pos.size > 0 else quantile_fallback
    suggested = int(math.ceil(float(np.quantile(base, q)) / round_to) * round_to)
    suggested = max(suggested, min_threshold)
    suggested = min(suggested, max_threshold)
    return suggested


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze compile-r1 debug rollout .pt files.")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-index-parquet", type=Path, required=True)
    parser.add_argument("--max-threshold", type=int, default=7000)
    parser.add_argument("--min-threshold", type=int, default=1024)
    parser.add_argument("--quantile-positive", type=float, default=0.995)
    parser.add_argument("--quantile-fallback", type=float, default=0.99)
    parser.add_argument("--round-to", type=int, default=64)
    args = parser.parse_args()

    files = _iter_rollout_files(args.rollout_dir)
    if not files:
        raise FileNotFoundError(f"No rollout .pt files found in {args.rollout_dir}")

    per_index: dict[int, dict[str, Any]] = {}
    sample_rewards: list[float] = []
    sample_lens: list[int] = []
    sample_truncated: list[int] = []

    for path in files:
        obj = torch.load(path, map_location="cpu")
        samples = obj.get("samples", []) if isinstance(obj, dict) else []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            idx = _safe_int(sample.get("index", -1), default=-1)
            if idx < 0:
                continue

            reward = _to_float_reward(sample.get("reward", 0.0))
            response_len = _safe_int(sample.get("response_length", 0), default=0)
            status = str(sample.get("status", ""))
            truncated = int(status.lower() == "truncated")

            sample_rewards.append(reward)
            sample_lens.append(response_len)
            sample_truncated.append(truncated)

            prev = per_index.get(idx)
            if prev is None:
                per_index[idx] = {
                    "index": idx,
                    "max_reward": reward,
                    "max_response_length": response_len,
                    "sample_count": 1,
                    "any_truncated": truncated,
                }
            else:
                prev["max_reward"] = max(prev["max_reward"], reward)
                prev["max_response_length"] = max(prev["max_response_length"], response_len)
                prev["sample_count"] += 1
                prev["any_truncated"] = max(prev["any_truncated"], truncated)

    if not per_index:
        raise RuntimeError("No valid sample records were parsed from rollout files.")

    index_df = pd.DataFrame(per_index.values()).sort_values("index").reset_index(drop=True)
    index_df.to_parquet(args.output_index_parquet, index=False)

    sample_rewards_arr = np.asarray(sample_rewards, dtype=np.float64)
    sample_lens_arr = np.asarray(sample_lens, dtype=np.int64)
    sample_trunc_arr = np.asarray(sample_truncated, dtype=np.int64)

    index_rewards_arr = index_df["max_reward"].to_numpy(dtype=np.float64)
    index_lens_arr = index_df["max_response_length"].to_numpy(dtype=np.int64)

    suggested_threshold = _choose_threshold(
        index_df=index_df,
        max_threshold=args.max_threshold,
        min_threshold=args.min_threshold,
        quantile_positive=args.quantile_positive,
        quantile_fallback=args.quantile_fallback,
        round_to=args.round_to,
    )

    kept_mask = index_df["max_response_length"] <= suggested_threshold
    pos_mask = index_df["max_reward"] > 0
    pos_kept = (kept_mask & pos_mask).sum()
    pos_total = max(1, int(pos_mask.sum()))

    summary = {
        "rollout_dir": str(args.rollout_dir),
        "num_rollout_files": len(files),
        "num_sample_records": int(sample_rewards_arr.size),
        "num_unique_indices": int(index_df.shape[0]),
        "sample_reward_stats": _stats(sample_rewards_arr),
        "sample_response_len_stats": _stats(sample_lens_arr),
        "sample_truncated_ratio": float(sample_trunc_arr.mean()) if sample_trunc_arr.size else 0.0,
        "index_reward_stats": _stats(index_rewards_arr),
        "index_response_len_stats": _stats(index_lens_arr),
        "index_positive_reward_ratio": float((index_rewards_arr > 0).mean()),
        "suggested_threshold": int(suggested_threshold),
        "suggested_threshold_retention": {
            "kept_ratio_all": float(kept_mask.mean()),
            "kept_ratio_positive_reward": float(pos_kept / pos_total),
            "dropped_count": int((~kept_mask).sum()),
        },
        "output_index_parquet": str(args.output_index_parquet),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

