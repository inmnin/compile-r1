#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from statistics import mean

import torch

from qa_em_format import em_check, extract_solution


def parse_args():
    parser = argparse.ArgumentParser(description="Export Search-R1 test trajectories from debug eval rollout data.")
    parser.add_argument("--eval-rollout-pt", type=str, required=True, help="Path to eval_xxx.pt debug rollout file.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Output jsonl path.")
    parser.add_argument("--output-metrics-json", type=str, required=True, help="Output metrics json path.")
    return parser.parse_args()


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        value = x.tolist()
        return value if isinstance(value, list) else [value]
    return [x]


def _extract_question(prompt):
    text = ""
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg["content"]))
        text = "\n".join(parts)
    else:
        text = str(prompt)

    m = re.search(r"Question:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def _safe_mean(values):
    valid = [v for v in values if v is not None]
    return mean(valid) if valid else None


def main():
    args = parse_args()

    eval_path = Path(args.eval_rollout_pt)
    out_jsonl = Path(args.output_jsonl)
    out_metrics = Path(args.output_metrics_json)

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval rollout file not found: {eval_path}")

    payload = torch.load(eval_path, map_location="cpu", weights_only=False)
    samples = payload.get("samples", [])

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    rewards = []
    accs = []
    losses = []
    entropies = []
    response_lengths = []
    search_calls = []

    with out_jsonl.open("w", encoding="utf-8") as f:
        for sample in samples:
            prompt = sample.get("prompt", "")
            response = sample.get("response", "")
            question = _extract_question(prompt)

            reward = sample.get("reward")
            if isinstance(reward, dict):
                reward_value = reward.get("score", None)
            else:
                reward_value = reward

            ground_truth = sample.get("label", {}).get("ground_truth", {}).get("target", [])
            ground_truth = _to_list(ground_truth)

            solution = f"{prompt}{response}"
            pred_answer = extract_solution(solution)
            acc = 0.0
            if pred_answer is not None and ground_truth:
                acc = float(em_check(pred_answer, ground_truth))

            rollout_log_probs = sample.get("rollout_log_probs")
            loss_mask = sample.get("loss_mask")
            loss = None
            entropy = None
            if isinstance(rollout_log_probs, list) and isinstance(loss_mask, list) and len(rollout_log_probs) == len(loss_mask):
                token_nll = [-float(lp) for lp, m in zip(rollout_log_probs, loss_mask, strict=True) if int(m) == 1]
                if token_nll:
                    loss = sum(token_nll) / len(token_nll)
                    # Entropy is not directly emitted by rollout. Use token-level surprisal proxy for analysis.
                    entropy = loss

            record = {
                "question": question,
                "traj": response,
                "reward": reward_value,
                "acc": acc,
                "loss": loss,
                "entropy": entropy,
                "response_length": sample.get("response_length"),
                "effective_response_length": int(sum(loss_mask)) if isinstance(loss_mask, list) else sample.get("response_length"),
                "search_calls": response.count("<search>"),
                "status": sample.get("status"),
                "ground_truth": ground_truth,
                "pred_answer": pred_answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            rewards.append(reward_value)
            accs.append(acc)
            losses.append(loss)
            entropies.append(entropy)
            response_lengths.append(sample.get("response_length"))
            search_calls.append(record["search_calls"])

    metrics = {
        "num_samples": len(samples),
        "reward_mean": _safe_mean(rewards),
        "acc_mean": _safe_mean(accs),
        "loss_mean": _safe_mean(losses),
        "entropy_mean": _safe_mean(entropies),
        "response_length_mean": _safe_mean(response_lengths),
        "search_calls_mean": _safe_mean(search_calls),
        "eval_rollout_id": payload.get("rollout_id"),
        "eval_rollout_path": str(eval_path),
        "note": "entropy_mean uses rollout token surprisal proxy because rollout API only returns sampled-token logprobs.",
    }

    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False))
    print(f"Saved trajectories: {out_jsonl}")
    print(f"Saved metrics: {out_metrics}")


if __name__ == "__main__":
    main()
