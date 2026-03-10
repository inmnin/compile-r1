#!/usr/bin/env python3

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DISTILL_SYSTEM_PROMPT = """Answer the given Python coding question.
Use tags in a natural way:
- <think>...</think>: your reasoning for the current step.
- <code>...</code>: Python code to execute when you need verification/debugging.
- <result>...</result>: execution feedback returned by the environment.
- <answer>...</answer>: the final runnable Python solution.

You may call <code> multiple rounds before finishing.
When tool feedback is needed, prefer running code instead of guessing.
Always put the final solution code in <answer>...</answer>.
"""


def build_user_prompt_natural(task_prompt: str, entry_point: str) -> list[dict[str, str]]:
    instruction = (
        "Answer the given Python coding question.\n"
        "Use tags in a natural way:\n"
        "- <think>...</think>: your reasoning for the current step.\n"
        "- <code>...</code>: Python code to execute when you need verification/debugging.\n"
        "- <result>...</result>: execution feedback returned by the environment.\n"
        "- <answer>...</answer>: the final runnable Python solution.\n\n"
        "You may call <code> multiple rounds before finishing.\n"
        "When tool feedback is needed, prefer running code instead of guessing.\n"
        "Always put the final solution code in <answer>...</answer>.\n\n"
        "Problem:\n"
        "```python\n"
        f"{task_prompt.rstrip()}\n"
        "```"
    )
    return [{"role": "user", "content": instruction}]


def build_user_prompt_distill_isomorphic(task_prompt: str) -> list[dict[str, str]]:
    # Keep prompt structure aligned with distillation records: single user turn with raw question.
    return [{"role": "user", "content": task_prompt.strip()}]


def build_user_prompt_distill_script(task_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Solve the Python programming task below. "
                "If uncertain, use <code> first and wait for <result>.\n\n"
                f"Problem:\n{task_prompt.strip()}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Reminder: use <think> for reasoning, <code> for tool calls, and put the final solution in <answer>."
            ),
        },
    ]


def build_user_prompt(task_prompt: str, entry_point: str, prompt_mode: str) -> list[dict[str, str]]:
    if prompt_mode == "distill_script_prompt":
        return build_user_prompt_distill_script(task_prompt)
    if prompt_mode == "distill_isomorphic":
        return build_user_prompt_distill_isomorphic(task_prompt)
    return build_user_prompt_natural(task_prompt, entry_point)


def normalize_test_cases(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if x is not None and str(x).strip()]
    if hasattr(raw, "tolist"):
        try:
            as_list = raw.tolist()
            if isinstance(as_list, list):
                return [str(x).strip() for x in as_list if x is not None and str(x).strip()]
        except Exception:
            pass
    text = str(raw).strip()
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def infer_entry_point(question: str, test_cases: list[str], default: str = "solution") -> str:
    for tc in test_cases:
        match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", tc)
        if match:
            return match.group(1)
    match = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", question)
    if match:
        return match.group(1)
    return default


def extract_assert_cases(test_code: str) -> list[str]:
    cases = []
    for line in str(test_code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            cases.append(stripped)
    return cases


def best_inference_completion(inferences) -> str:
    candidates = []
    if isinstance(inferences, list):
        candidates = inferences
    elif hasattr(inferences, "tolist"):
        try:
            as_list = inferences.tolist()
            if isinstance(as_list, list):
                candidates = as_list
        except Exception:
            pass

    best_completion = ""
    best_score = -1.0
    for item in candidates:
        if not isinstance(item, dict):
            continue
        completion = str(item.get("completion") or "").strip()
        if not completion:
            continue
        score = float(item.get("pass_rate") or 0.0)
        if score >= best_score:
            best_score = score
            best_completion = completion
    return best_completion


def convert_train(train_glob: str, output_path: Path, prompt_mode: str) -> None:
    rows = []
    dropped = 0
    for file in sorted(glob.glob(train_glob)):
        table = pq.read_table(file)
        for row in table.to_pylist():
            question = str(row.get("question") or "").strip()
            tests = normalize_test_cases(row.get("test_cases"))
            answer = best_inference_completion(row.get("inferences"))
            if not question or not tests or not answer:
                dropped += 1
                continue

            task_id = str(row.get("id") or "")
            entry_point = infer_entry_point(question, tests)
            ground_truth = {
                "task_id": task_id,
                "prompt": question,
                "test": "\n".join(tests),
                "test_cases": tests,
                "entry_point": entry_point,
                "canonical_solution": answer,
            }
            rows.append(
                {
                    "id": task_id,
                    "task_id": task_id,
                    "question": question,
                    "prompt": build_user_prompt(question, entry_point, prompt_mode),
                    "ability": "code-generation",
                    "answer": {"target": answer, "ground_truth": ground_truth},
                    "extra_info": {"source": str(row.get("source") or ""), "from_file": os.path.basename(file)},
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"train rows={len(rows)} dropped={dropped} -> {output_path}")


def convert_test(test_glob: str, output_path: Path, prompt_mode: str) -> None:
    rows = []
    dropped = 0
    for file in sorted(glob.glob(test_glob, recursive=True)):
        table = pq.read_table(file)
        for row in table.to_pylist():
            question = str(row.get("prompt") or "").strip()
            test_code = str(row.get("test") or "").strip()
            entry_point = str(row.get("entry_point") or "").strip()
            answer = str(row.get("canonical_solution") or "").strip()
            if not question or not test_code or not entry_point or not answer:
                dropped += 1
                continue

            task_id = str(row.get("task_id") or row.get("id") or "")
            ground_truth = {
                "task_id": task_id,
                "prompt": question,
                "test": test_code,
                "test_cases": extract_assert_cases(test_code),
                "entry_point": entry_point,
                "canonical_solution": answer,
            }
            rows.append(
                {
                    "id": task_id,
                    "task_id": task_id,
                    "question": question,
                    "prompt": build_user_prompt(question, entry_point, prompt_mode),
                    "ability": "code-generation",
                    "answer": {"target": answer, "ground_truth": ground_truth},
                    "extra_info": {"from_file": os.path.basename(file)},
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"test rows={len(rows)} dropped={dropped} -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare compile-r1 data with answer labels for GRPO.")
    parser.add_argument(
        "--train-glob",
        type=str,
        default="/mnt/workspace/jkh/slime/examples/compile-r1/data/train/data/*.parquet",
    )
    parser.add_argument(
        "--test-glob",
        type=str,
        default="/mnt/workspace/jkh/slime/examples/compile-r1/data/test/**/*.parquet",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=Path("/mnt/workspace/jkh/slime/examples/compile-r1/data/train_answer_rl.parquet"),
    )
    parser.add_argument(
        "--output-test",
        type=Path,
        default=Path("/mnt/workspace/jkh/slime/examples/compile-r1/data/test_answer_rl.parquet"),
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="natural_instruction",
        choices=["natural_instruction", "distill_isomorphic", "distill_script_prompt"],
        help="Prompt rendering mode for RL parquet outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert_train(args.train_glob, args.output_train, args.prompt_mode)
    convert_test(args.test_glob, args.output_test, args.prompt_mode)


if __name__ == "__main__":
    main()
