#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/workspace/jkh}"
COMPILE_R1_DIR="${COMPILE_R1_DIR:-${BASE_DIR}/slime/examples/compile-r1}"
CONDA_EXE="${CONDA_EXE:-${BASE_DIR}/miniconda3/bin/conda}"
SLIME_ENV_PATH="${SLIME_ENV_PATH:-/mnt/workspace/luofuwen/envs/slime}"

SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${COMPILE_R1_DIR}/train_log/cold_start_sft_qwen3_8b_full_natural}"
BASE_RL_CONFIG="${BASE_RL_CONFIG:-${COMPILE_R1_DIR}/configs/grpo_qwen3_8b_fullrl_1epoch_natural_answer.yaml}"
RL_RUN_SCRIPT="${RL_RUN_SCRIPT:-${COMPILE_R1_DIR}/run_qwen3_4B_grpo_jkh.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${COMPILE_R1_DIR}/eval_sft_ckpt_accuracy.py}"
EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${SFT_OUTPUT_DIR}/sft_ckpt_eval_accuracy.json}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Conda executable not found: ${CONDA_EXE}" >&2
  exit 1
fi
if [[ ! -f "${BASE_RL_CONFIG}" ]]; then
  echo "Base RL config not found: ${BASE_RL_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${SFT_OUTPUT_DIR}" ]]; then
  echo "SFT output dir not found: ${SFT_OUTPUT_DIR}" >&2
  exit 1
fi

set +u
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${SLIME_ENV_PATH}"
set -u

echo "[orchestrate-natural] Step 1/3: evaluate SFT checkpoints by test accuracy"
python "${EVAL_SCRIPT}" \
  --ckpt-root "${SFT_OUTPUT_DIR}" \
  --eval-data "${COMPILE_R1_DIR}/data/test_answer_rl.parquet" \
  --output-json "${EVAL_OUTPUT_JSON}" \
  --tp-size 1 \
  --max-samples 0

BEST_CKPT="$(python - "${EVAL_OUTPUT_JSON}" <<'PY'
import json
import sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    obj = json.load(f)
print(obj["best_checkpoint"])
PY
)"
if [[ -z "${BEST_CKPT}" || ! -d "${BEST_CKPT}" ]]; then
  echo "Failed to pick best checkpoint from ${EVAL_OUTPUT_JSON}" >&2
  exit 1
fi
echo "[orchestrate-natural] selected best checkpoint: ${BEST_CKPT}"

echo "[orchestrate-natural] Step 2/3: materialize RL config with best checkpoint"
STAMP="$(date +%Y%m%d-%H%M%S)"
AUTO_RL_CONFIG="${COMPILE_R1_DIR}/configs/grpo_qwen3_8b_fullrl_1epoch_natural_answer.auto_${STAMP}.yaml"
python - "${BASE_RL_CONFIG}" "${AUTO_RL_CONFIG}" "${BEST_CKPT}" <<'PY'
import sys
import yaml
src, dst, best_ckpt = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg["MODEL_DIR"] = best_ckpt
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
print(dst)
PY

echo "[orchestrate-natural] Step 3/3: launch GRPO"
export CONFIG_FILE="${AUTO_RL_CONFIG}"
export EXP_NAME="compile-r1-grpo-qwen3-8b-natural-1epoch-${STAMP}"
bash "${RL_RUN_SCRIPT}"
