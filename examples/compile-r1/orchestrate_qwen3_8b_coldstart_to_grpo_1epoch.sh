#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/workspace/jkh}"
SLIME_DIR="${SLIME_DIR:-${BASE_DIR}/slime}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${BASE_DIR}/LLaMA-Factory}"
COMPILE_R1_DIR="${COMPILE_R1_DIR:-${SLIME_DIR}/examples/compile-r1}"
CONDA_EXE="${ORCH_CONDA_EXE:-${BASE_DIR}/miniconda3/bin/conda}"
LF_ENV_NAME="${LF_ENV_NAME:-lf}"
SLIME_ENV_NAME="${SLIME_ENV_NAME:-slime}"

SFT_CONFIG="${SFT_CONFIG:-${LLAMA_FACTORY_DIR}/examples/train_full/qwen3_8b_full_sft_compile_r1_cold_start.yaml}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${COMPILE_R1_DIR}/train_log/cold_start_sft_qwen3_8b_full}"
RL_BASE_CONFIG="${RL_BASE_CONFIG:-${COMPILE_R1_DIR}/configs/grpo_qwen3_8b_fullrl_1epoch.yaml}"
RL_RUN_SCRIPT="${RL_RUN_SCRIPT:-${COMPILE_R1_DIR}/run_qwen3_4B_grpo_jkh.sh}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Conda executable not found: ${CONDA_EXE}" >&2
  exit 1
fi
if [[ ! -f "${SFT_CONFIG}" ]]; then
  echo "SFT config not found: ${SFT_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${RL_BASE_CONFIG}" ]]; then
  echo "RL base config not found: ${RL_BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${RL_RUN_SCRIPT}" ]]; then
  echo "RL run script not found: ${RL_RUN_SCRIPT}" >&2
  exit 1
fi

set +u
eval "$("${CONDA_EXE}" shell.bash hook)"
set -u

echo "[orchestrate-8b] Step 1/3: Cold-start SFT begin"
conda activate "${LF_ENV_NAME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
cd "${LLAMA_FACTORY_DIR}"
llamafactory-cli train "${SFT_CONFIG}"

echo "[orchestrate-8b] Step 2/3: Select best SFT checkpoint by eval metric"
BEST_CKPT="$(python - "${SFT_OUTPUT_DIR}" <<'PY'
import glob
import json
import os
import sys

out_dir = sys.argv[1]
trainer_state = os.path.join(out_dir, "trainer_state.json")
best = ""
if os.path.isfile(trainer_state):
    with open(trainer_state, "r", encoding="utf-8") as f:
        state = json.load(f)
    best = state.get("best_model_checkpoint") or ""
if not best:
    ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    best = ckpts[-1] if ckpts else ""
print(best)
PY
)"

if [[ -z "${BEST_CKPT}" || ! -d "${BEST_CKPT}" ]]; then
  echo "Failed to locate best SFT checkpoint under: ${SFT_OUTPUT_DIR}" >&2
  exit 1
fi
echo "[orchestrate-8b] Selected best checkpoint: ${BEST_CKPT}"

STAMP="$(date +%Y%m%d-%H%M%S)"
TMP_RL_CONFIG="${COMPILE_R1_DIR}/configs/grpo_qwen3_8b_fullrl_1epoch.auto_${STAMP}.yaml"
python - "${RL_BASE_CONFIG}" "${TMP_RL_CONFIG}" "${BEST_CKPT}" <<'PY'
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

echo "[orchestrate-8b] Step 3/3: Launch full GRPO RL for 1 epoch"
conda activate "${SLIME_ENV_NAME}"
cd "${COMPILE_R1_DIR}"
EXP_NAME="compile-r1-grpo-qwen3-8b-fullrl-1epoch-${STAMP}"
export EXP_NAME
export CONFIG_FILE="${TMP_RL_CONFIG}"
bash "${RL_RUN_SCRIPT}"
