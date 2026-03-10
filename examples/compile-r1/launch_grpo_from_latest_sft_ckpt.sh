#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/workspace/jkh}"
COMPILE_R1_DIR="${COMPILE_R1_DIR:-${BASE_DIR}/slime/examples/compile-r1}"
RUN_SCRIPT="${RUN_SCRIPT:-${COMPILE_R1_DIR}/run_qwen3_4B_grpo_jkh.sh}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${COMPILE_R1_DIR}/train_log/cold_start_sft_qwen3_8b_full_result_masked_val300}"
SFT_CMD_PATTERN="${SFT_CMD_PATTERN:-llamafactory-cli train examples/train_full/qwen3_8b_full_sft_compile_r1_cold_start_result_masked_val300.yaml}"

TRAIN_DATA="${TRAIN_DATA:-${COMPILE_R1_DIR}/data/train/answer_rl_prompt_checked.parquet}"
TEST_DATA="${TEST_DATA:-${COMPILE_R1_DIR}/data/test/answer_rl_prompt_checked.parquet}"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Missing or non-executable run script: ${RUN_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${SFT_OUTPUT_DIR}" ]]; then
  echo "Missing SFT output dir: ${SFT_OUTPUT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "Missing train parquet: ${TRAIN_DATA}" >&2
  exit 1
fi

if [[ ! -f "${TEST_DATA}" ]]; then
  echo "Missing test parquet: ${TEST_DATA}" >&2
  exit 1
fi

echo "[latest-ckpt-grpo] Waiting for SFT process to finish..."
while pgrep -f "${SFT_CMD_PATTERN}" >/dev/null 2>&1; do
  sleep 30
done

echo "[latest-ckpt-grpo] SFT process finished. Resolving latest checkpoint..."
LATEST_CKPT="$(
  python - "${SFT_OUTPUT_DIR}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
cands = []
for p in root.glob("checkpoint-*"):
    if not p.is_dir():
        continue
    m = re.fullmatch(r"checkpoint-(\d+)", p.name)
    if m:
        cands.append((int(m.group(1)), p))

if not cands:
    print("")
else:
    cands.sort(key=lambda x: x[0])
    print(cands[-1][1])
PY
)"

if [[ -z "${LATEST_CKPT}" || ! -d "${LATEST_CKPT}" ]]; then
  echo "Could not find checkpoint-* under ${SFT_OUTPUT_DIR}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
CFG_PATH="${COMPILE_R1_DIR}/configs/grpo_qwen3_8b_latestckpt_lr5e6_n4_len5096_${STAMP}.yaml"
EXP_NAME="compile-r1-grpo-qwen3-8b-latest-lr5e6-n4-l5096-${STAMP}"
RAY_HEAD_PORT=26631
RAY_DASHBOARD_PORT=28631
SLIME_ROLLOUT_START_PORT=15200

cat > "${CFG_PATH}" <<EOF
BASE_DIR: ${BASE_DIR}
SLIME_DIR: ${BASE_DIR}/slime
MEGATRON_DIR: ${BASE_DIR}/Megatron-LM
COMPILE_R1_DIR: ${COMPILE_R1_DIR}
MODEL_DIR: ${LATEST_CKPT}
MODEL_SCRIPT: ${BASE_DIR}/slime/scripts/models/qwen3-8B.sh
TRAIN_DATA: ${TRAIN_DATA}
TEST_DATA: ${TEST_DATA}
LABEL_KEY: answer
EVAL_LABEL_KEY: answer
APPLY_CHAT_TEMPLATE: 1
CONDA_EXE: ${BASE_DIR}/miniconda3/bin/conda
TRAIN_ENV_NAME: slime

NUM_GPUS: 8
SELECTED_GPUS:
MIN_FREE_MEM_MB: 45000
ROLLOUT_NUM_GPUS: 8
ROLLOUT_NUM_GPUS_PER_ENGINE: 1
ACTOR_NUM_NODES: 1
ACTOR_NUM_GPUS_PER_NODE: 8
TENSOR_MODEL_PARALLEL_SIZE: 4

MAX_TOKENS_PER_GPU: 3072
SGLANG_MEM_FRAC: 0.24
TRAIN_MEMORY_MARGIN_BYTES: 4294967296
ENABLE_COLOCATE: 1
ENABLE_DYNAMIC_BATCH_SIZE: 1
DISABLE_PACKED_SEQ: 1
SEQUENCE_PARALLEL: 0

NUM_EPOCH: 1
NUM_ROLLOUT: 600
ROLLOUT_BATCH_SIZE: 16
N_SAMPLES_PER_PROMPT: 4
GLOBAL_BATCH_SIZE: 64
ROLLOUT_MAX_RESPONSE_LEN: 5096
ROLLOUT_TEMPERATURE: 0.8

SAVE_INTERVAL: 100
EVAL_INTERVAL: 100

ADVANTAGE_ESTIMATOR: grpo
ENTROPY_COEF: 0.0
EPS_CLIP: 0.2
EPS_CLIP_HIGH: 0.28
GAMMA: 1.0
LAMBDA: 1.0

OPTIMIZER: adam
LR: 5.0e-06
LR_DECAY_STYLE: constant
WEIGHT_DECAY: 0.01
ADAM_BETA1: 0.9
ADAM_BETA2: 0.98

COMPILE_R1_MAX_TURNS: 6
COMPILE_R1_TOOL_CONCURRENCY: 64
COMPILE_R1_TIMEOUT_SECONDS: 12
COMPILE_R1_MEMORY_MB: 1536
COMPILE_R1_RETURN_LOGPROB: 1
COMPILE_R1_FORMAT_SCORE: 0.0
COMPILE_R1_TOOL_BONUS: 0.0
COMPILE_R1_SANDBOX_BACKEND: runpytool
COMPILE_R1_RUNPY_MAX_WORKERS: 32

RAY_HEAD_PORT: ${RAY_HEAD_PORT}
RAY_DASHBOARD_PORT: ${RAY_DASHBOARD_PORT}
RAY_STATUS_ADDRESS: 127.0.0.1:${RAY_HEAD_PORT}
RAY_JOB_ADDRESS: http://127.0.0.1:${RAY_DASHBOARD_PORT}
RAY_JOB_SUBMIT_RETRIES: 8
RAY_JOB_SUBMIT_RETRY_DELAY: 6
DIRECT_RUN_ON_SUBMIT_FAIL: 1
SLIME_ROLLOUT_START_PORT: ${SLIME_ROLLOUT_START_PORT}

WANDB_MODE: offline
WANDB_PROJECT: compile-r1
EXP_NAME: ${EXP_NAME}
EOF

echo "[latest-ckpt-grpo] Launching GRPO from checkpoint: ${LATEST_CKPT}"
echo "[latest-ckpt-grpo] Config: ${CFG_PATH}"
export CONFIG_FILE="${CFG_PATH}"
export TRAIN_ENV_NAME="slime"
"${RUN_SCRIPT}"
