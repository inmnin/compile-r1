#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/workspace/jkh}"
SLIME_DIR="${SLIME_DIR:-${BASE_DIR}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${BASE_DIR}/Megatron-LM}"
COMPILE_R1_DIR="${COMPILE_R1_DIR:-${SLIME_DIR}/examples/compile-r1}"

DEFAULT_CONFIG_FILE="${COMPILE_R1_DIR}/configs/grpo_qwen3_4b_coldstart.yaml"
CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
if [[ -n "${CONFIG_FILE}" && -f "${CONFIG_FILE}" ]]; then
  case "${CONFIG_FILE}" in
    *.yaml|*.yml)
      set +u
      eval "$(
        python - "${CONFIG_FILE}" <<'PY'
import re
import shlex
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
if not isinstance(data, dict):
    raise SystemExit(f"Top-level YAML object must be a mapping: {path}")
for key, value in data.items():
    if not isinstance(key, str):
        continue
    name = re.sub(r"[^A-Z0-9_]", "_", key.strip().upper())
    if not name:
        continue
    if value is None:
        rendered = ""
    elif isinstance(value, bool):
        rendered = "1" if value else "0"
    else:
        rendered = str(value)
    print(f"{name}={shlex.quote(rendered)}")
PY
      )"
      set -u
      echo "Loaded YAML config file: ${CONFIG_FILE}"
      ;;
    *.env|*.sh)
      # shellcheck disable=SC1090
      set +u
      source "${CONFIG_FILE}"
      set -u
      echo "Loaded env config file: ${CONFIG_FILE}"
      ;;
    *)
      echo "Unsupported CONFIG_FILE extension: ${CONFIG_FILE}" >&2
      exit 1
      ;;
  esac
fi

MODEL_DIR="${MODEL_DIR:-${BASE_DIR}/model/Qwen3-4B}"
MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3-4B.sh}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}}"
REF_LOAD="${REF_LOAD:-${MODEL_DIR}}"
ACTOR_LOAD="${ACTOR_LOAD:-}"
MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-bridge}"
TRAIN_DATA="${TRAIN_DATA:-${COMPILE_R1_DIR}/data/train.parquet}"
TEST_DATA="${TEST_DATA:-${COMPILE_R1_DIR}/data/test.parquet}"
LABEL_KEY="${LABEL_KEY:-reward_model}"
EVAL_LABEL_KEY="${EVAL_LABEL_KEY:-${LABEL_KEY}}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
SELECTED_GPUS="${SELECTED_GPUS:-}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-50000}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"

NUM_ROLLOUT="${NUM_ROLLOUT:-600}"
NUM_EPOCH="${NUM_EPOCH:-}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"

SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-3072}"
SGLANG_MEM_FRAC="${SGLANG_MEM_FRAC:-0.2}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-2147483648}"
ENABLE_COLOCATE="${ENABLE_COLOCATE:-0}"
ENABLE_DYNAMIC_BATCH_SIZE="${ENABLE_DYNAMIC_BATCH_SIZE:-0}"
DISABLE_PACKED_SEQ="${DISABLE_PACKED_SEQ:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
GAMMA="${GAMMA:-1.0}"
LAMBDA="${LAMBDA:-1.0}"

OPTIMIZER="${OPTIMIZER:-adam}"
LR="${LR:-1e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"

COMPILE_R1_MAX_TURNS="${COMPILE_R1_MAX_TURNS:-3}"
COMPILE_R1_TOOL_CONCURRENCY="${COMPILE_R1_TOOL_CONCURRENCY:-32}"
COMPILE_R1_ASYNC_TOOL_WORKERS="${COMPILE_R1_ASYNC_TOOL_WORKERS:-${COMPILE_R1_TOOL_CONCURRENCY}}"
COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER="${COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER:-3}"
COMPILE_R1_TIMEOUT_SECONDS="${COMPILE_R1_TIMEOUT_SECONDS:-8}"
COMPILE_R1_MEMORY_MB="${COMPILE_R1_MEMORY_MB:-1024}"
COMPILE_R1_RETURN_LOGPROB="${COMPILE_R1_RETURN_LOGPROB:-1}"
COMPILE_R1_FORMAT_SCORE="${COMPILE_R1_FORMAT_SCORE:-0.02}"
COMPILE_R1_TOOL_BONUS="${COMPILE_R1_TOOL_BONUS:-0.12}"
COMPILE_R1_TOOL_BONUS_CAP="${COMPILE_R1_TOOL_BONUS_CAP:-3}"
COMPILE_R1_TOOL_SUCCESS_BONUS="${COMPILE_R1_TOOL_SUCCESS_BONUS:-0.20}"
COMPILE_R1_TOOL_SUCCESS_CAP="${COMPILE_R1_TOOL_SUCCESS_CAP:-3}"
COMPILE_R1_TOOL_INVALID_PENALTY="${COMPILE_R1_TOOL_INVALID_PENALTY:-0.0}"
COMPILE_R1_TOOL_OVERCALL_PENALTY="${COMPILE_R1_TOOL_OVERCALL_PENALTY:-0.0}"
COMPILE_R1_TOOL_OVERCALL_FREE_CALLS="${COMPILE_R1_TOOL_OVERCALL_FREE_CALLS:-3}"
COMPILE_R1_REWARD_PASS_ONLY="${COMPILE_R1_REWARD_PASS_ONLY:-1}"
COMPILE_R1_REWARD_CLIP_MIN="${COMPILE_R1_REWARD_CLIP_MIN:--1.0}"
COMPILE_R1_REWARD_CLIP_MAX="${COMPILE_R1_REWARD_CLIP_MAX:-2.0}"
COMPILE_R1_SANDBOX_BACKEND="${COMPILE_R1_SANDBOX_BACKEND:-runpytool}"
COMPILE_R1_RUNPY_MAX_WORKERS="${COMPILE_R1_RUNPY_MAX_WORKERS:-8}"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"
SLIME_ROLLOUT_START_PORT="${SLIME_ROLLOUT_START_PORT:-15000}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_TRAIN_ARGS_ARR=()
if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_TRAIN_ARGS_ARR=(${EXTRA_TRAIN_ARGS})
fi

CONDA_EXE="${CONDA_EXE:-${BASE_DIR}/miniconda3/bin/conda}"
TRAIN_ENV_NAME="${TRAIN_ENV_NAME:-slime}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-26379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28265}"
RAY_NODE_IP="${RAY_NODE_IP:-$(hostname -I | awk '{print $1}')}"
RAY_STATUS_ADDRESS="${RAY_STATUS_ADDRESS:-127.0.0.1:${RAY_HEAD_PORT}}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}"
RAY_JOB_SUBMIT_RETRIES="${RAY_JOB_SUBMIT_RETRIES:-8}"
RAY_JOB_SUBMIT_RETRY_DELAY="${RAY_JOB_SUBMIT_RETRY_DELAY:-6}"
DIRECT_RUN_ON_SUBMIT_FAIL="${DIRECT_RUN_ON_SUBMIT_FAIL:-1}"
FORCE_DIRECT_RUN="${FORCE_DIRECT_RUN:-0}"

WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-compile-r1}"

RUN_TAG_TS="$(date +%Y%m%d-%H%M%S)"
EXP_NAME="${EXP_NAME:-compile-r1-grpo-qwen3-4b-${RUN_TAG_TS}}"
RUN_ROOT="${COMPILE_R1_DIR}/train_log/${EXP_NAME}"
CKPT_DIR="${RUN_ROOT}/ckpt"
WANDB_DIR="${RUN_ROOT}/wandb"
DEBUG_DIR="${RUN_ROOT}/debug"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${CKPT_DIR}" "${WANDB_DIR}" "${DEBUG_DIR}" "${LOG_DIR}"

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

pick_gpus() {
  local need="$1"
  local min_free="$2"
  local selected=()
  local fallback=()

  while IFS=',' read -r idx free; do
    idx="$(echo "${idx}" | xargs)"
    free="$(echo "${free}" | xargs)"
    fallback+=("${idx}")
    if (( free >= min_free )); then
      selected+=("${idx}")
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -nr)

  if (( ${#selected[@]} < need )); then
    selected=("${fallback[@]:0:${need}}")
  else
    selected=("${selected[@]:0:${need}}")
  fi

  (IFS=','; echo "${selected[*]}")
}

if [[ -z "${SELECTED_GPUS}" ]]; then
  SELECTED_GPUS="$(pick_gpus "${NUM_GPUS}" "${MIN_FREE_MEM_MB}")"
fi
export CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}"
VISIBLE_GPU_COUNT="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
if (( VISIBLE_GPU_COUNT < 1 )); then
  echo "No GPUs selected. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if (( VISIBLE_GPU_COUNT < NUM_GPUS )); then
  NUM_GPUS="${VISIBLE_GPU_COUNT}"
fi
if (( ROLLOUT_NUM_GPUS > NUM_GPUS )); then
  ROLLOUT_NUM_GPUS="${NUM_GPUS}"
fi
if (( TENSOR_MODEL_PARALLEL_SIZE > NUM_GPUS )); then
  TENSOR_MODEL_PARALLEL_SIZE="${NUM_GPUS}"
fi
if (( ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
  echo "ROLLOUT_NUM_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE." >&2
  exit 1
fi

set +u
if is_true "${SKIP_CONDA_ACTIVATE:-0}"; then
  echo "Skip conda activation (SKIP_CONDA_ACTIVATE=1), using current shell environment."
else
  if [[ -x "${CONDA_EXE}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
    conda activate "${TRAIN_ENV_NAME}"
  else
    echo "Conda not found: ${CONDA_EXE}" >&2
    exit 1
  fi
fi
set -u

if [[ ! -e "${TRAIN_DATA}" ]]; then
  echo "Missing train data path: ${TRAIN_DATA}" >&2
  exit 1
fi
if [[ ! -e "${TEST_DATA}" ]]; then
  echo "Missing test data path: ${TEST_DATA}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_SCRIPT}" ]]; then
  echo "Missing model script: ${MODEL_SCRIPT}" >&2
  exit 1
fi

if [[ -n "${NUM_EPOCH}" ]]; then
  if ! [[ "${NUM_EPOCH}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "NUM_EPOCH must be a positive number, got: ${NUM_EPOCH}" >&2
    exit 1
  fi
  if (( ROLLOUT_BATCH_SIZE <= 0 )); then
    echo "ROLLOUT_BATCH_SIZE must be > 0 when NUM_EPOCH is set." >&2
    exit 1
  fi
  read -r TOTAL_PROMPTS AUTO_NUM_ROLLOUT < <(
    python - "${TRAIN_DATA}" "${NUM_EPOCH}" "${ROLLOUT_BATCH_SIZE}" <<'PY'
import glob
import math
import os
import sys

import pyarrow.parquet as pq

train_data = sys.argv[1]
num_epoch = float(sys.argv[2])
rollout_batch_size = int(sys.argv[3])

files = []
if os.path.isdir(train_data):
    files = glob.glob(os.path.join(train_data, "**", "*.parquet"), recursive=True)
elif os.path.isfile(train_data):
    files = [train_data]

rows = 0
for f in files:
    rows += pq.ParquetFile(f).metadata.num_rows

if rows <= 0:
    raise SystemExit("Could not infer dataset size from TRAIN_DATA parquet files.")

num_rollout = max(1, math.ceil(rows * num_epoch / rollout_batch_size))
print(rows, num_rollout)
PY
  )
  NUM_ROLLOUT="${AUTO_NUM_ROLLOUT}"
  echo "[compile-r1-grpo] NUM_EPOCH=${NUM_EPOCH}, TOTAL_PROMPTS=${TOTAL_PROMPTS}, AUTO_NUM_ROLLOUT=${NUM_ROLLOUT}"
fi

cd "${SLIME_DIR}"

# Ensure Ray workers can import Megatron/compile-r1 modules in both submit and direct modes.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${MEGATRON_DIR}:${COMPILE_R1_DIR}:${SLIME_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${MEGATRON_DIR}:${COMPILE_R1_DIR}:${SLIME_DIR}"
fi

if ! timeout 12 ray status --address="${RAY_STATUS_ADDRESS}" >/dev/null 2>&1; then
  echo "Ray is not running at ${RAY_STATUS_ADDRESS}, starting an isolated local head..."
  ray start --head \
    --node-ip-address "${RAY_NODE_IP}" \
    --port "${RAY_HEAD_PORT}" \
    --dashboard-host 127.0.0.1 \
    --dashboard-port "${RAY_DASHBOARD_PORT}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats
else
  echo "Reusing existing Ray cluster at ${RAY_STATUS_ADDRESS}"
fi

source "${MODEL_SCRIPT}"

if (( ENABLE_COLOCATE )); then
  COLOCATE_ARGS=(--colocate)
else
  COLOCATE_ARGS=()
fi

if (( ENABLE_DYNAMIC_BATCH_SIZE )); then
  DYN_ARGS=(--use-dynamic-batch-size)
else
  DYN_ARGS=()
fi

if is_true "${APPLY_CHAT_TEMPLATE}"; then
  CHAT_TEMPLATE_ARGS=(--apply-chat-template)
else
  CHAT_TEMPLATE_ARGS=()
fi

if is_true "${SEQUENCE_PARALLEL}"; then
  SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
else
  SEQUENCE_PARALLEL_ARGS=()
fi

if [[ -n "${ACTOR_LOAD}" ]]; then
  LOAD_ARGS=(--load "${ACTOR_LOAD}")
else
  LOAD_ARGS=()
fi

export SLIME_DISABLE_PACKED_SEQ="${DISABLE_PACKED_SEQ}"
TRAIN_ENV_VARS_JSON="${TRAIN_ENV_VARS_JSON:-{\"SLIME_DISABLE_PACKED_SEQ\":\"${SLIME_DISABLE_PACKED_SEQ}\"}}"
TRAIN_ENV_VARS_ARGS=(--train-env-vars "${TRAIN_ENV_VARS_JSON}")

RUNTIME_ENV_JSON="$(cat <<EOF
{"env_vars":{"PYTHONPATH":"${MEGATRON_DIR}:${COMPILE_R1_DIR}:${SLIME_DIR}","CUDA_DEVICE_MAX_CONNECTIONS":"1","MASTER_ADDR":"127.0.0.1","RAY_ADDRESS":"${RAY_STATUS_ADDRESS}","COMPILE_R1_MAX_TURNS":"${COMPILE_R1_MAX_TURNS}","COMPILE_R1_TOOL_CONCURRENCY":"${COMPILE_R1_TOOL_CONCURRENCY}","COMPILE_R1_ASYNC_TOOL_WORKERS":"${COMPILE_R1_ASYNC_TOOL_WORKERS}","COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER":"${COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER}","COMPILE_R1_TIMEOUT_SECONDS":"${COMPILE_R1_TIMEOUT_SECONDS}","COMPILE_R1_MEMORY_MB":"${COMPILE_R1_MEMORY_MB}","COMPILE_R1_RETURN_LOGPROB":"${COMPILE_R1_RETURN_LOGPROB}","COMPILE_R1_FORMAT_SCORE":"${COMPILE_R1_FORMAT_SCORE}","COMPILE_R1_TOOL_BONUS":"${COMPILE_R1_TOOL_BONUS}","COMPILE_R1_TOOL_BONUS_CAP":"${COMPILE_R1_TOOL_BONUS_CAP}","COMPILE_R1_TOOL_SUCCESS_BONUS":"${COMPILE_R1_TOOL_SUCCESS_BONUS}","COMPILE_R1_TOOL_SUCCESS_CAP":"${COMPILE_R1_TOOL_SUCCESS_CAP}","COMPILE_R1_TOOL_INVALID_PENALTY":"${COMPILE_R1_TOOL_INVALID_PENALTY}","COMPILE_R1_TOOL_OVERCALL_PENALTY":"${COMPILE_R1_TOOL_OVERCALL_PENALTY}","COMPILE_R1_TOOL_OVERCALL_FREE_CALLS":"${COMPILE_R1_TOOL_OVERCALL_FREE_CALLS}","COMPILE_R1_REWARD_PASS_ONLY":"${COMPILE_R1_REWARD_PASS_ONLY}","COMPILE_R1_REWARD_CLIP_MIN":"${COMPILE_R1_REWARD_CLIP_MIN}","COMPILE_R1_REWARD_CLIP_MAX":"${COMPILE_R1_REWARD_CLIP_MAX}","SLIME_PY_SANDBOX_BACKEND":"${COMPILE_R1_SANDBOX_BACKEND}","RUNPY_TOOL_MAX_WORKERS":"${COMPILE_R1_RUNPY_MAX_WORKERS}","SLIME_DISABLE_PACKED_SEQ":"${SLIME_DISABLE_PACKED_SEQ}","SLIME_ROLLOUT_START_PORT":"${SLIME_ROLLOUT_START_PORT}","SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK":"${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK}"}}
EOF
)"

echo "[compile-r1-grpo] EXP_NAME=${EXP_NAME}"
echo "[compile-r1-grpo] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[compile-r1-grpo] MODEL_DIR=${MODEL_DIR}"
echo "[compile-r1-grpo] HF_CHECKPOINT=${HF_CHECKPOINT}"
echo "[compile-r1-grpo] REF_LOAD=${REF_LOAD}"
echo "[compile-r1-grpo] ACTOR_LOAD=${ACTOR_LOAD}"
echo "[compile-r1-grpo] MEGATRON_TO_HF_MODE=${MEGATRON_TO_HF_MODE}"
echo "[compile-r1-grpo] TRAIN_DATA=${TRAIN_DATA}"
echo "[compile-r1-grpo] TEST_DATA=${TEST_DATA}"
echo "[compile-r1-grpo] SLIME_DISABLE_PACKED_SEQ=${SLIME_DISABLE_PACKED_SEQ}"
echo "[compile-r1-grpo] COMPILE_R1_ASYNC_TOOL_WORKERS=${COMPILE_R1_ASYNC_TOOL_WORKERS}"
echo "[compile-r1-grpo] COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER=${COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER}"
echo "[compile-r1-grpo] COMPILE_R1_TOOL_BONUS=${COMPILE_R1_TOOL_BONUS}"
echo "[compile-r1-grpo] COMPILE_R1_TOOL_BONUS_CAP=${COMPILE_R1_TOOL_BONUS_CAP}"
echo "[compile-r1-grpo] COMPILE_R1_TOOL_SUCCESS_BONUS=${COMPILE_R1_TOOL_SUCCESS_BONUS}"
echo "[compile-r1-grpo] COMPILE_R1_TOOL_SUCCESS_CAP=${COMPILE_R1_TOOL_SUCCESS_CAP}"
echo "[compile-r1-grpo] COMPILE_R1_TOOL_INVALID_PENALTY=${COMPILE_R1_TOOL_INVALID_PENALTY}"
echo "[compile-r1-grpo] COMPILE_R1_TOOL_OVERCALL_PENALTY=${COMPILE_R1_TOOL_OVERCALL_PENALTY}"
echo "[compile-r1-grpo] COMPILE_R1_TOOL_OVERCALL_FREE_CALLS=${COMPILE_R1_TOOL_OVERCALL_FREE_CALLS}"
echo "[compile-r1-grpo] COMPILE_R1_REWARD_PASS_ONLY=${COMPILE_R1_REWARD_PASS_ONLY}"
echo "[compile-r1-grpo] SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK}"
echo "[compile-r1-grpo] TRAIN_ENV_VARS_JSON=${TRAIN_ENV_VARS_JSON}"
echo "[compile-r1-grpo] SLIME_ROLLOUT_START_PORT=${SLIME_ROLLOUT_START_PORT}"

RAY_VERSION_URL="${RAY_JOB_ADDRESS%/}/api/version"
RAY_DASH_READY=0
for _ in $(seq 1 60); do
  if curl -fsS --connect-timeout 1 --max-time 2 "${RAY_VERSION_URL}" >/dev/null 2>&1; then
    RAY_DASH_READY=1
    break
  fi
  sleep 1
done

if (( RAY_DASH_READY == 0 )); then
  echo "Ray dashboard API not ready at ${RAY_VERSION_URL}" >&2
  exit 1
fi

submit_job() {
  local train_entrypoint=(
    python3 train.py
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${REF_LOAD}"
    "${LOAD_ARGS[@]}"
    --save "${CKPT_DIR}"
    --save-interval "${SAVE_INTERVAL}"
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key "${LABEL_KEY}"
    "${CHAT_TEMPLATE_ARGS[@]}"
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
    --save-debug-rollout-data "${DEBUG_DIR}/rollout_data/{rollout_id}.pt"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data humaneval_test "${TEST_DATA}"
    --eval-input-key prompt
    --eval-label-key "${EVAL_LABEL_KEY}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --eval-top-k 1
    --skip-eval-before-train
    --optimizer "${OPTIMIZER}"
    --lr "${LR}"
    --lr-decay-style "${LR_DECAY_STYLE}"
    --weight-decay "${WEIGHT_DECAY}"
    --adam-beta1 "${ADAM_BETA1}"
    --adam-beta2 "${ADAM_BETA2}"
    --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
    --entropy-coef "${ENTROPY_COEF}"
    --eps-clip "${EPS_CLIP}"
    --eps-clip-high "${EPS_CLIP_HIGH}"
    --gamma "${GAMMA}"
    --lambd "${LAMBDA}"
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
    "${SEQUENCE_PARALLEL_ARGS[@]}"
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    "${DYN_ARGS[@]}"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRAC}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --actor-num-nodes "${ACTOR_NUM_NODES}"
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
    "${COLOCATE_ARGS[@]}"
    --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
    --no-rope-fusion
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --use-wandb
    --wandb-mode "${WANDB_MODE}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${EXP_NAME}"
    --wandb-dir "${WANDB_DIR}"
    --disable-wandb-random-suffix
    --custom-generate-function-path generate_with_compile.generate
    --custom-rm-path generate_with_compile.reward_func
    "${TRAIN_ENV_VARS_ARGS[@]}"
    "${EXTRA_TRAIN_ARGS_ARR[@]}"
  )

  ray job submit --address="${RAY_JOB_ADDRESS}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${train_entrypoint[@]}"
}

run_direct() {
  export PYTHONPATH="${MEGATRON_DIR}:${COMPILE_R1_DIR}:${SLIME_DIR}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export MASTER_ADDR=127.0.0.1
  export RAY_ADDRESS="${RAY_STATUS_ADDRESS}"
  export COMPILE_R1_MAX_TURNS="${COMPILE_R1_MAX_TURNS}"
  export COMPILE_R1_TOOL_CONCURRENCY="${COMPILE_R1_TOOL_CONCURRENCY}"
  export COMPILE_R1_ASYNC_TOOL_WORKERS="${COMPILE_R1_ASYNC_TOOL_WORKERS}"
  export COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER="${COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER}"
  export COMPILE_R1_TIMEOUT_SECONDS="${COMPILE_R1_TIMEOUT_SECONDS}"
  export COMPILE_R1_MEMORY_MB="${COMPILE_R1_MEMORY_MB}"
  export COMPILE_R1_RETURN_LOGPROB="${COMPILE_R1_RETURN_LOGPROB}"
  export COMPILE_R1_FORMAT_SCORE="${COMPILE_R1_FORMAT_SCORE}"
  export COMPILE_R1_TOOL_BONUS="${COMPILE_R1_TOOL_BONUS}"
  export COMPILE_R1_TOOL_BONUS_CAP="${COMPILE_R1_TOOL_BONUS_CAP}"
  export COMPILE_R1_TOOL_SUCCESS_BONUS="${COMPILE_R1_TOOL_SUCCESS_BONUS}"
  export COMPILE_R1_TOOL_SUCCESS_CAP="${COMPILE_R1_TOOL_SUCCESS_CAP}"
  export COMPILE_R1_TOOL_INVALID_PENALTY="${COMPILE_R1_TOOL_INVALID_PENALTY}"
  export COMPILE_R1_TOOL_OVERCALL_PENALTY="${COMPILE_R1_TOOL_OVERCALL_PENALTY}"
  export COMPILE_R1_TOOL_OVERCALL_FREE_CALLS="${COMPILE_R1_TOOL_OVERCALL_FREE_CALLS}"
  export COMPILE_R1_REWARD_PASS_ONLY="${COMPILE_R1_REWARD_PASS_ONLY}"
  export COMPILE_R1_REWARD_CLIP_MIN="${COMPILE_R1_REWARD_CLIP_MIN}"
  export COMPILE_R1_REWARD_CLIP_MAX="${COMPILE_R1_REWARD_CLIP_MAX}"
  export SLIME_PY_SANDBOX_BACKEND="${COMPILE_R1_SANDBOX_BACKEND}"
  export RUNPY_TOOL_MAX_WORKERS="${COMPILE_R1_RUNPY_MAX_WORKERS}"
  export SLIME_DISABLE_PACKED_SEQ="${SLIME_DISABLE_PACKED_SEQ}"
  export SLIME_ROLLOUT_START_PORT="${SLIME_ROLLOUT_START_PORT}"
  export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK}"

  python3 train.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${REF_LOAD}" \
    "${LOAD_ARGS[@]}" \
    --save "${CKPT_DIR}" \
    --save-interval "${SAVE_INTERVAL}" \
    --prompt-data "${TRAIN_DATA}" \
    --input-key prompt \
    --label-key "${LABEL_KEY}" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    --rollout-shuffle \
    --num-rollout "${NUM_ROLLOUT}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-temperature "${ROLLOUT_TEMPERATURE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --balance-data \
    --save-debug-rollout-data "${DEBUG_DIR}/rollout_data/{rollout_id}.pt" \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-prompt-data humaneval_test "${TEST_DATA}" \
    --eval-input-key prompt \
    --eval-label-key "${EVAL_LABEL_KEY}" \
    --n-samples-per-eval-prompt 1 \
    --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --eval-top-k 1 \
    --skip-eval-before-train \
    --optimizer "${OPTIMIZER}" \
    --lr "${LR}" \
    --lr-decay-style "${LR_DECAY_STYLE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --adam-beta1 "${ADAM_BETA1}" \
    --adam-beta2 "${ADAM_BETA2}" \
    --advantage-estimator "${ADVANTAGE_ESTIMATOR}" \
    --entropy-coef "${ENTROPY_COEF}" \
    --eps-clip "${EPS_CLIP}" \
    --eps-clip-high "${EPS_CLIP_HIGH}" \
    --gamma "${GAMMA}" \
    --lambd "${LAMBDA}" \
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
    "${SEQUENCE_PARALLEL_ARGS[@]}" \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    "${DYN_ARGS[@]}" \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRAC}" \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
    --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}" \
    "${COLOCATE_ARGS[@]}" \
    --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}" \
    --no-rope-fusion \
    --no-persist-layer-norm \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --use-wandb \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${EXP_NAME}" \
    --wandb-dir "${WANDB_DIR}" \
    --disable-wandb-random-suffix \
    --custom-generate-function-path generate_with_compile.generate \
    --custom-rm-path generate_with_compile.reward_func \
    "${TRAIN_ENV_VARS_ARGS[@]}" \
    "${EXTRA_TRAIN_ARGS_ARR[@]}"
}

: > "${LOG_DIR}/train_console.log"

if is_true "${FORCE_DIRECT_RUN}"; then
  echo "[compile-r1-grpo] FORCE_DIRECT_RUN=1, skip ray job submit and run direct." | tee -a "${LOG_DIR}/train_console.log"
  run_direct 2>&1 | tee -a "${LOG_DIR}/train_console.log"
  exit 0
fi

SUBMIT_OK=0
for ATTEMPT in $(seq 1 "${RAY_JOB_SUBMIT_RETRIES}"); do
  echo "[compile-r1-grpo] ray job submit attempt ${ATTEMPT}/${RAY_JOB_SUBMIT_RETRIES}" | tee -a "${LOG_DIR}/train_console.log"
  set +e
  submit_job 2>&1 | tee -a "${LOG_DIR}/train_console.log"
  SUBMIT_RC=${PIPESTATUS[0]}
  set -e
  if (( SUBMIT_RC == 0 )); then
    SUBMIT_OK=1
    break
  fi
  if tail -n 120 "${LOG_DIR}/train_console.log" | grep -q "No available agent to submit job"; then
    echo "[compile-r1-grpo] Ray agent not ready yet, retry after ${RAY_JOB_SUBMIT_RETRY_DELAY}s." | tee -a "${LOG_DIR}/train_console.log"
    sleep "${RAY_JOB_SUBMIT_RETRY_DELAY}"
    continue
  fi
  exit "${SUBMIT_RC}"
done

if (( SUBMIT_OK == 0 )); then
  if is_true "${DIRECT_RUN_ON_SUBMIT_FAIL}"; then
    echo "[compile-r1-grpo] ray job submit failed after retries, fallback to direct run." | tee -a "${LOG_DIR}/train_console.log"
    run_direct 2>&1 | tee -a "${LOG_DIR}/train_console.log"
  else
    echo "[compile-r1-grpo] ray job submit failed after retries." >&2
    exit 1
  fi
fi
