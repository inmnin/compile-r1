#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/workspace/jkh}"
SLIME_DIR="${SLIME_DIR:-${BASE_DIR}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${BASE_DIR}/Megatron-LM}"
COMPILE_R1_DIR="${COMPILE_R1_DIR:-${SLIME_DIR}/examples/compile-r1}"

DEFAULT_CONFIG_FILE="${COMPILE_R1_DIR}/configs/precheck.yaml"
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

MODEL_DIR="${MODEL_DIR:-${BASE_DIR}/model/Qwen2.5-3B}"
TRAIN_DATA="${TRAIN_DATA:-${TRAIN_PARQUET:-${COMPILE_R1_DIR}/data/train}}"
TEST_DATA="${TEST_DATA:-${TEST_PARQUET:-${COMPILE_R1_DIR}/data/test}}"

NUM_GPUS="${NUM_GPUS:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"

NUM_ROLLOUT="${NUM_ROLLOUT:-30}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"

EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

COMPILE_R1_MAX_TURNS="${COMPILE_R1_MAX_TURNS:-3}"
COMPILE_R1_TOOL_CONCURRENCY="${COMPILE_R1_TOOL_CONCURRENCY:-64}"
COMPILE_R1_TIMEOUT_SECONDS="${COMPILE_R1_TIMEOUT_SECONDS:-6}"
COMPILE_R1_MEMORY_MB="${COMPILE_R1_MEMORY_MB:-768}"
COMPILE_R1_RETURN_LOGPROB="${COMPILE_R1_RETURN_LOGPROB:-1}"
COMPILE_R1_FORMAT_SCORE="${COMPILE_R1_FORMAT_SCORE:-0.15}"
COMPILE_R1_TOOL_BONUS="${COMPILE_R1_TOOL_BONUS:-0.05}"
COMPILE_R1_SANDBOX_BACKEND="${COMPILE_R1_SANDBOX_BACKEND:-runpytool}"
COMPILE_R1_RUNPY_MAX_WORKERS="${COMPILE_R1_RUNPY_MAX_WORKERS:-4}"

export PYTHONBUFFERED=16

if [[ ! -e "${TRAIN_DATA}" ]]; then
  echo "Missing train data path: ${TRAIN_DATA}" >&2
  exit 1
fi
if [[ ! -e "${TEST_DATA}" ]]; then
  echo "Missing test data path: ${TEST_DATA}" >&2
  exit 1
fi

mkdir -p "${COMPILE_R1_DIR}/ckpt"

cd "${SLIME_DIR}"

ray stop --force >/dev/null 2>&1 || true
pkill -f sglang >/dev/null 2>&1 || true
sleep 2
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats

source "${SLIME_DIR}/scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${MODEL_DIR}"
  --save "${COMPILE_R1_DIR}/ckpt"
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt
  --label-key reward_model
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

EVAL_ARGS=(
  --eval-interval "${EVAL_INTERVAL}"
  --eval-prompt-data humaneval_test "${TEST_DATA}"
  --eval-input-key prompt
  --eval-label-key reward_model
  --n-samples-per-eval-prompt 1
  --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --eval-top-k 1
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
  --gamma 1.0
  --lambd 1.0
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.3
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --colocate
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_compile.generate
  --custom-rm-path generate_with_compile.reward_func
)

RUNTIME_ENV_JSON="$(cat <<EOF
{"env_vars":{"PYTHONPATH":"${MEGATRON_DIR}:${COMPILE_R1_DIR}:${SLIME_DIR}","CUDA_DEVICE_MAX_CONNECTIONS":"1","MASTER_ADDR":"127.0.0.1","COMPILE_R1_MAX_TURNS":"${COMPILE_R1_MAX_TURNS}","COMPILE_R1_TOOL_CONCURRENCY":"${COMPILE_R1_TOOL_CONCURRENCY}","COMPILE_R1_TIMEOUT_SECONDS":"${COMPILE_R1_TIMEOUT_SECONDS}","COMPILE_R1_MEMORY_MB":"${COMPILE_R1_MEMORY_MB}","COMPILE_R1_RETURN_LOGPROB":"${COMPILE_R1_RETURN_LOGPROB}","COMPILE_R1_FORMAT_SCORE":"${COMPILE_R1_FORMAT_SCORE}","COMPILE_R1_TOOL_BONUS":"${COMPILE_R1_TOOL_BONUS}","SLIME_PY_SANDBOX_BACKEND":"${COMPILE_R1_SANDBOX_BACKEND}","RUNPY_TOOL_MAX_WORKERS":"${COMPILE_R1_RUNPY_MAX_WORKERS}"}}
EOF
)"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"
