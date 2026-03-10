#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/workspace/jkh}"
# Optional config file. By default, load Search-R1 full-train YAML config if present.
DEFAULT_CONFIG_FILE="${BASE_DIR}/slime/examples/search-r1/configs/full_train.yaml"
CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
if [[ -n "${CONFIG_FILE}" && -f "${CONFIG_FILE}" ]]; then
  case "${CONFIG_FILE}" in
    *.yaml|*.yml)
      set +u
      eval "$(
        python - "${CONFIG_FILE}" <<'PY'
import shlex
import sys
import re

import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

if not isinstance(data, dict):
    raise SystemExit(f"Top-level YAML object must be a mapping: {path}")

for k, v in data.items():
    if not isinstance(k, str):
        continue
    key = re.sub(r"[^A-Z0-9_]", "_", k.strip().upper())
    if key and key[0].isdigit():
        key = f"CFG_{key}"
    if not key:
        continue
    if v is None:
        value = ""
    elif isinstance(v, bool):
        value = "1" if v else "0"
    else:
        value = str(v)
    print(f"{key}={shlex.quote(value)}")
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
      echo "Unsupported config extension for CONFIG_FILE=${CONFIG_FILE}" >&2
      echo "Use .yaml/.yml or .env/.sh" >&2
      exit 1
      ;;
  esac
fi

SLIME_DIR="${SLIME_DIR:-${BASE_DIR}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${BASE_DIR}/Megatron-LM}"
MODEL_DIR="${MODEL_DIR:-${BASE_DIR}/model/Qwen3-4B}"
SEARCH_R1_DIR="${SEARCH_R1_DIR:-${SLIME_DIR}/examples/search-r1}"

DATA_DIR="${DATA_DIR:-${SEARCH_R1_DIR}/data}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/train_97.parquet}"
TEST_PARQUET="${TEST_PARQUET:-${DATA_DIR}/test_3.parquet}"

RUN_TAG_TS="$(date +%Y%m%d-%H%M%S)"
EXP_NAME="${EXP_NAME:-search-r1-grpo-google-${RUN_TAG_TS}}"
RUN_ROOT="${SEARCH_R1_DIR}/train_log/${EXP_NAME}"
CKPT_DIR="${RUN_ROOT}/ckpt"
WANDB_DIR="${RUN_ROOT}/wandb"
DEBUG_DIR="${RUN_ROOT}/debug"
LOG_DIR="${RUN_ROOT}/logs"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-search-r1}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_NAME}}"
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
if [[ -n "${WANDB_API_KEY}" && ${#WANDB_API_KEY} -lt 40 ]]; then
  echo "WARN: WANDB_API_KEY length (${#WANDB_API_KEY}) is invalid for W&B, ignoring explicit key and using local login state."
  WANDB_API_KEY=""
fi

SEARCH_BACKEND="${SEARCH_BACKEND:-google}"
LOCAL_SEARCH_URL="${LOCAL_SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
LOCAL_SEARCH_PROXY="${LOCAL_SEARCH_PROXY:-}"
SEARCH_R1_SERPER_API_KEY="${SEARCH_R1_SERPER_API_KEY:-}"
if [[ "${SEARCH_BACKEND}" == "google" ]]; then
  if [[ -z "${SEARCH_R1_SERPER_API_KEY}" ]]; then
    echo "SEARCH_R1_SERPER_API_KEY is required when SEARCH_BACKEND=google." >&2
    exit 1
  fi
elif [[ "${SEARCH_BACKEND}" == "local" ]]; then
  echo "Using local search backend via ${LOCAL_SEARCH_URL}"
else
  echo "Unsupported SEARCH_BACKEND=${SEARCH_BACKEND}, expected google or local." >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-}"
CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-10000}"
SELECTED_GPUS="${SELECTED_GPUS:-}"

NUM_ROLLOUT="${NUM_ROLLOUT:-30}"
NUM_EPOCH="${NUM_EPOCH:-}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_TOP_K="${EVAL_TOP_K:-1}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"

SEARCH_CONCURRENCY="${SEARCH_CONCURRENCY:-2}"
GOOGLE_MIN_INTERVAL="${GOOGLE_MIN_INTERVAL:-0.6}"
GOOGLE_MAX_RETRIES="${GOOGLE_MAX_RETRIES:-10}"
GOOGLE_BACKOFF_BASE="${GOOGLE_BACKOFF_BASE:-1.5}"
GOOGLE_BACKOFF_CAP="${GOOGLE_BACKOFF_CAP:-30}"
SEARCH_MAX_TURNS="${SEARCH_MAX_TURNS:-2}"
SEARCH_TOPK="${SEARCH_TOPK:-3}"
GOOGLE_SNIPPET_ONLY="${GOOGLE_SNIPPET_ONLY:-1}"
SEARCH_RETURN_LOGPROB="${SEARCH_RETURN_LOGPROB:-1}"

SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-15}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-3072}"
SGLANG_MEM_FRAC="${SGLANG_MEM_FRAC:-0.15}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-0}"
ENABLE_COLOCATE="${ENABLE_COLOCATE:-1}"
ENABLE_OFFLOAD_ROLLOUT="${ENABLE_OFFLOAD_ROLLOUT:-1}"
ENABLE_DYNAMIC_BATCH_SIZE="${ENABLE_DYNAMIC_BATCH_SIZE:-1}"
DISABLE_PACKED_SEQ="${DISABLE_PACKED_SEQ:-0}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
MODEL_NUM_QUERY_GROUPS="${MODEL_NUM_QUERY_GROUPS:-2}"
NO_ROPE_FUSION="${NO_ROPE_FUSION:-1}"
NO_PERSIST_LAYER_NORM="${NO_PERSIST_LAYER_NORM:-1}"
NO_GRADIENT_ACCUMULATION_FUSION="${NO_GRADIENT_ACCUMULATION_FUSION:-1}"
NO_MASKED_SOFTMAX_FUSION="${NO_MASKED_SOFTMAX_FUSION:-0}"

OPTIMIZER="${OPTIMIZER:-adam}"
LR="${LR:-1e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
EPS_CLIP_C="${EPS_CLIP_C:-}"
VALUE_CLIP="${VALUE_CLIP:-0.2}"
KL_COEF="${KL_COEF:-0.0}"
LOSS_TYPE="${LOSS_TYPE:-policy_loss}"
CUSTOM_LOSS_FUNCTION_PATH="${CUSTOM_LOSS_FUNCTION_PATH:-}"
COMPUTE_ADVANTAGES_AND_RETURNS="${COMPUTE_ADVANTAGES_AND_RETURNS:-1}"
USE_KL_LOSS="${USE_KL_LOSS:-0}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-k1}"
USE_UNBIASED_KL="${USE_UNBIASED_KL:-0}"
REF_UPDATE_INTERVAL="${REF_UPDATE_INTERVAL:-}"
GAMMA="${GAMMA:-1.0}"
LAMBDA="${LAMBDA:-1.0}"
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-0}"
GRPO_STD_NORMALIZATION="${GRPO_STD_NORMALIZATION:-1}"
REWARDS_NORMALIZATION="${REWARDS_NORMALIZATION:-1}"
USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
USE_ROLLOUT_ENTROPY="${USE_ROLLOUT_ENTROPY:-0}"
RESET_OPTIMIZER_STATES="${RESET_OPTIMIZER_STATES:-0}"
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-0}"
CRITIC_LOAD="${CRITIC_LOAD:-}"
CRITIC_SAVE="${CRITIC_SAVE:-}"
CRITIC_LR="${CRITIC_LR:-}"
CRITIC_LR_WARMUP_ITERS="${CRITIC_LR_WARMUP_ITERS:-0}"
USE_TIS="${USE_TIS:-0}"
TIS_CLIP="${TIS_CLIP:-2.0}"
TIS_CLIP_LOW="${TIS_CLIP_LOW:-0.0}"
GET_MISMATCH_METRICS="${GET_MISMATCH_METRICS:-0}"
CUSTOM_TIS_FUNCTION_PATH="${CUSTOM_TIS_FUNCTION_PATH:-}"
CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH="${CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH:-}"
USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-0}"
USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-0}"
USE_OPD="${USE_OPD:-0}"
OPD_TYPE="${OPD_TYPE:-}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
OPD_TEACHER_LOAD="${OPD_TEACHER_LOAD:-}"
OPD_TEACHER_CKPT_STEP="${OPD_TEACHER_CKPT_STEP:-}"
USE_OPSM="${USE_OPSM:-0}"
OPSM_DELTA="${OPSM_DELTA:-1e-4}"
REF_CKPT_STEP="${REF_CKPT_STEP:-}"

PROXY_REFRESH_SECS="${PROXY_REFRESH_SECS:-300}"
EXPORT_TEST_TRAJ="${EXPORT_TEST_TRAJ:-1}"
TEST_EXPORT_MODE="${TEST_EXPORT_MODE:-final}"
TEST_EXPORT_EVERY_N_EVAL="${TEST_EXPORT_EVERY_N_EVAL:-1}"
# Safety switch: do not kill shared sglang services unless explicitly requested.
CLEANUP_SGLANG_PROCS="${CLEANUP_SGLANG_PROCS:-0}"
export http_proxy="http://127.0.0.1:7898"
export https_proxy="http://127.0.0.1:7898"
export all_proxy="http://127.0.0.1:7898"

MAMBA_EXE="${MAMBA_EXE:-${HOME}/.local/bin/micromamba}"
CONDA_EXE="${CONDA_EXE:-${BASE_DIR}/miniconda3/bin/conda}"
TRAIN_ENV_NAME="${TRAIN_ENV_NAME:-slime}"

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
if (( MODEL_NUM_QUERY_GROUPS % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
  if (( NUM_GPUS >= 2 )) && (( MODEL_NUM_QUERY_GROUPS % 2 == 0 )); then
    echo "WARN: tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} is incompatible with num_query_groups=${MODEL_NUM_QUERY_GROUPS}, fallback to 2."
    TENSOR_MODEL_PARALLEL_SIZE=2
  else
    echo "WARN: tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} is incompatible with num_query_groups=${MODEL_NUM_QUERY_GROUPS}, fallback to 1."
    TENSOR_MODEL_PARALLEL_SIZE=1
  fi
fi
if (( ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
  echo "ROLLOUT_NUM_GPUS must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE." >&2
  exit 1
fi
if is_true "${USE_KL_LOSS}" && awk "BEGIN {exit !(${KL_COEF} != 0 && ${KL_LOSS_COEF} != 0)}"; then
  echo "USE_KL_LOSS=1 with KL_COEF!=0 and KL_LOSS_COEF!=0 is invalid; set one of them to 0." >&2
  exit 1
fi
if is_true "${USE_OPD}" && [[ -z "${OPD_TYPE}" ]]; then
  echo "USE_OPD=1 requires OPD_TYPE to be set to 'sglang' or 'megatron'." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}" "${CKPT_DIR}" "${WANDB_DIR}" "${DEBUG_DIR}" "${LOG_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
set +u
if [[ -x "${MAMBA_EXE}" ]]; then
  eval "$("${MAMBA_EXE}" shell hook --shell bash)"
  if ! micromamba activate "${TRAIN_ENV_NAME}"; then
    echo "Failed to activate micromamba env '${TRAIN_ENV_NAME}'." >&2
    exit 1
  fi
elif [[ -x "${CONDA_EXE}" ]]; then
  eval "$("${CONDA_EXE}" shell.bash hook)"
  if ! conda activate "${TRAIN_ENV_NAME}" 2>/dev/null; then
    for fallback_env in r1 lf base; do
      if conda activate "${fallback_env}" 2>/dev/null; then
        echo "WARN: env '${TRAIN_ENV_NAME}' not found, fallback to '${fallback_env}'."
        TRAIN_ENV_NAME="${fallback_env}"
        break
      fi
    done
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${TRAIN_ENV_NAME}" ]]; then
    echo "Failed to activate conda env '${TRAIN_ENV_NAME}' via ${CONDA_EXE}." >&2
    exit 1
  fi
else
  echo "Neither micromamba nor conda was found." >&2
  echo "Checked MAMBA_EXE=${MAMBA_EXE} and CONDA_EXE=${CONDA_EXE}" >&2
  exit 1
fi
set -u

cd "${SLIME_DIR}"

export PYTHONBUFFERED=16
# Prefer a CUDA_HOME that actually contains nvcc; FlashInfer JIT needs it.
# Always prefer the real CUDA toolkit root to avoid missing headers when nvcc
# in conda env is only a symlink wrapper.
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  CUDA_HOME="${CUDA_HOME}"
elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
  CUDA_HOME="/usr/local/cuda"
elif [[ -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  CONDA_NVCC_REAL="$(readlink -f "${CONDA_PREFIX}/bin/nvcc" || true)"
  if [[ "${CONDA_NVCC_REAL}" == "/usr/local/cuda/bin/nvcc" ]]; then
    CUDA_HOME="/usr/local/cuda"
  else
    CUDA_HOME="${CONDA_PREFIX}"
  fi
elif command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
  CUDA_HOME="${CONDA_PREFIX}"
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${MEGATRON_DIR}:${SEARCH_R1_DIR}:${PYTHONPATH:-}"
LOCAL_NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
LOCAL_NODE_IP="${LOCAL_NODE_IP:-127.0.0.1}"
# Local retrieval does not require outbound proxy; keep internal engine checks direct.
if [[ "${SEARCH_BACKEND}" == "local" ]]; then
  export http_proxy=""
  export https_proxy=""
  export all_proxy=""
fi
export no_proxy="127.0.0.1,localhost,${LOCAL_NODE_IP}"
export NO_PROXY="${no_proxy}"
export http_proxy
export https_proxy
export all_proxy
export SLIME_DISABLE_PACKED_SEQ="${DISABLE_PACKED_SEQ}"

# Best-effort proxy refresh loop (note: environment vars are inherited at process start).
if [[ "${PROXY_REFRESH_SECS}" -gt 0 ]] && [[ -n "${http_proxy}" || -n "${https_proxy}" || -n "${all_proxy}" ]]; then
  PROXY_HTTP="${http_proxy}"
  PROXY_HTTPS="${https_proxy}"
  PROXY_ALL="${all_proxy}"
  (
    while true; do
      export http_proxy="${PROXY_HTTP}"
      export https_proxy="${PROXY_HTTPS}"
      export all_proxy="${PROXY_ALL}"
      sleep "${PROXY_REFRESH_SECS}"
    done
  ) >/dev/null 2>&1 &
fi

# FlashInfer JIT compatibility for local conda CUDA layout.
if [[ ! -e "${CUDA_HOME}/lib64" && -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]]; then
  ln -s "${CUDA_HOME}/targets/x86_64-linux/lib" "${CUDA_HOME}/lib64"
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing MODEL_DIR: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
  echo "Missing train parquet: ${TRAIN_PARQUET}" >&2
  exit 1
fi
if [[ ! -f "${TEST_PARQUET}" ]]; then
  echo "Missing test parquet: ${TEST_PARQUET}" >&2
  exit 1
fi

# PPO 使用 critic，若未显式设置保存目录则回落到本次 run 的 ckpt 目录，
# 避免 critic 在保存阶段拿到 None 导致异常退出。
if [[ "${ADVANTAGE_ESTIMATOR}" == "ppo" && -z "${CRITIC_SAVE}" ]]; then
  CRITIC_SAVE="${CKPT_DIR}"
fi

{
  echo "date=$(date -Iseconds)"
  echo "exp_name=${EXP_NAME}"
  echo "config_file=${CONFIG_FILE}"
  echo "run_root=${RUN_ROOT}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "cuda_home=${CUDA_HOME}"
  echo "num_gpus=${NUM_GPUS}"
  echo "actor_num_nodes=${ACTOR_NUM_NODES}"
  echo "actor_num_gpus_per_node=${ACTOR_NUM_GPUS_PER_NODE}"
  echo "critic_num_nodes=${CRITIC_NUM_NODES}"
  echo "critic_num_gpus_per_node=${CRITIC_NUM_GPUS_PER_NODE}"
  echo "rollout_num_gpus=${ROLLOUT_NUM_GPUS}"
  echo "rollout_num_gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  echo "tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}"
  echo "num_rollout=${NUM_ROLLOUT}"
  echo "num_epoch=${NUM_EPOCH}"
  echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
  echo "n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}"
  echo "n_samples_per_eval_prompt=${N_SAMPLES_PER_EVAL_PROMPT}"
  echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
  echo "rollout_max_response_len=${ROLLOUT_MAX_RESPONSE_LEN}"
  echo "rollout_temperature=${ROLLOUT_TEMPERATURE}"
  echo "skip_eval_before_train=${SKIP_EVAL_BEFORE_TRAIN}"
  echo "search_concurrency=${SEARCH_CONCURRENCY}"
  echo "search_max_turns=${SEARCH_MAX_TURNS}"
  echo "search_topk=${SEARCH_TOPK}"
  echo "google_snippet_only=${GOOGLE_SNIPPET_ONLY}"
  echo "google_min_interval=${GOOGLE_MIN_INTERVAL}"
  echo "google_max_retries=${GOOGLE_MAX_RETRIES}"
  echo "google_backoff_base=${GOOGLE_BACKOFF_BASE}"
  echo "google_backoff_cap=${GOOGLE_BACKOFF_CAP}"
  echo "search_return_logprob=${SEARCH_RETURN_LOGPROB}"
  echo "eval_interval=${EVAL_INTERVAL}"
  echo "eval_top_k=${EVAL_TOP_K}"
  echo "save_interval=${SAVE_INTERVAL}"
  echo "export_test_traj=${EXPORT_TEST_TRAJ}"
  echo "test_export_mode=${TEST_EXPORT_MODE}"
  echo "test_export_every_n_eval=${TEST_EXPORT_EVERY_N_EVAL}"
  echo "max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
  echo "sglang_mem_frac=${SGLANG_MEM_FRAC}"
  echo "enable_colocate=${ENABLE_COLOCATE}"
  echo "enable_offload_rollout=${ENABLE_OFFLOAD_ROLLOUT}"
  echo "enable_dynamic_batch_size=${ENABLE_DYNAMIC_BATCH_SIZE}"
  echo "disable_packed_seq=${DISABLE_PACKED_SEQ}"
  echo "no_rope_fusion=${NO_ROPE_FUSION}"
  echo "no_persist_layer_norm=${NO_PERSIST_LAYER_NORM}"
  echo "no_gradient_accumulation_fusion=${NO_GRADIENT_ACCUMULATION_FUSION}"
  echo "no_masked_softmax_fusion=${NO_MASKED_SOFTMAX_FUSION}"
  echo "optimizer=${OPTIMIZER}"
  echo "lr=${LR}"
  echo "lr_decay_style=${LR_DECAY_STYLE}"
  echo "weight_decay=${WEIGHT_DECAY}"
  echo "adam_beta1=${ADAM_BETA1}"
  echo "adam_beta2=${ADAM_BETA2}"
  echo "clip_grad=${CLIP_GRAD}"
  echo "advantage_estimator=${ADVANTAGE_ESTIMATOR}"
  echo "entropy_coef=${ENTROPY_COEF}"
  echo "eps_clip=${EPS_CLIP}"
  echo "eps_clip_high=${EPS_CLIP_HIGH}"
  echo "eps_clip_c=${EPS_CLIP_C}"
  echo "value_clip=${VALUE_CLIP}"
  echo "kl_coef=${KL_COEF}"
  echo "loss_type=${LOSS_TYPE}"
  echo "custom_loss_function_path=${CUSTOM_LOSS_FUNCTION_PATH}"
  echo "compute_advantages_and_returns=${COMPUTE_ADVANTAGES_AND_RETURNS}"
  echo "use_kl_loss=${USE_KL_LOSS}"
  echo "kl_loss_coef=${KL_LOSS_COEF}"
  echo "kl_loss_type=${KL_LOSS_TYPE}"
  echo "use_unbiased_kl=${USE_UNBIASED_KL}"
  echo "ref_update_interval=${REF_UPDATE_INTERVAL}"
  echo "gamma=${GAMMA}"
  echo "lambd=${LAMBDA}"
  echo "normalize_advantages=${NORMALIZE_ADVANTAGES}"
  echo "grpo_std_normalization=${GRPO_STD_NORMALIZATION}"
  echo "rewards_normalization=${REWARDS_NORMALIZATION}"
  echo "use_rollout_logprobs=${USE_ROLLOUT_LOGPROBS}"
  echo "use_rollout_entropy=${USE_ROLLOUT_ENTROPY}"
  echo "reset_optimizer_states=${RESET_OPTIMIZER_STATES}"
  echo "num_critic_only_steps=${NUM_CRITIC_ONLY_STEPS}"
  echo "critic_load=${CRITIC_LOAD}"
  echo "critic_save=${CRITIC_SAVE}"
  echo "critic_lr=${CRITIC_LR}"
  echo "critic_lr_warmup_iters=${CRITIC_LR_WARMUP_ITERS}"
  echo "use_tis=${USE_TIS}"
  echo "tis_clip=${TIS_CLIP}"
  echo "tis_clip_low=${TIS_CLIP_LOW}"
  echo "get_mismatch_metrics=${GET_MISMATCH_METRICS}"
  echo "custom_tis_function_path=${CUSTOM_TIS_FUNCTION_PATH}"
  echo "custom_pg_loss_reducer_function_path=${CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH}"
  echo "use_routing_replay=${USE_ROUTING_REPLAY}"
  echo "use_rollout_routing_replay=${USE_ROLLOUT_ROUTING_REPLAY}"
  echo "use_opd=${USE_OPD}"
  echo "opd_type=${OPD_TYPE}"
  echo "opd_kl_coef=${OPD_KL_COEF}"
  echo "opd_teacher_load=${OPD_TEACHER_LOAD}"
  echo "opd_teacher_ckpt_step=${OPD_TEACHER_CKPT_STEP}"
  echo "use_opsm=${USE_OPSM}"
  echo "opsm_delta=${OPSM_DELTA}"
  echo "ref_ckpt_step=${REF_CKPT_STEP}"
  echo "wandb_project=${WANDB_PROJECT}"
  echo "wandb_group=${WANDB_GROUP}"
  echo "search_backend=${SEARCH_BACKEND}"
  echo "local_search_url=${LOCAL_SEARCH_URL}"
  echo "train_parquet=${TRAIN_PARQUET}"
  echo "test_parquet=${TEST_PARQUET}"
} > "${RUN_ROOT}/run_config.env"

ray stop --force >/dev/null 2>&1 || true
if [[ "${CLEANUP_SGLANG_PROCS}" == "1" ]]; then
  # Only clean up child processes of this launcher when explicitly enabled.
  pkill -P $$ -f sglang >/dev/null 2>&1 || true
fi
sleep 2

ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SLIME_DIR}/scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${MODEL_DIR}"
  --save "${CKPT_DIR}"
  --save-interval "${SAVE_INTERVAL}"
  --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
  --critic-lr-warmup-iters "${CRITIC_LR_WARMUP_ITERS}"
)
if [[ -n "${REF_CKPT_STEP}" ]]; then
  CKPT_ARGS+=(--ref-ckpt-step "${REF_CKPT_STEP}")
fi
if [[ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
  CKPT_ARGS+=(--load "${CKPT_DIR}")
fi
if [[ -n "${CRITIC_LOAD}" ]]; then
  CKPT_ARGS+=(--critic-load "${CRITIC_LOAD}")
fi
if [[ -n "${CRITIC_SAVE}" ]]; then
  CKPT_ARGS+=(--critic-save "${CRITIC_SAVE}")
fi
if [[ -n "${CRITIC_LR}" ]]; then
  CKPT_ARGS+=(--critic-lr "${CRITIC_LR}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_PARQUET}"
  --input-key prompt
  --label-key reward_model
  --apply-chat-template
  --rollout-shuffle
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
  --save-debug-rollout-data "${DEBUG_DIR}/rollout_data/{rollout_id}.pt"
)
if [[ -n "${NUM_EPOCH}" ]]; then
  ROLLOUT_ARGS+=(--num-epoch "${NUM_EPOCH}")
else
  ROLLOUT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
fi

EVAL_ARGS=(
  --eval-interval "${EVAL_INTERVAL}"
  --eval-prompt-data search_r1_test "${TEST_PARQUET}"
  --eval-input-key prompt
  --eval-label-key reward_model
  --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
  --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --eval-top-k "${EVAL_TOP_K}"
)
if is_true "${SKIP_EVAL_BEFORE_TRAIN}"; then
  EVAL_ARGS+=(--skip-eval-before-train)
fi

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
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if is_true "${ENABLE_DYNAMIC_BATCH_SIZE}"; then
  PERF_ARGS+=(--use-dynamic-batch-size)
fi

GRPO_ARGS=(
  --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip "${EPS_CLIP}"
  --value-clip "${VALUE_CLIP}"
  --gamma "${GAMMA}"
  --lambd "${LAMBDA}"
  --kl-coef "${KL_COEF}"
  --loss-type "${LOSS_TYPE}"
)
if ! is_true "${COMPUTE_ADVANTAGES_AND_RETURNS}"; then
  GRPO_ARGS+=(--disable-compute-advantages-and-returns)
fi
if [[ -n "${CUSTOM_LOSS_FUNCTION_PATH}" ]]; then
  GRPO_ARGS+=(--custom-loss-function-path "${CUSTOM_LOSS_FUNCTION_PATH}")
fi
if [[ -n "${EPS_CLIP_HIGH}" ]]; then
  GRPO_ARGS+=(--eps-clip-high "${EPS_CLIP_HIGH}")
fi
if [[ -n "${EPS_CLIP_C}" ]]; then
  GRPO_ARGS+=(--eps-clip-c "${EPS_CLIP_C}")
fi
if is_true "${USE_KL_LOSS}"; then
  GRPO_ARGS+=(
    --use-kl-loss
    --kl-loss-coef "${KL_LOSS_COEF}"
    --kl-loss-type "${KL_LOSS_TYPE}"
  )
fi
if is_true "${USE_UNBIASED_KL}"; then
  GRPO_ARGS+=(--use-unbiased-kl)
fi
if [[ -n "${REF_UPDATE_INTERVAL}" ]]; then
  GRPO_ARGS+=(--ref-update-interval "${REF_UPDATE_INTERVAL}")
fi
if is_true "${NORMALIZE_ADVANTAGES}"; then
  GRPO_ARGS+=(--normalize-advantages)
fi
if ! is_true "${GRPO_STD_NORMALIZATION}"; then
  GRPO_ARGS+=(--disable-grpo-std-normalization)
fi
if ! is_true "${REWARDS_NORMALIZATION}"; then
  GRPO_ARGS+=(--disable-rewards-normalization)
fi
if is_true "${USE_ROLLOUT_LOGPROBS}"; then
  GRPO_ARGS+=(--use-rollout-logprobs)
fi
if is_true "${USE_ROLLOUT_ENTROPY}"; then
  GRPO_ARGS+=(--use-rollout-entropy)
fi
if is_true "${RESET_OPTIMIZER_STATES}"; then
  GRPO_ARGS+=(--reset-optimizer-states)
fi
if is_true "${GET_MISMATCH_METRICS}"; then
  GRPO_ARGS+=(--get-mismatch-metrics)
fi
if is_true "${USE_TIS}"; then
  GRPO_ARGS+=(
    --use-tis
    --tis-clip "${TIS_CLIP}"
    --tis-clip-low "${TIS_CLIP_LOW}"
  )
fi
if [[ -n "${CUSTOM_TIS_FUNCTION_PATH}" ]]; then
  GRPO_ARGS+=(--custom-tis-function-path "${CUSTOM_TIS_FUNCTION_PATH}")
fi
if [[ -n "${CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH}" ]]; then
  GRPO_ARGS+=(--custom-pg-loss-reducer-function-path "${CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH}")
fi
if is_true "${USE_ROUTING_REPLAY}"; then
  GRPO_ARGS+=(--use-routing-replay)
fi
if is_true "${USE_ROLLOUT_ROUTING_REPLAY}"; then
  GRPO_ARGS+=(--use-rollout-routing-replay)
fi
if is_true "${USE_OPD}"; then
  GRPO_ARGS+=(
    --use-opd
    --opd-type "${OPD_TYPE}"
    --opd-kl-coef "${OPD_KL_COEF}"
  )
  if [[ -n "${OPD_TEACHER_LOAD}" ]]; then
    GRPO_ARGS+=(--opd-teacher-load "${OPD_TEACHER_LOAD}")
  fi
  if [[ -n "${OPD_TEACHER_CKPT_STEP}" ]]; then
    GRPO_ARGS+=(--opd-teacher-ckpt-step "${OPD_TEACHER_CKPT_STEP}")
  fi
fi
if is_true "${USE_OPSM}"; then
  GRPO_ARGS+=(
    --use-opsm
    --opsm-delta "${OPSM_DELTA}"
  )
fi

OPTIMIZER_ARGS=(
  --optimizer "${OPTIMIZER}"
  --lr "${LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 "${ADAM_BETA1}"
  --adam-beta2 "${ADAM_BETA2}"
  --clip-grad "${CLIP_GRAD}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRAC}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"
  --megatron-to-hf-mode bridge
)
if is_true "${ENABLE_COLOCATE}"; then
  MISC_ARGS+=(--colocate)
fi
if is_true "${ENABLE_OFFLOAD_ROLLOUT}"; then
  MISC_ARGS+=(--offload-rollout)
fi
if is_true "${NO_ROPE_FUSION}"; then
  MISC_ARGS+=(--no-rope-fusion)
fi
if is_true "${NO_PERSIST_LAYER_NORM}"; then
  MISC_ARGS+=(--no-persist-layer-norm)
fi
if is_true "${NO_GRADIENT_ACCUMULATION_FUSION}"; then
  MISC_ARGS+=(--no-gradient-accumulation-fusion)
fi
if is_true "${NO_MASKED_SOFTMAX_FUSION}"; then
  MISC_ARGS+=(--no-masked-softmax-fusion)
fi
if [[ -n "${CRITIC_NUM_NODES}" ]]; then
  MISC_ARGS+=(--critic-num-nodes "${CRITIC_NUM_NODES}")
fi
if [[ -n "${CRITIC_NUM_GPUS_PER_NODE}" ]]; then
  MISC_ARGS+=(--critic-num-gpus-per-node "${CRITIC_NUM_GPUS_PER_NODE}")
fi

WANDB_ARGS=(
  --use-wandb
  --wandb-mode "${WANDB_MODE}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-dir "${WANDB_DIR}"
  --disable-wandb-random-suffix
)
if [[ -n "${WANDB_API_KEY}" ]]; then
  WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
fi

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_search.generate
  --custom-rm-path generate_with_search.reward_func
)

RUNTIME_ENV_JSON="$(cat <<EOF
{"env_vars":{"PYTHONPATH":"${MEGATRON_DIR}:${SEARCH_R1_DIR}","CUDA_DEVICE_MAX_CONNECTIONS":"1","MASTER_ADDR":"127.0.0.1","CUDA_HOME":"${CUDA_HOME}","FLASHINFER_NVCC":"${CUDA_HOME}/bin/nvcc","PATH":"${PATH}","CPATH":"${CPATH}","CPLUS_INCLUDE_PATH":"${CPLUS_INCLUDE_PATH}","LD_LIBRARY_PATH":"${LD_LIBRARY_PATH:-}","SEARCH_R1_SEARCH_BACKEND":"${SEARCH_BACKEND}","SEARCH_R1_LOCAL_SEARCH_URL":"${LOCAL_SEARCH_URL}","SEARCH_R1_LOCAL_PROXY":"${LOCAL_SEARCH_PROXY}","SEARCH_R1_SERPER_API_KEY":"${SEARCH_R1_SERPER_API_KEY}","SEARCH_R1_RETURN_LOGPROB":"${SEARCH_RETURN_LOGPROB}","SEARCH_R1_SEARCH_CONCURRENCY":"${SEARCH_CONCURRENCY}","SEARCH_R1_MAX_TURNS":"${SEARCH_MAX_TURNS}","SEARCH_R1_TOPK":"${SEARCH_TOPK}","SEARCH_R1_GOOGLE_SNIPPET_ONLY":"${GOOGLE_SNIPPET_ONLY}","SEARCH_R1_GOOGLE_MIN_INTERVAL":"${GOOGLE_MIN_INTERVAL}","SEARCH_R1_GOOGLE_MAX_RETRIES":"${GOOGLE_MAX_RETRIES}","SEARCH_R1_GOOGLE_BACKOFF_BASE":"${GOOGLE_BACKOFF_BASE}","SEARCH_R1_GOOGLE_BACKOFF_CAP":"${GOOGLE_BACKOFF_CAP}","SLIME_DISABLE_PACKED_SEQ":"${SLIME_DISABLE_PACKED_SEQ}","no_proxy":"${no_proxy}","NO_PROXY":"${NO_PROXY}","http_proxy":"${http_proxy}","https_proxy":"${https_proxy}","all_proxy":"${all_proxy}"}}
EOF
)"

set -x
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
  "${WANDB_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" | tee "${LOG_DIR}/train_console.log"

set +x
if [[ "${EXPORT_TEST_TRAJ}" == "1" ]]; then
  if ! [[ "${TEST_EXPORT_EVERY_N_EVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARN: TEST_EXPORT_EVERY_N_EVAL=${TEST_EXPORT_EVERY_N_EVAL} is invalid, fallback to 1"
    TEST_EXPORT_EVERY_N_EVAL=1
  fi

  eval_dir="${DEBUG_DIR}/rollout_data"
  if compgen -G "${eval_dir}/eval_*.pt" > /dev/null; then
    mapfile -t eval_pts < <(find "${eval_dir}" -maxdepth 1 -type f -name 'eval_*.pt' | sort -V)
    if [[ "${TEST_EXPORT_MODE}" == "final" ]]; then
      last_idx=$((${#eval_pts[@]} - 1))
      eval_pts=("${eval_pts[${last_idx}]}")
    elif [[ "${TEST_EXPORT_MODE}" != "all" ]]; then
      echo "WARN: TEST_EXPORT_MODE=${TEST_EXPORT_MODE} is unsupported, fallback to final"
      last_idx=$((${#eval_pts[@]} - 1))
      eval_pts=("${eval_pts[${last_idx}]}")
    fi

    export_idx=0
    for eval_pt in "${eval_pts[@]}"; do
      export_idx=$((export_idx + 1))
      if (((export_idx - 1) % TEST_EXPORT_EVERY_N_EVAL != 0)); then
        continue
      fi
      bn="$(basename "${eval_pt}")"
      rid="${bn#eval_}"
      rid="${rid%.pt}"
      if [[ "${rid}" =~ ^[0-9]+$ ]]; then
        step_id=$((rid + 1))
      else
        step_id="${rid}"
      fi

      out_jsonl="${RUN_ROOT}/test_${step_id}.jsonl"
      out_metrics="${RUN_ROOT}/test_${step_id}_metrics.json"
      echo "Exporting eval trajectories from ${eval_pt} -> ${out_jsonl}"
      python3 "${SEARCH_R1_DIR}/export_test_traj.py" \
        --eval-rollout-pt "${eval_pt}" \
        --output-jsonl "${out_jsonl}" \
        --output-metrics-json "${out_metrics}" | tee -a "${LOG_DIR}/test_export.log"
    done
  else
    echo "No eval_*.pt found under ${eval_dir}, skip test trajectory export."
  fi
fi
