#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/workspace/jkh}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${BASE_DIR}/LLaMA-Factory}"
CONDA_EXE="${CONDA_EXE:-${BASE_DIR}/miniconda3/bin/conda}"
LF_ENV_NAME="${LF_ENV_NAME:-lf}"
LF_ENV_PATH="${LF_ENV_PATH:-${BASE_DIR}/miniconda3/envs/${LF_ENV_NAME}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SFT_CONFIG="${SFT_CONFIG:-${LLAMA_FACTORY_DIR}/examples/train_full/qwen3_8b_full_sft_compile_r1_cold_start_natural.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/slime/examples/compile-r1/train_log/cold_start_sft_qwen3_8b_full_natural}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_console.log}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Conda executable not found: ${CONDA_EXE}" >&2
  exit 1
fi
if [[ ! -f "${SFT_CONFIG}" ]]; then
  echo "SFT config not found: ${SFT_CONFIG}" >&2
  exit 1
fi

set +u
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${LF_ENV_PATH}"
set -u

export CUDA_VISIBLE_DEVICES

cd "${LLAMA_FACTORY_DIR}"
echo "[SFT] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[SFT] config: ${SFT_CONFIG}"
echo "[SFT] output: ${OUTPUT_DIR}"
echo "[SFT] gpus: ${CUDA_VISIBLE_DEVICES}"

llamafactory-cli train "${SFT_CONFIG}" 2>&1 | tee "${LOG_FILE}"
