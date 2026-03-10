#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="/mnt/workspace/jkh/slime/examples/compile-r1"
CONFIG_FILE="${BASE_DIR}/configs/grpo_qwen3_8b_raw_ref785td_8gpu.yaml"
TS="$(date +%Y%m%d-%H%M%S)"
EXP_NAME="compile-r1-grpo-qwen3-8b-raw-ref785td-${TS}"
LOG_FILE="${BASE_DIR}/train_log/launch_${EXP_NAME}.log"

echo "[launch] config=${CONFIG_FILE}"
echo "[launch] exp_name=${EXP_NAME}"
echo "[launch] log=${LOG_FILE}"

CONFIG_FILE="${CONFIG_FILE}" \
EXP_NAME="${EXP_NAME}" \
bash "${BASE_DIR}/run_qwen3_4B_grpo_jkh.sh" |& tee "${LOG_FILE}"
