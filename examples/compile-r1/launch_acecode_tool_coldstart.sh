#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="${SESSION_NAME:-jkh_sft}"
WINDOW_NAME="${WINDOW_NAME:-acecode_coldstart_v2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
API_KEY="${JUDGE_SET_3_API_KEY:-${JUDGE_SET_2_API_KEY:-}}"
BASE_URL="${JUDGE_SET_3_BASE_URL:-${JUDGE_SET_2_BASE_URL:-https://api.deepseek.com/v1}}"
MODEL_NAME="${JUDGE_SET_3_MODEL_NAME:-${JUDGE_SET_2_MODEL_NAME:-deepseek-chat}}"

if [[ -z "${API_KEY}" ]]; then
  echo "Missing DeepSeek API key in JUDGE_SET_3_API_KEY or JUDGE_SET_2_API_KEY" >&2
  exit 1
fi

LOG_DIR="${ROOT_DIR}/examples/compile-r1/train_log/acecode_tool_coldstart_v2"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/distill_${STAMP}.log"

CMD=(
  "${PYTHON_BIN}" "examples/compile-r1/distill_acecode_tool_coldstart.py"
  "--data-root" "examples/compile-r1/data"
  "--output-dir" "examples/compile-r1/data/cold_start_v2"
  "--output-prefix" "acecode_tool_coldstart"
  "--sample-concurrency" "24"
  "--runpy-max-workers" "24"
  "--runpy-max-inflight" "96"
  "--runpy-worker-max-tasks" "400"
  "--auto-start-compile-server"
  "--save-failures"
  "--shuffle"
  "--base-url" "${BASE_URL}"
  "--model-name" "${MODEL_NAME}"
  "--api-key" "${API_KEY}"
)

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux new-window -t "${SESSION_NAME}" -n "${WINDOW_NAME}" "cd '${ROOT_DIR}' && ${CMD[*]} | tee '${LOG_FILE}'"
else
  tmux new-session -d -s "${SESSION_NAME}" -n "${WINDOW_NAME}" "cd '${ROOT_DIR}' && ${CMD[*]} | tee '${LOG_FILE}'"
fi

echo "${LOG_FILE}"
