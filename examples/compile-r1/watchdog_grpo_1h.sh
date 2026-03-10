#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/workspace/jkh}"
COMPILE_R1_DIR="${COMPILE_R1_DIR:-${BASE_DIR}/slime/examples/compile-r1}"
RUN_SCRIPT="${RUN_SCRIPT:-${COMPILE_R1_DIR}/run_qwen3_4B_grpo_jkh.sh}"
TMUX_TARGET="${TMUX_TARGET:-jkh_sft:grpo5500_r2}"
BASE_CONFIG="${BASE_CONFIG:-${COMPILE_R1_DIR}/configs/grpo_qwen3_8b_full_8gpu_5500_n4_async_retry2_20260308.yaml}"
EXP_PREFIX="${EXP_PREFIX:-compile-r1-grpo-qwen3-4b-}"

DURATION_SEC="${DURATION_SEC:-3600}"
CHECK_SEC="${CHECK_SEC:-30}"
STALL_SEC="${STALL_SEC:-900}"

WATCHDOG_DIR="${COMPILE_R1_DIR}/train_log/watchdog"
mkdir -p "${WATCHDOG_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
WATCHDOG_LOG="${WATCHDOG_DIR}/watchdog_1h_${TS}.log"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "${msg}" | tee -a "${WATCHDOG_LOG}"
}

strip_ansi() {
  sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g'
}

latest_run_dir() {
  ls -dt "${COMPILE_R1_DIR}/train_log/${EXP_PREFIX}"* 2>/dev/null | head -n 1 || true
}

last_step_from_log() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo ""
    return 0
  fi
  rg -o 'model.py:671 - step [0-9]+' "${log_file}" 2>/dev/null | tail -n 1 | awk '{print $NF}' || true
}

last_step_line() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo ""
    return 0
  fi
  tail -n 500 "${log_file}" | strip_ansi | rg 'model.py:671 - step [0-9]+:' | tail -n 1 || true
}

last_rollout_line() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo ""
    return 0
  fi
  tail -n 500 "${log_file}" | strip_ansi | rg 'rollout.py:917 - perf|data.py:211 - rollout [0-9]+:' | tail -n 1 || true
}

is_train_alive() {
  pgrep -f "python3 train.py .*--custom-generate-function-path generate_with_compile.generate" >/dev/null 2>&1
}

ensure_tmux_target() {
  local session="${TMUX_TARGET%%:*}"
  local window="${TMUX_TARGET#*:}"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    log "session ${session} not found, skip relaunch"
    return 1
  fi
  if ! tmux list-windows -t "${session}" -F '#W' | rg -x "${window}" >/dev/null 2>&1; then
    tmux new-window -t "${session}" -n "${window}"
    log "created tmux window ${TMUX_TARGET}"
  fi
  return 0
}

detect_failure_type() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo "NO_LOG"
    return 0
  fi
  local recent
  recent="$(tail -n 1200 "${log_file}" | strip_ansi)"
  if echo "${recent}" | rg -qi 'CUDA out of memory|OutOfMemoryError|Not enough memory|memory capacity is unbalanced|OOM'; then
    echo "OOM"
    return 0
  fi
  if echo "${recent}" | rg -qi '503 Service Unavailable|all circuits open|no_available_workers|router.*retry'; then
    echo "SERVICE_503"
    return 0
  fi
  if echo "${recent}" | rg -qi 'PENDING_NODE_ASSIGNMENT|SGLangEngine.init RUNNING|ActorDiedError|RuntimeError'; then
    echo "SCHED_OR_ACTOR"
    return 0
  fi
  echo "STALLED_OR_EXITED"
}

write_diagnosis() {
  local run_dir="$1"
  local log_file="$2"
  local reason="$3"
  local diag_file="${WATCHDOG_DIR}/diag_${reason}_$(date +%Y%m%d-%H%M%S).log"
  {
    echo "reason=${reason}"
    echo "run_dir=${run_dir}"
    echo "log_file=${log_file}"
    echo "time=$(date '+%F %T')"
    echo
    echo "=== latest_step_line ==="
    last_step_line "${log_file}"
    echo
    echo "=== latest_rollout_line ==="
    last_rollout_line "${log_file}"
    echo
    echo "=== recent_errors ==="
    if [[ -f "${log_file}" ]]; then
      tail -n 1600 "${log_file}" | strip_ansi | rg -n 'ERROR|Exception|Traceback|SIGTERM|OOM|memory|503|no_available_workers|circuit|PENDING_NODE_ASSIGNMENT' || true
    fi
    echo
    echo "=== nvidia-smi ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo
    echo "=== train_processes ==="
    ps -eo pid,etimes,cmd | rg 'train.py|run_qwen3_4B_grpo_jkh.sh|sglang::scheduler|sglang::detokenizer' -n || true
  } > "${diag_file}"
  log "diagnosis written: ${diag_file}"
}

prepare_restart_config() {
  local base_cfg="$1"
  local reason="$2"
  local out_cfg="${COMPILE_R1_DIR}/configs/watchdog_restart_${reason}_$(date +%Y%m%d-%H%M%S).yaml"
  cp "${base_cfg}" "${out_cfg}"
  python - "${out_cfg}" "${reason}" <<'PY'
import copy
import random
import sys
from datetime import datetime
import yaml

cfg_path = sys.argv[1]
reason = sys.argv[2]

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def as_int(k, d):
    try:
        return int(float(cfg.get(k, d)))
    except Exception:
        return int(d)

def as_float(k, d):
    try:
        return float(cfg.get(k, d))
    except Exception:
        return float(d)

if reason == "OOM":
    rb = as_int("ROLLOUT_BATCH_SIZE", 8)
    gb = as_int("GLOBAL_BATCH_SIZE", 32)
    mt = as_int("MAX_TOKENS_PER_GPU", 6144)
    frac = as_float("SGLANG_MEM_FRAC", 0.42)
    margin = as_int("TRAIN_MEMORY_MARGIN_BYTES", 6442450944)
    cfg["ROLLOUT_BATCH_SIZE"] = max(4, rb // 2)
    cfg["GLOBAL_BATCH_SIZE"] = max(16, gb // 2)
    cfg["MAX_TOKENS_PER_GPU"] = max(4608, mt - 1024)
    cfg["SGLANG_MEM_FRAC"] = max(0.34, round(frac - 0.04, 3))
    cfg["TRAIN_MEMORY_MARGIN_BYTES"] = margin + 2147483648
    cfg["COMPILE_R1_TOOL_CONCURRENCY"] = max(12, as_int("COMPILE_R1_TOOL_CONCURRENCY", 24) // 2)
    cfg["COMPILE_R1_ASYNC_TOOL_WORKERS"] = max(12, as_int("COMPILE_R1_ASYNC_TOOL_WORKERS", 24) // 2)
elif reason == "SERVICE_503":
    cfg["COMPILE_R1_TOOL_CONCURRENCY"] = max(12, min(20, as_int("COMPILE_R1_TOOL_CONCURRENCY", 24)))
    cfg["COMPILE_R1_ASYNC_TOOL_WORKERS"] = max(12, min(20, as_int("COMPILE_R1_ASYNC_TOOL_WORKERS", 24)))
    extra = str(cfg.get("EXTRA_TRAIN_ARGS", "")).strip()
    required = [
        "--router-disable-health-check",
        "--router-disable-circuit-breaker",
        "--router-retry-max-retries 20",
        "--router-queue-size 512",
        "--router-max-concurrent-requests 2048",
    ]
    for arg in required:
        if arg not in extra:
            extra = (extra + " " + arg).strip()
    cfg["EXTRA_TRAIN_ARGS"] = f"\"{extra}\""
else:
    cfg["COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER"] = max(8, as_int("COMPILE_R1_ASYNC_TOOL_TIMEOUT_BUFFER", 8))

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
cfg["EXP_NAME"] = f"compile-r1-grpo-qwen3-4b-watchdog-{stamp}"
cfg["FORCE_DIRECT_RUN"] = 1
cfg["DIRECT_RUN_ON_SUBMIT_FAIL"] = 1
cfg["RAY_HEAD_PORT"] = 26000 + random.randint(0, 799)
cfg["RAY_DASHBOARD_PORT"] = 28000 + random.randint(0, 799)
cfg["SLIME_ROLLOUT_START_PORT"] = 17100 + random.randint(0, 599)

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
PY
  echo "${out_cfg}"
}

relaunch_training() {
  local cfg="$1"
  local launch_log="${COMPILE_R1_DIR}/train_log/launch_watchdog_restart_$(date +%Y%m%d-%H%M%S).log"
  ensure_tmux_target || return 1
  local cmd="cd ${COMPILE_R1_DIR} && export CONFIG_FILE=${cfg} && bash run_qwen3_4B_grpo_jkh.sh 2>&1 | tee ${launch_log}"
  tmux send-keys -t "${TMUX_TARGET}" C-c
  sleep 2
  tmux send-keys -t "${TMUX_TARGET}" "${cmd}" C-m
  log "relaunch sent to ${TMUX_TARGET}, config=${cfg}, launch_log=${launch_log}"
  return 0
}

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  log "missing run script: ${RUN_SCRIPT}"
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  log "missing base config: ${BASE_CONFIG}"
  exit 1
fi

log "watchdog start, duration=${DURATION_SEC}s, check=${CHECK_SEC}s, stall=${STALL_SEC}s"
log "tmux_target=${TMUX_TARGET}, base_config=${BASE_CONFIG}"
log "watchdog_log=${WATCHDOG_LOG}"

current_config="${BASE_CONFIG}"
last_step=""
last_progress_ts="$(date +%s)"
end_ts="$(( $(date +%s) + DURATION_SEC ))"
restarts=0

while (( $(date +%s) < end_ts )); do
  run_dir="$(latest_run_dir)"
  if [[ -z "${run_dir}" ]]; then
    log "no run dir found for prefix=${EXP_PREFIX}, waiting..."
    sleep "${CHECK_SEC}"
    continue
  fi
  log_file="${run_dir}/logs/train_console.log"
  step="$(last_step_from_log "${log_file}")"
  now_ts="$(date +%s)"
  alive=0
  if is_train_alive; then
    alive=1
  fi

  if [[ -n "${step}" ]]; then
    if [[ "${step}" != "${last_step}" ]]; then
      last_step="${step}"
      last_progress_ts="${now_ts}"
    fi
  elif [[ -f "${log_file}" ]]; then
    log_mtime="$(stat -c %Y "${log_file}" 2>/dev/null || echo 0)"
    if (( now_ts - log_mtime <= STALL_SEC )); then
      last_progress_ts="${now_ts}"
    fi
  fi

  idle_sec="$(( now_ts - last_progress_ts ))"
  step_line="$(last_step_line "${log_file}")"
  rollout_line="$(last_rollout_line "${log_file}")"
  log "status run=$(basename "${run_dir}") alive=${alive} step=${step:-NA} idle=${idle_sec}s"
  if [[ -n "${step_line}" ]]; then
    log "step_line ${step_line}"
  fi
  if [[ -n "${rollout_line}" ]]; then
    log "rollout_line ${rollout_line}"
  fi

  if (( alive == 0 || idle_sec > STALL_SEC )); then
    reason="$(detect_failure_type "${log_file}")"
    log "failure detected: alive=${alive}, idle=${idle_sec}s, reason=${reason}"
    write_diagnosis "${run_dir}" "${log_file}" "${reason}"
    new_cfg="$(prepare_restart_config "${current_config}" "${reason}")"
    current_config="${new_cfg}"
    relaunch_training "${new_cfg}" || log "relaunch failed to send"
    restarts="$((restarts + 1))"
    last_step=""
    last_progress_ts="$(date +%s)"
    sleep 60
    continue
  fi

  sleep "${CHECK_SEC}"
done

log "watchdog finished: duration reached, restarts=${restarts}"
