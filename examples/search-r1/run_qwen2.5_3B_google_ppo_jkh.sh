#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/configs/ppo_train.yaml"

# PPO 默认入口：仍复用原始 Search-R1 rollout 与工具调用逻辑。
export CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-ppo}"

# critic 相关默认值（可在 YAML 覆盖）
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-0}"
export VALUE_CLIP="${VALUE_CLIP:-0.2}"

if [[ -z "${CRITIC_SAVE:-}" ]]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  export EXP_NAME="${EXP_NAME:-search-r1-ppo-google-${ts}}"
  # 仅在未显式设置 CRITIC_SAVE 时，给一个独立目录，避免与 actor ckpt 混淆。
  export CRITIC_SAVE="${SCRIPT_DIR}/train_log/${EXP_NAME}/ckpt_critic"
fi

exec "${SCRIPT_DIR}/run_qwen2.5_3B_google_grpo_jkh.sh" "$@"
