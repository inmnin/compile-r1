#!/usr/bin/env bash
set -euo pipefail

SFT_CMD_PATTERN="llamafactory-cli train examples/train_full/qwen3_full_sft_compile_r1_cold_start.yaml"
SFT_OUT_DIR="/mnt/workspace/jkh/slime/examples/compile-r1/train_log/cold_start_sft_qwen3_4b_full"
GRPO_CFG="/mnt/workspace/jkh/slime/examples/compile-r1/configs/grpo_qwen3_4b_coldstart.yaml"
GRPO_SCRIPT="/mnt/workspace/jkh/slime/examples/compile-r1/run_qwen3_4B_grpo_jkh.sh"
SESSION="jkh_sft"
WIN_NAME="compile_grpo"

echo "[orchestrate] waiting for SFT process to exit..."
while pgrep -af "$SFT_CMD_PATTERN" >/dev/null 2>&1; do
  sleep 30
done

echo "[orchestrate] SFT process ended, selecting best checkpoint..."
BEST_MODEL_DIR="$(python - <<'PY'
import json, pathlib, glob, os
out=pathlib.Path('/mnt/workspace/jkh/slime/examples/compile-r1/train_log/cold_start_sft_qwen3_4b_full')
state=out/'trainer_state.json'
best=''
if state.exists():
    try:
        obj=json.loads(state.read_text())
        best=str(obj.get('best_model_checkpoint') or '').strip()
    except Exception:
        pass
if best and pathlib.Path(best).exists():
    print(best)
    raise SystemExit
# fallback to latest checkpoint-* directory
ckpts=sorted(out.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else -1)
if ckpts:
    print(str(ckpts[-1]))
else:
    print(str(out))
PY
)"

echo "[orchestrate] selected model dir: ${BEST_MODEL_DIR}"

python - <<PY
import yaml
cfg_path='${GRPO_CFG}'
with open(cfg_path,'r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f) or {}
cfg['MODEL_DIR'] = '${BEST_MODEL_DIR}'
with open(cfg_path,'w',encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print('updated MODEL_DIR in', cfg_path)
PY

if tmux list-windows -t "${SESSION}" | rg -q "${WIN_NAME}"; then
  tmux kill-window -t "${SESSION}:${WIN_NAME}" || true
fi

echo "[orchestrate] launching GRPO in tmux ${SESSION}:${WIN_NAME}"
tmux new-window -t "${SESSION}" -n "${WIN_NAME}" \
  "bash -lc 'CONFIG_FILE=${GRPO_CFG} bash ${GRPO_SCRIPT}'"

echo "[orchestrate] done"
