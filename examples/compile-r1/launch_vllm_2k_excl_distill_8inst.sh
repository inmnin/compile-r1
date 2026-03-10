#!/usr/bin/env bash
set -euo pipefail

source /mnt/workspace/jkh/miniconda3/bin/activate vllm

MODEL="/mnt/workspace/jkh/slime/examples/compile-r1/train_log/cold_start_sft_qwen3_8b_full_cleaned_e1_lr1e5_8g_fast/checkpoint-275"
SHARD_DIR="/mnt/workspace/jkh/slime/examples/compile-r1/data/train_excl_distill_2k/shards"
BASE_LOG_DIR="/mnt/workspace/jkh/slime/examples/compile-r1/train_log"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${BASE_LOG_DIR}/vllm_excl_distill2k_8inst_max7000_${TS}"
mkdir -p "${RUN_DIR}"
CACHE_BASE="/tmp/vllm_cache_${USER:-root}_${TS}"
mkdir -p "${CACHE_BASE}"

cat > "${RUN_DIR}/run_config.txt" <<EOF
MODEL=${MODEL}
SHARD_DIR=${SHARD_DIR}
RUN_DIR=${RUN_DIR}
TP=1
GPU_MEMORY_UTILIZATION=0.68
MAX_MODEL_LEN=12288
BATCH_SIZE=16
MAX_NEW_TOKENS=7000
PROMPT_RULE=ENGLISH_STRICT_CODE_RULE
EOF

echo "[launch] run_dir=${RUN_DIR}"
for g in {0..7}; do
  shard="${SHARD_DIR}/shard_$(printf '%02d' "$g").parquet"
  out="${RUN_DIR}/traj_shard_$(printf '%02d' "$g").jsonl"
  log="${RUN_DIR}/traj_shard_$(printf '%02d' "$g").log"

  # GPU2 has less free memory on this host; use safer settings there.
  util="0.68"
  bs="16"
  if [[ "$g" == "2" ]]; then
    util="0.45"
    bs="8"
  fi

  ti_cache="${CACHE_BASE}/torchinductor_g${g}"
  triton_cache="${CACHE_BASE}/triton_g${g}"
  xdg_cache="${CACHE_BASE}/xdg_g${g}"
  mkdir -p "$ti_cache" "$triton_cache" "$xdg_cache"

  echo "[launch] gpu=${g} shard=${shard}"
  CUDA_VISIBLE_DEVICES="$g" \
  TORCHINDUCTOR_CACHE_DIR="$ti_cache" \
  TRITON_CACHE_DIR="$triton_cache" \
  XDG_CACHE_HOME="$xdg_cache" \
  python -u /mnt/workspace/jkh/slime/examples/compile-r1/vllm_infer_save_trajectories.py \
    --model "$MODEL" \
    --input-glob "$shard" \
    --prompt-field question \
    --id-field id \
    --output-jsonl "$out" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$util" \
    --max-model-len 12288 \
    --batch-size "$bs" \
    --max-new-tokens 7000 \
    2>&1 | tee "$log" &
done
wait
echo "[done] all shards finished."
