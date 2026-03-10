#!/usr/bin/env bash
set -euo pipefail

source /mnt/workspace/jkh/miniconda3/bin/activate vllm

MODEL="/mnt/workspace/jkh/slime/examples/compile-r1/train_log/cold_start_sft_qwen3_8b_full_cleaned_e1_lr1e5_8g_fast/checkpoint-275"
SHARD_DIR="/mnt/workspace/jkh/slime/examples/compile-r1/data/train_excl_distill_5k/shards"
BASE_LOG_DIR="/mnt/workspace/jkh/slime/examples/compile-r1/train_log"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${BASE_LOG_DIR}/vllm_excl_distill5k_8inst_max7000_${TS}"
mkdir -p "${RUN_DIR}"

# Save run config
cat > "${RUN_DIR}/run_config.txt" <<EOF
MODEL=${MODEL}
SHARD_DIR=${SHARD_DIR}
RUN_DIR=${RUN_DIR}
TP=1
GPU_MEMORY_UTILIZATION=0.70
MAX_MODEL_LEN=12288
BATCH_SIZE=16
MAX_NEW_TOKENS=7000
EOF

echo "[launch] run_dir=${RUN_DIR}"

for g in {0..7}; do
  shard="${SHARD_DIR}/shard_$(printf '%02d' "$g").parquet"
  out="${RUN_DIR}/traj_shard_$(printf '%02d' "$g").jsonl"
  log="${RUN_DIR}/traj_shard_$(printf '%02d' "$g").log"

  if [[ ! -f "$shard" ]]; then
    echo "[error] missing shard: $shard"
    exit 1
  fi

  echo "[launch] gpu=${g} shard=${shard}"
  CUDA_VISIBLE_DEVICES="$g" python -u /mnt/workspace/jkh/slime/examples/compile-r1/vllm_infer_save_trajectories.py \
    --model "$MODEL" \
    --input-glob "$shard" \
    --prompt-field question \
    --id-field id \
    --output-jsonl "$out" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 12288 \
    --batch-size 16 \
    --max-new-tokens 7000 \
    2>&1 | tee "$log" &
done

wait

echo "[done] all shards finished."
