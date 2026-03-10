#!/usr/bin/env bash
set -euo pipefail

while pgrep -f "llamafactory-cli train examples/train_full/qwen3_8b_full_sft_compile_r1_cold_start_result_masked_val300.yaml" >/dev/null; do
  date -u +"%Y-%m-%dT%H:%M:%SZ waiting_sft"
  sleep 60
done

source /mnt/workspace/jkh/miniconda3/etc/profile.d/conda.sh
conda activate lf
cd /mnt/workspace/jkh/slime/examples/compile-r1

CUDA_VISIBLE_DEVICES=0 python eval_sft_ckpt_rollout_passrate.py \
  --ckpt-root /mnt/workspace/jkh/slime/examples/compile-r1/train_log/cold_start_sft_qwen3_8b_full_result_masked_val300 \
  --val-jsonl /mnt/workspace/jkh/LLaMA-Factory/data/compile_r1_cold_start_val300_result_masked_alpaca \
  --source-train-glob '/mnt/workspace/jkh/slime/examples/compile-r1/data/train/data/*.parquet' \
  --output-json /mnt/workspace/jkh/slime/examples/compile-r1/train_log/cold_start_sft_qwen3_8b_full_result_masked_val300/eval_rollout_val300_result_masked.json \
  --temperature 0.2 \
  --top-p 0.95 \
  --max-new-tokens 7000 \
  --max-turns 8 \
  --tp-size 1 \
  --gpu-memory-utilization 0.9 \
  --timeout-seconds 10 \
  --memory-mb 1024
