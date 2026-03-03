# Search-R1 (Google Search + GRPO) 复现实验操作日志

## 1. 最终成功实验信息
- 成功实验名: `search-r1-full-20260222-200512`
- Ray job id: `raysubmit_xLdeuYqWhgxJfKpm`
- 运行位置: `tmux` session/window `jkh:jkh`
- 最终状态: `SUCCEEDED`（tmux 尾日志显示 `Job 'raysubmit_xLdeuYqWhgxJfKpm' succeeded`）
- W&B 项目: `search-r1`（实际平台显示项目路径为 `Search-R1`）
- W&B run: `https://wandb.ai/1789979647-peking-university/Search-R1/runs/4jg3nl1p`

## 2. Google Search 启动方式（不使用本地检索）
本实验没有启动本地检索服务；工具调用直接走 `google_search_server.py`（Serper Google API），并在训练 runtime env 中强制：
- `SEARCH_R1_SEARCH_BACKEND=google`
- `SEARCH_R1_SERPER_API_KEY=0ec804b592087df08dc3a0d98ac5cbb67ad549db`
- `SEARCH_R1_SEARCH_CONCURRENCY=2`
- `SEARCH_R1_GOOGLE_MIN_INTERVAL=1.0`
- `SEARCH_R1_GOOGLE_MAX_RETRIES=12`
- `SEARCH_R1_GOOGLE_BACKOFF_BASE=1.5`
- `SEARCH_R1_GOOGLE_BACKOFF_CAP=30`

关键代码路径:
- `examples/search-r1/generate_with_search.py`（搜索后端选择、工具调用、多轮 `<search>/<answer>`）
- `examples/search-r1/google_search_server.py`（异步 HTTP 请求、429/5xx 退避重试、节流）

## 3. 网络稳定设置（按你的要求定时 export）
所有关键命令均显式带：
- `export http_proxy=http://127.0.0.1:7898`
- `export https_proxy=http://127.0.0.1:7898`
- `export all_proxy=http://127.0.0.1:7898`

并在训练脚本里加入后台刷新循环（默认每 300s）：
- `PROXY_REFRESH_SECS=300`

## 4. 数据准备（3% 测试集）
使用脚本:
- `examples/search-r1/prepare_search_r1_data.py`

可复现命令:
```bash
/root/micromamba/envs/slime/bin/python /mnt/workspace/jkh/slime/examples/search-r1/prepare_search_r1_data.py \
  --input-parquet /mnt/workspace/jkh/slime/examples/search-r1/data/train.parquet \
  --output-train /mnt/workspace/jkh/slime/examples/search-r1/data/train_97.parquet \
  --output-test /mnt/workspace/jkh/slime/examples/search-r1/data/test_3.parquet \
  --test-ratio 0.03 \
  --seed 20260222
```

本次训练实际使用:
- `train_97.parquet` 行数: `11640`
- `test_3.parquet` 行数: `360`

## 5. 全量训练启动命令（tmux 内执行）
```bash
cd /mnt/workspace/jkh/slime/examples/search-r1 && \
export http_proxy=http://127.0.0.1:7898 https_proxy=http://127.0.0.1:7898 all_proxy=http://127.0.0.1:7898 && \
WANDB_KEY=e5780f752ed310ff6307 \
SEARCH_R1_SERPER_API_KEY=0ec804b592087df08dc3a0d98ac5cbb67ad549db \
EXP_NAME=search-r1-full-20260222-200512 \
NUM_GPUS=2 ROLLOUT_NUM_GPUS=2 ROLLOUT_NUM_GPUS_PER_ENGINE=1 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
NUM_ROLLOUT=30 SAVE_INTERVAL=10 EVAL_INTERVAL=15 \
ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=4 \
MAX_TOKENS_PER_GPU=2048 SGLANG_MEM_FRAC=0.12 \
SEARCH_CONCURRENCY=2 GOOGLE_MIN_INTERVAL=1.0 GOOGLE_MAX_RETRIES=12 \
bash ./run_qwen2.5_3B_google_grpo_jkh.sh
```

说明:
- 你的 `wandbkey` 长度不满足 SDK 的 key 校验，脚本会忽略该显式 key 并使用机器已登录账号（日志有告警）。
- 为避免显存冲突，最终采用 `2 GPU` 保守配置（8*A100 有他人占用）。

## 5.1 排障与修复记录（本次实际操作）
- 问题: Google API 在 eval 阶段出现 `429 Too Many Requests`，导致作业中断。
  - 修复: 在 `examples/search-r1/google_search_server.py` 增加限流与重试（`throttle_once`、`Retry-After`、指数退避）。
  - 修复: 在 `examples/search-r1/generate_with_search.py` 的 `execute_predictions` 增加工具调用异常兜底，避免单次搜索异常直接杀掉训练。
- 问题: `generate_with_search` 在 Ray actor 中偶发 `ModuleNotFoundError`。
  - 修复: 在 `run_qwen2.5_3B_google_grpo_jkh.sh` 中将 runtime `PYTHONPATH` 固定包含 `.../slime/examples/search-r1`。
- 问题: 脚本相对路径在不同启动上下文下失败（`qwen2.5-3B.sh`）。
  - 修复: 改为绝对引用 `${SLIME_DIR}/scripts/models/qwen2.5-3B.sh`。
- 问题: 作业工作目录错误时会找不到 `train.py`。
  - 修复: 训练执行目录固定为 `${SLIME_DIR}`，同时保留 Search-R1 路径在 `PYTHONPATH` 中。

## 6. 训练参数（最终成功 run_config）
来源: `train_log/search-r1-full-20260222-200512/run_config.env`
- `cuda_visible_devices=7,6`
- `num_gpus=2`
- `rollout_num_gpus=2`
- `rollout_num_gpus_per_engine=1`
- `tensor_model_parallel_size=2`
- `num_rollout=30`
- `rollout_batch_size=2`
- `n_samples_per_prompt=2`
- `global_batch_size=4`
- `rollout_max_response_len=384`
- `search_concurrency=2`
- `google_min_interval=1.0`
- `google_max_retries=12`
- `google_backoff_base=1.5`
- `google_backoff_cap=30`
- `eval_interval=15`
- `save_interval=10`
- `max_tokens_per_gpu=2048`
- `sglang_mem_frac=0.12`
- `wandb_project=search-r1`
- `wandb_group=search-r1-full-20260222-200512`
- `train_parquet=/mnt/workspace/jkh/slime/examples/search-r1/data/train_97.parquet`
- `test_parquet=/mnt/workspace/jkh/slime/examples/search-r1/data/test_3.parquet`

## 7. ckpt 与验证可用性
- checkpoint 正常保存:
  - `ckpt/iter_0000009`
  - `ckpt/iter_0000019`
  - `ckpt/iter_0000029`
- eval 正常执行并落盘:
  - `debug/rollout_data/eval_14.pt`
  - `debug/rollout_data/eval_29.pt`

## 8. 训练过程 GRPO 关键指标（已记录到 W&B）
W&B/日志中可见并持续记录的关键指标包括:
- `train/loss`
- `train/pg_loss`
- `train/entropy_loss`
- `train/pg_clipfrac`
- `train/ppo_kl`
- `train/train_rollout_logprob_abs_diff`
- `train/grad_norm`
- `rollout/rewards`
- `rollout/raw_reward`
- `rollout/response_lengths`
- `rollout/truncated`
- `rollout/log_probs`
- `rollout/rollout_log_probs`
- `rollout/advantages`
- `rollout/returns`
- 各类 `perf/*` 指标（tokens/s, step time 等）

训练统计汇总文件:
- `train_log/search-r1-full-20260222-200512/train_metric_summary.json`

该文件中的关键统计（30 steps, step 0~29）:
- `train/entropy_loss` mean `1.8668`
- `train/loss` mean `0.0`
- `train/pg_loss` mean `0.0`
- `rollout/rewards` mean `0.0`
- `rollout/raw_reward` mean `0.01667`
- `rollout/response_lengths` mean `450.64`

## 9. 测试集统计与轨迹导出
最终测试轨迹导出命令:
```bash
/root/micromamba/envs/slime/bin/python /mnt/workspace/jkh/slime/examples/search-r1/export_test_traj.py \
  --eval-rollout-pt /mnt/workspace/jkh/slime/examples/search-r1/train_log/search-r1-full-20260222-200512/debug/rollout_data/eval_29.pt \
  --output-jsonl /mnt/workspace/jkh/slime/examples/search-r1/train_log/search-r1-full-20260222-200512/test_30.jsonl \
  --output-metrics-json /mnt/workspace/jkh/slime/examples/search-r1/train_log/search-r1-full-20260222-200512/test_30_metrics.json
```

按你要求生成文件:
- `train_log/search-r1-full-20260222-200512/test_30.jsonl`
  - 每行字段包含: `{question, traj, reward, acc, loss, entropy, response_length, ...}`
  - 行数: `360`
- `train_log/search-r1-full-20260222-200512/test_30_metrics.json`
  - `reward_mean=0.06666666666666667`
  - `acc_mean=0.06666666666666667`
  - `loss_mean=0.3620768196940284`
  - `entropy_mean=0.3620768196940284`
  - `response_length_mean=333.425`
  - `search_calls_mean=1.7222222222222223`

## 10. Slime 异步 rollout 采样策略（含工具调用）
核心机制（对应代码）:
- `slime/rollout/sglang_rollout.py`
  - `generate_rollout_async(...)`: 构造并发任务，异步驱动一批 sample rollout。
  - `generate_and_rm_group(...)`: 对一个 group 用 `asyncio.gather(...)` 并发生成与打分。
  - `generate_and_rm(...)`: 每条样本调用生成函数；若配置了 `custom_generate_function_path`，会动态加载并 `await` 自定义生成函数。
  - `eval_rollout_single_dataset(...)`: eval 阶段用 `asyncio.as_completed(tasks)` 按“先完成先处理”收集结果，避免慢样本拖住全部结果。
- `examples/search-r1/generate_with_search.py`
  - `generate(...)`: 多轮交互，模型生成 `<search>` / `<answer>` action。
  - `execute_predictions(...)`: action 为 search 时触发工具调用；用 `SEMAPHORE` 控制同进程并发搜索。
- `examples/search-r1/google_search_server.py`
  - `google_search(...)`: `aiohttp` 异步请求 Google Serper API。
  - 通过 `throttle_once` + `max_retries` + 指数退避 + `Retry-After` 处理 429/5xx，减小工具抖动导致的 rollout 中断。

一句话总结:
- Slime 在 rollout 层做“样本并发 + 结果异步回收”，在工具层做“异步 I/O + 限流重试”，两层叠加后能把搜索工具调用延迟隐藏在并发流水中。

## 11. API credit 估算（是否够用）
本次**成功全量训练**实测工具调用统计见:
- `train_log/search-r1-full-20260222-200512/search_api_usage_summary.json`

实测值:
- 训练 rollout（30轮）: `171` 次搜索调用
- eval_14: `628` 次
- eval_29: `620` 次
- 合计观测: `1419` 次

若按 `1 request = 1 credit` 估算:
- 单次完整训练大约消耗 `~1419` credits（不含失败重试额外损耗）
- 考虑网络抖动/重试，建议按 `1800~2200` credits 预算一轮
- 你提供的 `2500 credits` 对**一轮完整训练**是够的
- 若要预留多次失败重跑，建议准备 `>=4000` 更稳妥

## 12. 本次关键产物路径
- 总日志: `train_log/search-r1-full-20260222-200512/logs/train_console.log`
- 训练参数快照: `train_log/search-r1-full-20260222-200512/run_config.env`
- ckpt: `train_log/search-r1-full-20260222-200512/ckpt/`
- 测试轨迹: `train_log/search-r1-full-20260222-200512/test_30.jsonl`
- 测试统计: `train_log/search-r1-full-20260222-200512/test_30_metrics.json`
- 训练指标汇总: `train_log/search-r1-full-20260222-200512/train_metric_summary.json`
- 搜索调用统计: `train_log/search-r1-full-20260222-200512/search_api_usage_summary.json`
