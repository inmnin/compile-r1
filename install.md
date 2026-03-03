# slime 安装与跨设备运行训练脚本说明（jkh 环境）

本文目标：把 `/mnt/workspace/jkh/slime` 在另一台机器上完整跑起来，重点覆盖 `examples/search-r1` 的训练脚本（GRPO/PPO）。

## 1. 适用范围与目录约定

本文按以下目录布局说明（与当前机器一致）：

```bash
/mnt/workspace/jkh/
├── slime
├── sglang
└── Megatron-LM
```

如果你使用其它路径，请把命令中的绝对路径替换成你的实际路径。

## 2. 硬件与系统前置要求

1. Linux（推荐 Ubuntu 20.04/22.04）。
2. NVIDIA GPU（A100/H100/B200 等），建议至少 2 张卡用于 Search-R1 训练闭环验证。
3. 驱动与 CUDA 运行时可用（`nvidia-smi` 正常）。
4. 外网访问能力（下载依赖、调用 W&B、Google Serper API）。

## 3. 一次性安装依赖（推荐用现成脚本）

项目已提供本地环境安装脚本：`/mnt/workspace/jkh/setup_slime_in_jkh.sh`，包含：

1. `micromamba` 安装与 `slime` 环境创建（Python 3.12）。
2. CUDA/NCCL/cuDNN 组件安装。
3. `torch==2.9.1`（cu129）安装。
4. `sglang` / `Megatron-LM` 克隆与指定 commit 切换。
5. `slime` 可编辑安装与 patch 应用。

直接执行：

```bash
cd /mnt/workspace/jkh
bash setup_slime_in_jkh.sh
```

## 4. 每次新 shell 的环境激活

```bash
source ~/.bashrc >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate slime

export BASE_DIR=/mnt/workspace/jkh
export SLIME_DIR=$BASE_DIR/slime
export MEGATRON_DIR=$BASE_DIR/Megatron-LM
export PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR/examples/search-r1:${PYTHONPATH:-}"
cd "$SLIME_DIR"
```

可选快速检查：

```bash
python -V
python -c "import torch, ray, wandb; print(torch.__version__)"
nvidia-smi
```

## 5. Search-R1 训练所需数据与模型

### 5.1 模型目录

`examples/search-r1/run_qwen2.5_3B_google_grpo_jkh.sh` 默认读取：

- `MODEL_DIR=/mnt/workspace/jkh/model/Qwen3-4B`（可在 config 覆盖）

请确保该目录存在可用 HF 权重。

### 5.2 训练/测试数据

脚本默认读取：

- `TRAIN_PARQUET=/mnt/workspace/jkh/slime/examples/search-r1/data/train_97.parquet`
- `TEST_PARQUET=/mnt/workspace/jkh/slime/examples/search-r1/data/test_3.parquet`

若只有一个原始 parquet，可用以下命令切分：

```bash
python /mnt/workspace/jkh/slime/examples/search-r1/prepare_search_r1_data.py \
  --input-parquet /mnt/workspace/jkh/slime/examples/search-r1/data/train.parquet \
  --output-train /mnt/workspace/jkh/slime/examples/search-r1/data/train_97.parquet \
  --output-test /mnt/workspace/jkh/slime/examples/search-r1/data/test_3.parquet \
  --test-ratio 0.03 \
  --seed 20260222
```

## 6. 训练配置文件与入口

主要配置文件在：

- `examples/search-r1/configs/full_train.yaml`（Google + GRPO）
- `examples/search-r1/configs/full_train_local.yaml`（Local search + GRPO）
- `examples/search-r1/configs/ppo_train.yaml`（Google + PPO）
- `examples/search-r1/configs/precheck_local.yaml`（本地预跑）

主要启动脚本：

- `examples/search-r1/run_qwen2.5_3B_google_grpo_jkh.sh`
- `examples/search-r1/run_qwen2.5_3B_google_ppo_jkh.sh`

## 7. Google 搜索后端运行（推荐先跑）

1. 修改 `examples/search-r1/configs/full_train.yaml`：
   - `SEARCH_R1_SERPER_API_KEY` 填你的 serper key。
2. 启动训练：

```bash
cd /mnt/workspace/jkh/slime/examples/search-r1
CONFIG_FILE=/mnt/workspace/jkh/slime/examples/search-r1/configs/full_train.yaml \
bash /mnt/workspace/jkh/slime/examples/search-r1/run_qwen2.5_3B_google_grpo_jkh.sh
```

脚本行为（已内置）：

1. 自动选卡并校验显存阈值。
2. 启动 Ray head（127.0.0.1:8265）。
3. 提交 `ray job submit ... python3 train.py`。
4. 训练结束后自动导出 `test_*.jsonl`（取决于 `EXPORT_TEST_TRAJ` 配置）。

## 8. tmux 长任务运行（推荐）

```bash
tmux new -s jkh
cd /mnt/workspace/jkh/slime/examples/search-r1
CONFIG_FILE=/mnt/workspace/jkh/slime/examples/search-r1/configs/full_train.yaml \
bash ./run_qwen2.5_3B_google_grpo_jkh.sh
```

常用操作：

- 分离：`Ctrl+b` 后按 `d`
- 重新连接：`tmux attach -t jkh`

## 9. 断点续训机制

该脚本会把输出写到：

- `examples/search-r1/train_log/<EXP_NAME>/ckpt`

当检测到 `latest_checkpointed_iteration.txt` 时，会自动追加 `--load <ckpt_dir>` 继续训练。

## 10. 本地检索后端（可选）

如果改用 local search：

1. 启动 `local_dense_retriever/retrieval_server.py`。
2. 使用 `CONFIG_FILE=.../full_train_local.yaml`。
3. 确保 `LOCAL_SEARCH_URL` 可访问（默认 `http://127.0.0.1:8000/retrieve`）。

## 11. 结果与日志位置

每次训练会在 `examples/search-r1/train_log/<EXP_NAME>/` 生成：

1. `run_config.env`（本次参数快照）
2. `logs/train_console.log`（控制台日志）
3. `ckpt/`（checkpoint）
4. `debug/rollout_data/eval_*.pt`（评估 rollout）
5. `test_*.jsonl` 与 `test_*_metrics.json`（导出轨迹与统计）

## 12. 常见问题排查

1. 报错 `Missing MODEL_DIR`：
   - 检查 config 中 `MODEL_DIR` 是否存在。
2. 报错 `SEARCH_R1_SERPER_API_KEY is required`：
   - Google 后端必须配置该 key。
3. OOM：
   - 降低 `MAX_TOKENS_PER_GPU`、`ROLLOUT_BATCH_SIZE`、`N_SAMPLES_PER_PROMPT`。
4. 搜索 429：
   - 降低 `SEARCH_CONCURRENCY`，提高 `GOOGLE_MIN_INTERVAL`，保留重试参数。
5. FlashInfer/CUDA 链接问题：
   - 脚本已包含 `CUDA_HOME/lib64 -> targets/x86_64-linux/lib` 的兼容逻辑。

## 13. 跨设备复现最小清单

迁移到新机器时，至少同步并确认以下内容：

1. `slime`、`sglang`、`Megatron-LM` 三个代码目录。
2. `slime/examples/search-r1/configs/*.yaml` 与运行脚本。
3. `model` 目录下目标模型权重。
4. `examples/search-r1/data/train_97.parquet`、`test_3.parquet`（或可重新切分生成）。
5. API key（Serper/W&B）通过环境变量或配置注入，不建议硬编码到公开仓库。
