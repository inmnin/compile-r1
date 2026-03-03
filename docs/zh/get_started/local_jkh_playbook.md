# slime 本地环境实战手册（jkh 版本）

本文面向你当前这套本地环境：`/mnt/workspace/jkh`。  
目标是把下面四件事打通：

1. 看懂 `slime` 代码结构和关键文件职责。
2. 明确本地部署与官方 Docker 说明的差异。
3. 从 conda/micromamba 环境激活到 agentic RL 参数设置再到训练启动。
4. 接入 W&B 做全程监控（online/offline 都支持）。

---

## 1. 本地版与官方 Docker 版的关键差异

官方文档（例如 `README.md`、`docs/zh/get_started/quick_start.md`）默认路径和依赖通常是 Docker 场景。你本地环境有以下差异：

1. 代码路径是 `/mnt/workspace/jkh`，不是 `/root`。
2. 你是手动安装链路（micromamba + 指定 commit + patch），核心脚本在 `/mnt/workspace/jkh/setup_slime_in_jkh.sh`。
3. 你的 `sglang`、`Megatron-LM`、`slime` 是并列源码目录：
   - `/mnt/workspace/jkh/sglang`
   - `/mnt/workspace/jkh/Megatron-LM`
   - `/mnt/workspace/jkh/slime`
4. 你的环境有一个本地修正：`$CUDA_HOME/lib64` 需要指向 `targets/x86_64-linux/lib`，否则 flashinfer JIT 链接可能找不到 `-lcudart/-lcuda`。

---

## 2. 仓库结构与关键文件职责

下面只列最核心、最常改的目录和文件。

### 2.1 顶层目录（`/mnt/workspace/jkh/slime`）

- `train.py`：主训练入口（同步 rollout/train 循环）。
- `train_async.py`：异步训练入口（rollout 与 train 部分重叠）。
- `scripts/`：官方训练脚本模板、模型结构参数模板。
- `examples/`：agentic/multi-turn/retool/search 等示例。
- `tests/`：端到端与功能测试（你跑通的是 `test_qwen2.5_0.5B_debug_rollout_then_train.py`）。
- `slime/`：框架主代码。
- `slime_plugins/`：插件和扩展（rollout buffer、模型桥接等）。
- `docker/patch/`：给 `sglang`/`Megatron-LM` 打补丁的 patch 集合。

### 2.2 核心代码（`/mnt/workspace/jkh/slime/slime`）

- `slime/utils/arguments.py`：主参数定义入口（Megatron + SGLang + slime 自身参数）。
- `slime/utils/logging_utils.py`：统一 logging / wandb / tensorboard 初始化。
- `slime/utils/wandb_utils.py`：W&B 主/从进程初始化、metric 定义。
- `slime/ray/placement_group.py`：GPU 资源切分、actor/rollout 分配。
- `slime/ray/rollout.py`：`RolloutManager`，负责调 SGLang 采样、数据处理、评估。
- `slime/ray/train_actor.py`：训练 actor 抽象基类（分布式初始化、设备上下文）。
- `slime/backends/megatron_utils/actor.py`：Megatron 后端训练主逻辑（init/train/update/save）。
- `slime/backends/fsdp_utils/actor.py`：FSDP 后端训练主逻辑。
- `slime/rollout/sglang_rollout.py`：默认 rollout 实现。
- `slime/rollout/rm_hub/`：内置 reward 计算逻辑（math/deepscaler/f1/gpqa 等）。

### 2.3 agentic RL 相关目录（高频）

- `examples/multi_agent/`：多 agent 组织与 rollout 示例。
- `examples/retool/`：工具调用与 sandbox 流程。
- `examples/search-r1/`：search 型 agent rollout。
- `examples/true_on_policy/`：严格 on-policy 路径。
- `examples/fully_async/`：全异步训练流程示例。

---

## 3. 训练流程（代码层）

以 `train.py` 为主线，逻辑是：

1. `parse_args()`：解析参数并确定后端（megatron/fsdp）。
2. `create_placement_groups()`：按训练与 rollout 需求切分 GPU。
3. `create_rollout_manager()`：拉起 `RolloutManager` + SGLang 引擎。
4. `create_training_models()`：拉起 actor（可选 critic）训练进程组。
5. 循环：
   - `rollout_manager.generate()` 生成样本；
   - `actor_model.async_train()`（可带 critic）更新；
   - `actor_model.update_weights()` 同步权重到 rollout 引擎；
   - 按间隔执行 `save`/`eval`。
6. 结束时 `rollout_manager.dispose()` 清理资源。

---

## 4. 本地环境启动（从 shell 到 Python）

建议每次新 shell 先执行：

```bash
source ~/.bashrc >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate slime

cd /mnt/workspace/jkh/slime
export PYTHONPATH="/mnt/workspace/jkh/Megatron-LM:${PYTHONPATH:-}"
export CUDA_HOME="${CONDA_PREFIX}"

# 避免 flashinfer JIT 链接找不到 CUDA runtime
if [ ! -e "${CUDA_HOME}/lib64" ] && [ -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]; then
  ln -s "${CUDA_HOME}/targets/x86_64-linux/lib" "${CUDA_HOME}/lib64"
fi
```

可选检查：

```bash
python -V
python -c "import torch,wandb;print(torch.__version__, wandb.__version__)"
```

---

## 5. 数据与模型的本地建议路径

- 模型：`/root/models/Qwen2.5-0.5B-Instruct`
- 数据：`/root/datasets/gsm8k/train.parquet`、`/root/datasets/gsm8k/test.parquet`
- 训练输出：`/mnt/workspace/jkh/tmp/slime_runs/...`
- W&B 本地目录：`/mnt/workspace/jkh/tmp/wandb/...`

---

## 6. agentic RL 最关键的可配置环节

你可以把 agentic 逻辑拆成 4 层来接：

1. rollout 生成层：
   - `--rollout-function-path`：替换整个 rollout 主函数。
   - `--custom-generate-function-path`：只替换请求生成逻辑。
2. reward/后处理层：
   - `--rm-type`（内置）或 `--custom-rm-path`（自定义）。
   - `--custom-reward-post-process-path`：二次处理 reward。
3. 训练数据转换层：
   - `--custom-convert-samples-to-train-data-path`：把样本映射为训练 batch。
4. 评估层：
   - `--eval-function-path`：训练中 eval 的独立逻辑。

多 agent 场景优先看：

- `examples/multi_agent/`
- `examples/retool/`
- `examples/search-r1/`

---

## 7. W&B 接入（本地版）

本仓库本身已内置 W&B，不需要二次开发。核心参数是：

- `--use-wandb`
- `--wandb-mode`（`online` / `offline` / `disabled`）
- `--wandb-project`
- `--wandb-group`
- `--wandb-key`（可选，推荐通过环境变量）
- `--wandb-dir`（建议指定到 workspace）

你可以直接用本手册配套脚本：

- `/mnt/workspace/jkh/slime/scripts/local/run_qwen2.5-0.5B_wandb.sh`

---

## 8. 一键训练脚本（带 W&B）

### 8.1 在线监控（W&B cloud）

```bash
cd /mnt/workspace/jkh/slime

export WANDB_API_KEY="<your_key>"
export WANDB_MODE="online"
export WANDB_PROJECT="slime-local"
export WANDB_GROUP="qwen2.5-0.5b-$(date +%Y%m%d-%H%M%S)"

bash scripts/local/run_qwen2.5-0.5B_wandb.sh
```

### 8.2 离线监控（本地写盘）

```bash
cd /mnt/workspace/jkh/slime

export WANDB_MODE="offline"
export WANDB_PROJECT="slime-local-offline"
export WANDB_GROUP="qwen2.5-0.5b-offline-$(date +%Y%m%d-%H%M%S)"

bash scripts/local/run_qwen2.5-0.5B_wandb.sh
```

离线跑完后可手动同步：

```bash
wandb sync /mnt/workspace/jkh/tmp/wandb
```

---

## 9. 结果查看与排障

### 9.1 训练状态

- Ray dashboard: `http://127.0.0.1:8265`
- 训练日志：命令行输出 + Ray job 日志

### 9.2 W&B 指标

默认会看到：

- `train/*`
- `rollout/*`
- `eval/*`
- `perf/*`

### 9.3 断点续训与产物

- 保存目录来自 `--save`（脚本里默认是 `${OUTPUT_DIR}`）。
- 可恢复训练目录来自 `--load`。

### 9.4 常见本地问题

1. `cannot find -lcudart/-lcuda`：检查 `$CUDA_HOME/lib64` 是否存在并指向 conda CUDA lib。
2. `wandb` 不上报：确认 `--use-wandb` 与 `--wandb-group` 非空，online 模式下确认 key 有效。
3. `ray job submit` 卡住：先 `ray stop --force` 再重启脚本。

---

## 10. 最小闭环验证建议

每次升级依赖后，先跑一个短任务（少量 rollout）确认：

1. rollout 正常产样本；
2. train 正常更新；
3. wandb 持续有 `train/*` 和 `rollout/*`；
4. checkpoint 目录按预期写入。

这一步通过后，再跑长任务或复杂 agentic 逻辑。

