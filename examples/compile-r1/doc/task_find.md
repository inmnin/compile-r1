0. 目标

用 Qwen/Qwen3-4B 先跑 no-tool baseline，再用 teacher + python_exec 跑 tool-enabled，只保留：

no-tool 错
tool-enabled 对
题目确实属于 数值 / 表格 / scale / 组合推理 难点

这样做出来的训练集，才会天然形成“没工具明显差、会用工具明显涨”的信号。Qwen3-4B在 /mnt/workspace/jkh/model/Qwen3-4B

1. 数据集与下载链接
训练候选集
TabMWP：表格数学题，官方仓库在 PromptPG，数据在 data/tabmwp；也有 Hugging Face 镜像。
FinQA：金融数值推理数据集，官方仓库。
TAT-QA：表格+文本混合金融问答，官方仓库 / Hugging Face 镜像。
外部主评测集
TabularGSM：优先用 hard 和 robustness 两个子集，Hugging Face 数据集页直接提供这两个子集。
可选迁移评测
MATH：只做迁移观察，不做主筛题标准；官方仓库和 Hugging Face 都能下。
2. 下载到本地的建议目录
mkdir -p data/raw data/normalized data/filtered data/final logs

# TabMWP
git clone https://github.com/lupantech/PromptPG.git external/PromptPG

# FinQA
git clone https://github.com/czyssrs/FinQA.git external/FinQA

# TAT-QA
git clone https://github.com/NExTplusplus/TAT-QA.git external/TAT-QA

# TabularGSM / MATH 建议直接用 datasets 下载
python - <<'PY'
from datasets import load_dataset
load_dataset("kevin715/TabularGSM")
load_dataset("EleutherAI/hendrycks_math")
PY
3. 统一成一个内部格式

先把所有原始数据转成统一 JSONL。统一字段只保留这些：

{
  "id": "dataset_name__sample_id",
  "dataset": "tabmwp | finqa | tatqa | tabulargsm | math",
  "split": "train | dev | test",
  "question": "...",
  "table": "...",       
  "context": "...",     
  "answer": "...",      
  "meta": {
    "source_path": "...",
    "is_numeric": true,
    "has_table": true,
    "has_text_context": false,
    "candidate_hard_reason": ["scale", "multi_step", "hybrid"]
  }
}

规则：

question 必填
answer 必填
table 没有就设 ""
context 没有就设 ""
不依赖原始字段名；Codex 直接对每个数据集写单独 adapter
4. 先做“候选难题”预筛，不要全跑 teacher（deepseek api）
TabMWP：只保留
数值答案
不是纯选择题
需要表格，不是纯文字
题目看起来至少要 2 步运算
FinQA：优先保留
明显是数值题
涉及比例、百分比、变化量、除法、比率
同时依赖表格和文本的题优先
TAT-QA：优先保留
算术题 / derivation 题
依赖表格 + 文本
涉及 million / billion / thousand / % 等 scale 的题优先
TabularGSM：直接拿
hard
robustness

这一步的目的：先把明显简单题扔掉，减少后面 no-tool / teacher 成本。TabularGSM 官方数据页已经明确给出 hard 和 robustness 等子集。

5. 先跑 no-tool baseline

对训练候选集中的每条样本，先用 Qwen3-4B 跑一遍 不带工具 的单轮回答。

输出统一保存：

{
  "id": "...",
  "pred_no_tool": "....",
  "ok_no_tool": true,
  "judge_no_tool": {
    "type": "exact | numeric | dataset_specific",
    "detail": "..."
  }
}

要求：

prompt 固定
不给任何 tool schema
每题只出最终答案，不做中间轨迹

判分：

TabMWP / FinQA / TAT-QA：用你自己的标准化数值比较器
如果原数据集有官方 evaluator，就直接封装调用
TabularGSM：同样走数值/字符串标准化比较
MATH：只做可选迁移，不参与筛题
6. 再跑 teacher + python_exec

对同一批样本，跑一遍 teacher（你现在的 deepseek-chat + 本地 python_exec）。

只允许最多 2 次工具调用，超过 2 次直接丢样本。
每条样本保存原始轨迹：

{
  "id": "...",
  "messages": [...],
  "tools": [...],
  "pred_tool": "...",
  "ok_tool": true,
  "tool_stats": {
    "num_calls": 1,
    "num_success_calls": 1
  }
}

硬规则：

python_exec 只能接收一个参数：code
code 必须是自包含脚本
你的执行服务固定：
extract_code=false
auto_invoke=false
只要有 tool call，就必须真实执行并记录 observation
7. 留样规则（最重要）

只保留满足全部条件的样本：

训练正样本（tool-positive）
ok_no_tool == false
ok_tool == true
至少 1 次合法 tool call
至少 1 次工具执行成功
tool call 里的 code 是可解析 + 可执行 的自包含脚本
错误原因属于下面至少一类：
数值运算
表格读值
scale（million / billion / % / 单位）
多步组合推理
表格 + 文本混合取值

**丢弃**：
no-tool 也对
tool-enabled 也错
tool call 为空 / 参数坏 / 非 JSON
code 不是脚本，只是半截函数
observation 没被后续答案利用
只是格式修修补补，不是数值/表格/scale/组合推理问题
8. 再补一部分 “不要乱调工具” 的直答样本

最终冷启动 SFT 不要 100% 都是工具正样本。
要从训练候选集中再补一批 direct/no-tool 正样本，规则：

ok_no_tool == true
题型明显简单
不需要表格+文本混合推理
不涉及复杂 scale / ratio / multi-step
最终建议配比

如果你的目标是“强行把工具能力打出来”，用这个配比：

30% direct
50% single-tool
20% repair（两轮 tool）

9. 测试集怎么做
主测试集
TabularGSM_hard
TabularGSM_robustness
FinQA_hard_subset
TAT-QA_hard_subset
可选迁移测试
MATH
FinQA_hard_subset 规则

从官方 dev/test 里筛：

明显是数值题
至少两步推理
有 ratio / percent / change / divide
table + text 混合优先
TAT-QA_hard_subset 规则

从官方 dev/test 里筛：

arithmetic / derivation
table + text 混合
含 scale
至少两步运算

不要把训练筛过的样本混进 eval。

10. 交给 Codex 的最小任务清单
任务 1：下载并读取 5 个数据源
- TabMWP
- FinQA
- TAT-QA
- TabularGSM
- MATH

任务 2：写 5 个 adapter，把原始样本转成统一 normalized jsonl

任务 3：写 candidate 筛选器
- tabmwp_candidate_filter()
- finqa_candidate_filter()
- tatqa_candidate_filter()

任务 4：写 qwen3_4b_no_tool_runner()
输入：normalized jsonl
输出：pred_no_tool.jsonl

任务 5：写 teacher_tool_runner()
输入：normalized jsonl + python_exec server
输出：teacher_traj.jsonl

任务 6：写 grader
- grade_no_tool()
- grade_tool()
- classify_error_type()

任务 7：写 keep_filter()
只保留：
- no-tool 错
- tool 对
- 数值/表格/scale/组合推理
- tool code 可执行

任务 8：写 final_builder()
输出：
- train_direct.jsonl
- train_single_tool.jsonl
- train_repair.jsonl
- eval_tabulargsm_hard.jsonl
- eval_tabulargsm_robust.jsonl
- eval_finqa_hard.jsonl
- eval_tatqa_hard.jsonl
11. 最短执行逻辑
先下数据
→ 转统一格式
→ 先跑 Qwen3-4B no-tool
→ 再跑 teacher + python_exec
→ 对比 no-tool 和 tool-enabled
→ 只留 no-tool 错 / tool 对 / 且属于数值表格类难点的样本
→ 再补一部分 no-tool 直接正确样本，防止过度调用工具
→ 输出训练集和评测集

**设计tool use 执行任务的交互协议：用本session中标准协议格式,教师返回tool call我们本地处理后回填，回填的message格式见session中标准记录**
如果qwen3-4b暂时没办法部署到GPU上（已经被别人占用了，我们不要去抢，不要去硬挤！用cpu的方法进行推理，可以慢一些）