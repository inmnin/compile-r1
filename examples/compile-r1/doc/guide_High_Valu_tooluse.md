# 下一步 Codex 执行指南：产出真正高收益/复杂的 tool-use 轨迹

## 1. 不变项（不要改）

保持现在的 teacher 主策略不变：

- 继续使用 `explicit_xml`
- 继续使用本地 parser 校验
- 继续使用 repair 重试
- 继续把 teacher 轨迹再转换成 Qwen3 标准 `messages`
- 继续只保留 **最终答案 100% 正确** 的样本

这次要改的不是协议主干，而是：

1. **选题更难**
2. **让 tool call 更有价值**
3. **让 repair 轨迹更像真正的“诊断-修复”**

---

## 2. 这次的唯一目标

让新一批蒸馏数据不再只是“简单算术 + print”，而是更多地产生下面三种高价值轨迹：

1. **关键不确定性验证**
   - 工具调用直接针对题目里最容易做错的部分
   - 例如：scale、ratio、percentage、符号、组合关系、表格取值

2. **候选实现比较 / falsification**
   - 不是只算一个数
   - 而是比较两个候选公式、验证一个假设是否错、用 assert 排除错误路径

3. **多轮修复**
   - 第一次工具调用发现问题
   - 第二次工具调用针对问题修复
   - 最后再给最终答案

---

## 3. 下一步要做什么（总策略）

### 总原则

不要一上来全量跑。

先只对 **hard subset** 跑新策略，目标是先验证：

- tool call 是否明显更复杂、更有用
- repair 轨迹是否明显变多
- trivial `print-only` 是否显著下降
- 仍然保持最终答案 100% 正确

### 建议执行顺序

1. 从当前候选样本里先筛出 **hard subset**
2. 只对 hard subset 使用新的 prompt 约束
3. 先生成一小批（建议 200～500 条）
4. 做质量统计
5. 通过阈值后再放量

---

**涉及题目筛选/数据筛选的注意事项：**
**下面所有人工烧tokens对题目进行人工评测，也可以调用deepseek-chat api进行llm judge评测我都建议使用前者，前者codex做不到再用后者！！！！！！！！**

## 4. 如何选“真正值得调工具”的 hard subset

**核心约束**：筛选难题的时候用两种方法：1.写程序辅助   2.你人工去消耗tokens进行review评价，不要只依赖程序，**请你（codex）自己消耗tokens去对题目做人工的评测 或者依赖llm judge**

### 4.1 只优先抽这些题（**筛选标准，记得一定要人工烧tokens对题目进行人工评测，也可以调用deepseek-chat api进行llm judge评测***）

优先抽满足以下任意条件的题：

- 需要 **percentage / percent change / ratio / margin / growth / contribution**
- 需要处理 **million / billion / thousand / % / basis points / scale conversion**
- 需要同时看 **table + context** 才能答
- 需要 **至少两步** 数值组合
- 存在多个相似字段，容易拿错值
- question 本身包含“share of / what percent / how much of total / excluding / net of / change from ... to ...”
- 题目描述长，且表格里数值多于 4 个

### 4.2 先不要优先抽这些题（**筛选标准，记得一定要人工烧tokens对题目进行人工评测，也可以调用deepseek-chat api进行llm judge评测***）

这些题即使可调工具，也往往只会产生低价值轨迹：

- 直接查表就能答
- 只做一步加减乘除
- 题目里只涉及两个数字
- 没有 scale / ratio / hybrid reasoning
- 不需要 context，只看 question 就够

### 4.3 数据路由建议

把候选题先分成两类：

- `easy_or_direct`
- `hard_tool_worthy`

新 prompt 只打在 `hard_tool_worthy` 上。

---

## 5. prompt 只改约束形式，不改整体 teacher 策略

下面只是在你现有 teacher prompt 基础上**追加/替换约束**。

## 5.1 对现有 system prompt 的修改建议

### 需要修改的部分

#### A. 把“8 行限制”放宽为 hard 模式下最多 12～16 行（甚至更多，或者不加约束）

把：

- `keep it short, usually no more than 8 lines`

改成：

- `keep it concise; for simple checks aim for <= 8 lines, but for hard tasks you may use up to 16 lines if needed to verify the main uncertainty`（16行是不是还是少，你可以看情况再加多甚至无上限）

**意义**：
当前很多轨迹太短，实际上是在逼 teacher 只写“末端算术 + print”。

#### B. 明确禁止“只做末端算术确认”的无效调用

追加：

- `Do not use the tool only for the final one-line arithmetic step if the main uncertainty lies earlier in scale interpretation, value selection, or formula composition.`

**意义**：
逼 teacher 把工具调用放在真正困难的地方，而不是最后随手算一下。

#### C. 强化“围绕核心不确定性”

追加：

- `Use the tool only to resolve the single most important uncertainty in the task.`
- `Prefer checking the most error-prone part of the reasoning over checking an obvious final arithmetic operation.`

**意义**：
把 tool 从“算数器”提升成“关键疑点验证器”。

#### D. 强化候选比较 / falsification

追加：

- `Prefer assertions, candidate comparison, or minimal falsification tests over generic print-only checks whenever possible.`
- `If two formulas or interpretations are plausible, use the tool to disambiguate them.`

**意义**：
让轨迹里出现更强的验证模式，而不是只是 `print(value)`。

#### E. 要求 helper function 必须显式调用

保留并强化：

- `If you define helper functions, call them explicitly in the same script.`

**意义**：
避免 teacher 产出“只有定义、没有执行”的伪脚本。

---

## 5.2 给 hard subset 单独加一个 suffix

只在 hard subset 上追加下面这段约束：

```text
For this task, if you use the tool, the tool call should target the hardest or most ambiguous part of the problem.
A good tool call should usually do at least one of the following:
- verify a nontrivial ratio, percentage, or scale conversion,
- compare two candidate formulas or interpretations,
- check a likely failure mode with an assert,
- validate that the chosen values from the table/context are the correct ones.
Avoid trivial print-only checks unless they directly resolve the main uncertainty.
```

**意义**：
同一套 teacher 协议不变，只在难题上把“高价值调用”要求讲清楚。

---

## 5.3 repair prompt 也要加强

当前 repair prompt 主要是在修协议错误。现在再加一句，专门修“低价值调用”：

```text
Your previous tool call was valid but not useful enough.
The next tool call must target the main uncertainty of the task, not a trivial final arithmetic step.
Prefer an assert, a candidate comparison, or an explicit check on scale/value selection.
```

新增一个 reason 类别，例如：

- `tool_call_low_utility`

**意义**：
这样 repair 不只是修格式，还能修“有格式、没价值”的调用。

---

## 6. 要不要加显式 thinking / planning

### 主建议

**主流数据先不要加显式 thinking。**

继续保持：

- 非终止 assistant turn 只输出 `<tool_call>...</tool_call>`
- 最后一个 assistant 才输出 final answer

### 原因

你现在最需要稳定的是：

- 工具调用时机
- 工具调用质量
- observation 后继续修复
- final answer 收口

不是先蒸显式思维文本。

### 允许的小实验

可以单独做一个小分支（最多 5%～10% 样本）：

- 只在 `repair` 的 hard 样本上
- 在第一次 tool call 前允许一小段 planning
- 但这个 planning 不进入主流数据，不作为当前大规模生成主线

**意义**：
先用小实验判断“显式 planning”是否真的能让 tool script 更有价值。

---

## 7. tool call 质量过滤规则（新增）（**筛选标准，记得一定要人工烧tokens对题目进行人工评测，也可以调用deepseek-chat api进行llm judge评测**）

下一批数据不要只看“能不能执行”，还要看“有没有价值”。

给每个 tool call 打一个 `utility_tag`。

### 7.1 高价值 tool call（保留优先）（**筛选标准，记得一定要人工烧tokens对题目进行人工评测，也可以调用deepseek-chat api进行llm judge评测**）

满足任意一个：

- 使用 `assert` 检查一个关键假设
- 比较两个候选公式 / 候选解释
- 检查 scale / 百分比 / ratio / margin / change
- 显式验证从 table/context 取出的值是否正确组合
- 第一次 observation 明显导致第二次修复

### 7.2 低价值 tool call（不直接丢，但不要进入主训练集）（**筛选标准，记得一定要人工烧tokens对题目进行人工评测，也可以调用deepseek-chat api进行llm judge评测**）

满足任意一个：

- 只是把 final arithmetic 算一遍
- 只是把题目里的数抄一遍再 `print`
- 只是输出一个显然中间值，没有验证意义
- 没有帮助排除错误路径

### 7.3 坏轨迹（直接丢）

满足任意一个：

- tool call 不可执行
- code 为空 / prose / comments-only
- tool response 没被后续使用
- repair 第二轮只是机械重复第一轮
- final answer 不正确

---

## 8. 允许的 tool turn 数量

### direct
- 不要求工具
- 有工具但很 trivial 的样本不要保留为主样本

### single_tool
- 最多 1 次 tool call
- 但必须是高价值调用，不是 trivial print

### double_tool

- 最多 2 次 tool call
- 两次是高价值调用，不是 trivial print
- 这类数据比较少

### repair
- 允许最多 3 次 tool call
- 推荐目标：让 `repair` 样本里更多出现 **2 次调用**
- 第 1 次用于定位主不确定性
- 第 2 次用于修复/验证修复结果

**意义**：
你现在的 repair 样本如果只有“一次算错，一次重算”，价值还是不够。目标是形成真正的“诊断 -> 修复”。

---

## 9. Codex 具体要做的事

按下面顺序做，不要跳步。

### 任务 1：加一个 hard subset 选择器

新增脚本：

- `select_hard_toolworthy_samples.py`
记得一定要人工烧tokens对题目进行人工评测，或者调用deepseek-chat api进行llm judge评测


输入：现有候选题池
输出：`hard_toolworthy_ids.jsonl`

规则：按第 4 节筛选。

---

### 任务 2：更新 teacher prompt 模板

修改：

- `task_find_common.py` 中的 system prompt suffix
- `task_find_teacher_tool_runner.py` 中 repair prompt

要求：

- 不改协议主干
- 只加“高价值 tool call”约束
- 放宽 hard 题的 code 行数上限

---

### 任务 3：新增 low-utility 检测

在本地 parser / post-process 里增加 `tool_call_low_utility` 检测。

最低实现要求：

如果满足下面任意条件，则标记 `low_utility=true`：

- 只有 `print(<single arithmetic expression>)`
- 没有 `assert`
- 没有候选比较
- 没有 scale / ratio / percent / value-selection 检查
- 工具调用目标明显不是核心难点

对 `auto` 模式：
- 先标记
- 不直接修

对 `required` 模式：
- 允许走一次 repair
- repair prompt 要求更高价值的 tool call


可以调用先用的deepseek-chat api进行llm judge评测



---

### 任务 4：先做一批 200～500 条 pilot

不要直接全量生成。先跑：

- `hard_toolworthy` 子集
- 200～500 条
- `tool_choice=auto` 为主
- 一小部分 `required` 兜底

输出统计：

- `protocol_error_rate`
- `tool_success_rate`
- `avg_tool_turns`
- `repair_rate`
- `assert_rate`
- `comparison_rate`
- `low_utility_rate`
- `final_answer_accuracy`

---

## 10. 放量前必须满足的阈值

只有全部满足，才开始大规模生成：

- `protocol_error_rate == 0`
- `final_answer_accuracy == 100%`
- `low_utility_rate <= 20%`
- `assert_rate + comparison_rate >= 40%`
- `repair` 样本里，至少 30% 有真正的第二次修复调用
- trivial `print-only` 调用占比明显下降

如果这些阈值没过：

- 不放量
- 继续改 hard subset 路由和 prompt suffix

---

## 11. 这么做的意义

这一步的意义不是“让轨迹更长”，而是让轨迹更值得学。

如果不做这一步，模型会继续学到：

- 工具只是拿来做末端算术确认
- tool call 只是“把题里数字抄进 print”
- repair 只是机械重试

做完这一步，模型更可能学到的是：

- 先识别任务里最不确定的地方
- 用工具去验证关键假设
- 通过 observation 修正后续动作
- 在真正有收益的地方调用工具，而不是到处乱调

这才是“高收益 / 复杂 tool-use 轨迹”的意义。

---


