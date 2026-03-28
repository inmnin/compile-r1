先给一个总判断：
 你这批冷启动样本本质上是“自然语言代码规格 → 目标 Python 函数/类实现”任务。样本里 question_input 是具体题目，source 来自 oss / evol / bigcode_python_fns，最后都落到可运行的 Python 解答；题型既有很直接的业务/格式化/映射题，如 change_activity、list_virtual_networks、can_reach_bananas，也有更适合借助执行验证的题，如 reverse_number、lcm_three_numbers 等。并且在这 30 条样本里，num_tool_calls 既有 0 次，也有 1 次，甚至有多轮调用，这说明你当前的底层任务不是“所有题都必须用工具”，而是“代码题 + 可选执行验证”的混合分布。
qwen8b_code_tool_failed_explora…
qwen8b_code_tool_failed_explora…
qwen8b_code_tool_failed_explora…
qwen8b_code_tool_failed_explora…
冷启动 SFT 蒸馏制作指南**。
一、冷启动 SFT 到底要教会模型什么
你这次冷启动不要再追求“让模型学会旧标签格式”，而要明确教会 6 件事：
工具调用名称和参数格式
 模型要稳定输出 python_exec(code=...) 这种结构化调用，而不是自由文本乱写。
什么叫 final answer
 最后一轮 assistant 的内容必须是最终 Python 解答代码本体，不能混解释、不能混测试、不能混 markdown fence。
看见 observation 后如何继续
 模型要么继续发下一次更有根据的工具调用，要么结束并给最终代码。
什么时候该调工具
 例如：边界条件不确定、字符串/解析/正则/Unicode/递归/位运算/格式细节有风险、需要快速验证候选实现时。
什么时候不该调工具
 对非常直接的一步式题，不应该强行调用工具。
如何写“最小自包含”的验证脚本
 工具调用里的 code 应该是完整可执行的小脚本，而不是半截函数、不是 markdown、不是自然语言。
二、AceCode-87K 在这个项目里应该怎么用
AceCode-87K 官方数据结构里包含 question、test_cases、inferences、context_messages 等字段；官方 README 还说明它可以直接用于 RL tuning，并且数据平均每题大约有 16 条测试样例。这对你非常重要：
question：给模型看的任务描述
test_cases：只给内部评测/筛选器看，不要轻易暴露给学生模型
context_messages：可作为参考，但我不建议直接复用
inferences/pass_rate：可用于辅助挑选 hard / easy 样本。 里旧的 question_instruction 包装。**
 旧包装里那些 <think>/<code>/<result>/<answer> 已经是你抛弃的协议。
 你真正该保留的是：题目的语义内容，也就是 question_input/原始 question。
qwen8b_code_tool_failed_explora…
因此给模型看的输入**：只用 question
给离线筛选器看的隐藏监督：test_cases
可选参考：source、context_messages
丢弃：旧蒸馏样本里那套 <think>/<code>/<result>/<answer> 的表层格式
三、冷启动数据的总策略：不是“全量蒸馏”，而是“分桶蒸馏”
1. 你不应该把 87K 全部无脑蒸馏
因为你文档样本已经说明，很多题是直接完成型，根本不需要执行器；如果你把它们全都强行蒸馏成“先调工具再回答”，模型会学会过度调用。
qwen8b_code_tool_failed_explora…
 
qwen8b_code_tool_failed_explora…
把l Direct）
 这类题通常是一眼可写、低不确定性、一步式实现：
简单字符串替换/拼接
简单列表过滤/计数
简单字典映射/更新
固定公式计算
简单布尔判断
从你文档样本看，change_activity、can_reach_bananas、contains_duplicates、filter_even_numbers、store_credentials 这类就非常像 Direct。
qwen8b_code_tool_failed_explora…
 
qwen8b_code_tool_failed_explora…
T但有明显边界风险，执行一次小脚本能显著降低错误率：
解析/格式转换
忽略空格/大小写/标点
简单递归
Unicode / escape sequence / 正则
版本号比较
精确字符串格式保持
从样本看，decode_escape_sequences、shared_chars、is_valid_upgrade、parse_duration 这类更适合单次验证。
qwen8b_code_tool_failed_explora…
 
qwen8b_code_tool_failed_explora…
Cu写错；看一次 observation 后再修一轮更合理：
递归/位运算/特殊约束题
“不得使用某 builtin / 某操作”
边界极多的模拟/搜索/解析
易超时、易漏特殊 case 的算法题
样本里 reverse_number、lcm_three_numbers、add_without_plus_operator、maze_solver、robust_median 这类更接近 Repair 桶。
qwen8b_code_tool_failed_explora…
 
qwen8b_code_tool_failed_explora…
四、单条样本“到*”的两阶段打标。
第一阶段：静态预判
先用题目文本做一个粗筛，不花 Teacher API 钱。
判成 “Direct 候选” 的特征
满足越多越像 direct：
题目要求是简单映射、过滤、拼接、比较、格式化、去重、计数
题目没有 “preserve punctuation / recursion / regex / unicode / escape / bitwise / no builtin / version compare / maze / parser” 这类高风险词
题目短、规则少、约束简单
一次线性扫描就能实现
判成 “Tool 候选” 的特征
满足越多越像 tool-needed：
题目里出现：
recursion
regex / parse / decode / escape / unicode
preserve spaces / punctuation / exact formatting
bitwise / without using + / without sorting / without built-ins
version / timedelta / date
maze / graph / path / search
edge case / invalid input / mixed input types
测试样例数多、规则多、格式约束强
你读题就知道“一次写对概率不高”
静态预判只是候选集路由，不是最终标签。
第二阶段：动态验证（真正决定样本性质）
这是你最应该实现的“单条数据确认性质的方法”。
对每个样本，跑三条 teacher 路径：
路径 1：No-tool baseline
调 DeepSeek Chat
tool_choice="none"
只看问题，不给工具
得到最终代码 answer_direct
用隐藏 test_cases 跑离线验证
得到 pass_direct
路径 2：Auto-tool teacher
调 DeepSeek Chat
tool_choice="auto"
工具可用
让 teacher 自己决定是否调工具
得到轨迹 traj_auto
抽最后一个 assistant final code 作为 answer_auto
用隐藏 test_cases 跑验证
得到 pass_auto
记录：
是否有合法 tool call
是否至少 1 次执行成功
tool turn 数
observation 后是否发生改写
路径 3：Required-tool fallback（只对 Tool 候选跑）
只对静态上像 tool-needed，但 auto 没调工具的样本跑
tool_choice="required"
得到 traj_required
hidden tests 验证得到 pass_required
最终标签规则
我建议你按下面的规则定桶：
Direct
满足：
pass_direct = 1
且 pass_auto <= pass_direct
且 teacher 在 auto 模式下没有明显从工具中获益
Single-Tool
满足：
存在至少 1 次合法且成功的 tool call
pass_tool = 1
且 pass_tool > pass_direct
Repair / Multi-Turn
满足：
存在至少 2 次工具调用
第一次工具调用后没有直接结束
后续根据 observation 修改了代码或策略
最终 pass_tool = 1
丢弃
满足任一：
final answer 不过隐藏 tests
tool call 参数不合法
tool code 不是自包含脚本
final answer 里混说明文字
direct/tool 两条路都失败
强制 required 也只是在乱调工具，没有真实改进
五、我建议的冷启动数据总条数与配比
推荐主方案：18,000 条 train + 500 条 val + 500 条 smoke-eval
配比
40% Direct：7,200
40% Single-Tool：7,200
20% Repair / Multi-Turn：3,600
这是我最推荐的比例。
为什么不是 no-tool 占大头？
 因为 AceCode 里简单题很多，如果你不刻意拉高 tool-beneficial 样本占比，模型会学到“几乎都不需要工具”。而你的冷启动目标里明确包含：
学会工具名字和参数格式
学会 observation 后怎么继续
学会什么时候该调 / 不该调
这三件事都需要足够多的正向 tool 轨迹。
如果预算更紧：12,000 条 train + 300 条 val + 300 条 smoke-eval
也可以照这个比例缩放：
Direct：4,800
Single-Tool：4,800
Repair：2,400
一个额外约束
Repair 桶宁缺毋滥。
 如果你只能挖到 12% 的高质量 repair 样本，就只留 12%，不要靠低质量“强制 required”样本硬凑到 20%。
六、模型面对的工具：只保留一个，而且要极简
你本地已经有很好的一套执行服务了。我的建议是：
模型面对的唯一工具
只暴露一个：
{
  "type": "function",
  "function": {
    "name": "python_exec",
    "strict": true,
    "description": "Execute a short self-contained Python script to reduce uncertainty on a coding task. Use this tool only when execution can materially improve correctness: checking tricky edge cases, validating a candidate implementation on small examples, debugging suspected syntax or runtime issues, or comparing alternative implementations. Do not use it for trivial tasks when execution would not change the final answer. The code must be a complete runnable Python script with any helper functions, prints, or assertions needed for verification. Do not include Markdown fences. Do not use files, network access, subprocesses, or third-party packages.",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "A complete self-contained Python script to execute. It may define helper functions, run assertions, and print diagnostic outputs. No Markdown fences."
        }
      },
      "required": ["code"],
      "additionalProperties": false
    }
  }
}
这里有三个关键点：
只保留一个参数 code
 DeepSeek 的 tool schema description 会被模型用于决定“什么时候调、怎么调”；tool_choice 还支持 none/auto/required 三种模式，所以把时机写进 description 非常重要。
开 strict mode
 Deep个 function 上 strict=true；模型会更严格遵守 JSON Schema。
**additionalProperties=fal:contentReference[oaicite:32]{index=32}应明确约束，并把 additionalProperties设为false`。
七、你现有本地工具服务怎么接入最合理
你给我的这份服务代码其实已经够好了。关约**。
模型侧只调用 /run
请求固定为：
POST /run
{
  "raw_text": "<self-contained python script>",
  "extract_code": false,
  "timeout": 12,
  "auto_invoke": false
}
我建议这么设的原因：
extract_code=false
不要让模型依赖 markdown fence 提取。
 模型应该学会直接把完整 Python 脚本放到 code 参数里。
auto_invoke=false
不要依赖你服务里“自动找 main/run/solution/solve”这种启发式。
 蒸馏阶段要让 teacher/student 都学会写明确的自包含验证脚本：自己定义函数、自己调用、自己 assert / print。
 否则契约会漂。
工具 observation 的标准化
把 /run 返回值规范成固定 observation：
{
  "status": "success | error | timeout | busy | system_error",
  "stdout": "...",
  "error": "...",
  "return_value": ...
}
然后作为 tool message 的 content 回填给 teacher。
不要把 /run_humaneval 暴露给模型
/run_humaneval 适合离线评测/内部验证，不适合做模型面向的通用工具。
 对 AceCode 冷启动蒸馏，模型面向的工具只要 python_exec(code) 就够了。
你还应该额外加一个“内部评测器”
不是模型调用，而是蒸馏脚本内部调用。
建议额外做一个内部函数或新 endpoint：
POST /run_tests
{
  "code": "...final candidate code...",
  "tests": ["assert ...", "assert ...", "..."],
  "timeout": 12
}
它的逻辑是：
program = candidate_code + "\n\n" + "\n".join(test_cases) + "\n"
然后内部调用你现有的 SERVICE_SANDBOX.run_async(extract_code=False, auto_invoke=False)。
这个 internal grader 只用于：
打 pass_direct
打 pass_auto
判定样本属于哪一桶
做最终留样/弃样
不要把 hidden test executor 暴露成模型工具。
你的服务部署建议
根据你现在的代码：
先用 RUNPY_API_WORKERS=1
先调 RUNPY_MAX_WORKERS
再调 RUNPY_MAX_INFLIGHT
原因很简单：
 你现在 SERVICE_SANDBOX 是进程级全局对象，而每个 Uvicorn worker 都会各自创建一套 ToolSandbox + ProcessPoolExecutor。所以如果你盲目增加 API workers，会把执行池数量和内存占用一起乘上去。
 对于本地蒸馏，通常更稳的做法是：
RUNPY_API_WORKERS=1
RUNPY_MAX_WORKERS = 可用 vCPU / 2
RUNPY_MAX_INFLIGHT = 3 * RUNPY_MAX_WORKERS
RUNPY_TIMEOUT = 10~15
如果后面发现单进程 API 层成瓶颈，再把 RUNPY_API_WORKERS 从 1 提到 2。
八、DeepSeek 官方 API 蒸馏流程
DeepSeek 的 Chat Completion API 本身就是 OpenAI-compatible，messages 支持 system / user / assistant / tool，tool message 还要带 tool_call_id；tools 支持 JSON Schema；tool_choice 支持 none / auto / required；assistant 响应里会返回 tool_calls。此外，函数的 description 会被模型用于决定“何时调用、如何调用”。
我建议的 teache/api.deepseek.com/beta`
工具：上面的 python_exec
strict：开
thinking：disabled
tool_choice：
Direct 桶：none
普通工具桶：auto
Tool 候选但 auto 不调：required
max_turns：3（绝大多数冷启动样本不要超过 2 次 tool）
temperature：低一点，比如 0.2
我建议 thinking=disabled，因为你的冷启动目标是稳定工具协议，不是蒸馏 teacher 的思维文本。DeepSeek API 里 thinking 支持 enabled/disabled 切换。
蒸馏时的 canonical system oding assistant solving function-implementation tasks.
You may use one tool:
python_exec(code: string)
Use the tool only when execution can materially reduce uncertainty, especially for:
checking tricky edge cases,
validating a candidate implementation on small examples,
debugging suspected syntax or runtime issues,
comparing candidate implementations when you are unsure.
Do not use the tool for trivial tasks when execution would not change your final answer.
When you call the tool:
produce exactly one function call at a time,
put only a complete self-contained Python script in the code argument,
include any helper functions, assertions, or prints needed for verification,
do not include Markdown fences,
do not rely on files, network access, subprocesses, or third-party packages.
After receiving a tool result, either make another grounded tool call or provide the final answer.
Your final answer must contain only the final Python solution implementing the required function or class.
 Do not include tests.
 Do not include explanation.
 Do not include Markdown fences.

## 中文翻译

```text
你是一个 Python 编码助手，负责完成函数/类实现任务。

你可以使用一个工具：

python_exec(code: string)

只有当执行代码能够显著降低你的不确定性时，才使用这个工具，尤其是在：
1. 检查棘手边界条件；
2. 用小样例验证候选实现；
3. 调试你怀疑存在的语法或运行时错误；
4. 在不确定时比较多个候选实现。

对于非常直接、执行并不会改变最终结论的题，不要使用工具。

调用工具时：
- 每次只发出一个函数调用；
- code 参数里只能放完整、自包含的 Python 脚本；
- 你可以包含辅助函数、assert 或 print 来做验证；
- 不要使用 Markdown 代码围栏；
- 不要依赖文件、网络、子进程或第三方包。

拿到工具结果后，你要么继续发起下一次有根据的工具调用，要么给出最终答案。

你的最终答案必须只包含实现目标函数或类的最终 Python 代码。
不要包含测试。
不要包含解释。
不要包含 Markdown 代码围栏。
user prompt 模板（英文）
Solve the following Python coding task.

Return only the final Python solution in your last assistant message.

Task:
{question}
中文翻译
请解决下面这个 Python 编码任务。

在最后一条 assistant 消息里，只返回最终 Python 解答代码。

题目：
{question}
这个 prompt 要不要在蒸馏 / 训练 / 推理保持一致？
要，尽量一致。
最好的做法是：
system prompt 一致
user prompt 一致
变化只放在外层 API 参数：
tool_choice=none
tool_choice=auto
tool_choice=required
这样你收集 Direct / Single-Tool / Repair 三种桶时，协议本体不变，只是采样策略不同。
九、DeepSeek 蒸馏得到的“原始日志格式”
我建议你蒸馏时先存成 OpenAI-style raw trajectory，不要一开始就直接写成 Alpaca。
No-tool 样本原始格式
{
  "id": "acecode_xxx",
  "bucket": "direct",
  "source": "oss",
  "question": "...",
  "tools": [...],
  "messages": [
    {"role": "system", "content": "...system prompt..."},
    {"role": "user", "content": "...question..."},
    {"role": "assistant", "content": "def solution(...):\n    ..."}
  ],
  "hidden_eval": {
    "pass_rate": 1.0
  }
}
Single-tool 样本原始格式
{
  "id": "acecode_xxx",
  "bucket": "single_tool",
  "source": "evol",
  "question": "...",
  "tools": [...],
  "messages": [
    {"role": "system", "content": "...system prompt..."},
    {"role": "user", "content": "...question..."},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "python_exec",
            "arguments": "{\"code\": \"def f(...): ...\\nprint(...)\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "{\"status\":\"success\",\"stdout\":\"...\",\"error\":\"\",\"return_value\":null}"
    },
    {
      "role": "assistant",
      "content": "def target_function(...):\n    ..."
    }
  ],
  "hidden_eval": {
    "pass_rate": 1.0
  }
}
Repair / 多轮样本原始格式
{
  "id": "acecode_xxx",
  "bucket": "repair",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_1", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_2", "content": "..."},
    {"role": "assistant", "content": "def final_solution(...):\n    ..."}
  ]
}
十、如何把原始日志转成 LLaMA-Factory 恰好适配的格式
LLaMA-Factory 对 ShareGPT 的 SFT 数据格式，明确支持 human / function_call / observation / gpt 这些角色，并且样本还能带 system 和 tools；文档还明确说明了 function_call、observation 的位置要求。
你应该转成这个格式（推PROMPT",
"tools": "[{...python_exec schema...}]",
 "conversations": [
 {
 "from": "human",
 "value": "Solve the following Python coding task.\n\nReturn only the final Python solution in your last assistant message.\n\nTask:\n..."
 },
 {
 "from": "gpt",
 "value": "def solution(...):\n    ..."
 }
 ]
 }

### Single-tool 样本

```json
{
  "system": "YOUR_SYSTEM_PROMPT",
  "tools": "[{...python_exec schema...}]",
  "conversations": [
    {
      "from": "human",
      "value": "Solve the following Python coding task.\n\nReturn only the final Python solution in your last assistant message.\n\nTask:\n..."
    },
    {
      "from": "function_call",
      "value": "{\"name\":\"python_exec\",\"arguments\":{\"code\":\"def f(...): ...\\nprint(...)\"}}"
    },
    {
      "from": "observation",
      "value": "{\"status\":\"success\",\"stdout\":\"...\",\"error\":\"\",\"return_value\":null}"
    },
    {
      "from": "gpt",
      "value": "def target_function(...):\n    ..."
    }
  ]
}
Repair 样本
{
  "system": "YOUR_SYSTEM_PROMPT",
  "tools": "[{...python_exec schema...}]",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "function_call", "value": "{\"name\":\"python_exec\",\"arguments\":{\"code\":\"...\"}}"},
    {"from": "observation", "value": "{\"status\":\"error\",\"stdout\":\"\",\"error\":\"AssertionError\"}"},
    {"from": "function_call", "value": "{\"name\":\"python_exec\",\"arguments\":{\"code\":\"...fixed...\"}}"},
    {"from": "observation", "value": "{\"status\":\"success\",\"stdout\":\"ok\",\"error\":\"\"}"},
    {"from": "gpt", "value": "def final_solution(...):\n    ..."}
  ]
}
对应的 dataset_info.json
LLaMA-Factory 文档给出的 ShareGPT 数据集描述是：
{
  "acecode_tool_coldstart": {
    "file_name": "acecode_tool_coldstart_sharegpt.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools"
    }
  }
}
这个 conversations/system/tools 的映射是文档明确支持的。
训练参数建议
LLaMA-Factory 的参数里，train_on_prompt=:contentReference[oaicite:42]{index=42}训练，mask_history=False` 则不只训练当前轮。对你的冷启动第一版，我建议：
train_on_prompt: false
mask_history: false
这两个参数在官方参数文档里都有。
十一、冷启动数据的质量过滤规则（必须严同时满足：
final answer 是纯 Python 代码
final answer 能被 ast.parse
final answer 不含 markdown fence
final answer 不含自然语言解释
用隐藏 test_cases 评测通过
与 question id 一一对应
工具 schema 名称、参数名完全匹配
工具样本额外过滤
对于 Single-Tool / Repair 样本，再加：
至少 1 次合法 function call
arguments 是合法 JSON
code 非空
至少 1 次工具执行成功
tool snippet 是自包含脚本
observation 后 assistant 有“继续决策”或“最终结束”行为
Repair 桶必须至少 2 次调用，且第二次不是第一次的机械重复
强制丢弃样本
出现任一就丢：
工具调用里带 markdown fence
工具调用里只有函数定义，没有执行/验证逻辑
最终答案里混测试代码
teacher 因 required 被逼着乱调工具，但隐藏测试无收益
多轮调用只是重复同一坏调用
observation 根本没被利用
十二、建议的数据生成流程（直接给 Codex）
下面这套流程是我建议你直接让 Codex 落地的。
阶段 A：读取原始 AceCode
读取：
id
source
question
test_cases
context_messages（只做可选参考）
AceCode-87K 官方 README 明确说明这些字段存在。
阶段 B：静态预判
脚本：score_acecode_tool_need.py
 direct_candidate | tool_candidate`
static_features = {...}
阶段 C：teacher 采样
脚本：distill_with_deepseek.py
对每条样本：
跑 tool_choice=none
跑 tool_choice=auto
必要时跑 tool_choice=required
内部循环：
发 messages + tools
收 assistant
如果有 tool_calls：
取 function.name
解析 arguments
调本地 /run
回填 tool message
没有 tool_calls：
认为是 final answer
结束
DeepSeek API 的 messages 支持 system/user/assistant/tool，tool message 需要 tool_call_id；assistant 响应可带 tool_calls。
阶段 D：隐藏测试验证pass_direct
pass_auto
pass_required
并决定最终桶标签。
阶段 E：转 ShareGPT
脚本：convert_to_llamafactory_sharegpt.py
把原始日志转成：
system
tools
conversations
阶段 F：配比采样
脚本：build_final_coldstart_dataset.py
输出：
train.jsonl
val.jsonl
smoke_eval.jsonl
并按：
40% direct
40% single-tool
20% repair
采样。
十三、我对你现有本地工具实现的具体建议
你这份服务已经具备：
FastAPI 服务层
/run
/run_humaneval
ProcessPoolExecutor
asyncio Semaphore
过载保护 / busy
超时与 pool restart
/stats
从蒸馏角度我建议只做三个改动：
改动 1：给模型的工具只用 /run
不要让模型知道 extract_code 或 auto_invoke 这些内部细节。
 你在 orchestrator 里固定：
payload = {
    "raw_text": code,
    "extract_code": False,
    "timeout": 12,
    "auto_invoke": False,
}
改动 2：加内部 /run_tests
给筛选器用，不给模型用。
 它接收 code + tests，返回：
{
  "passed_all": true,
  "passed_count": 14,
  "total_count": 16,
  "status": "success",
  "stdout": "...",
  "error": ""
}
改动 3：可选加 code-hash cache
蒸馏时 teacher 很容易反复执行相同小脚本。
 在服务或 orchestrator 层做：
key = sha256(code)
cache success / stdout / error 30~120 分钟
这会非常省时间。
十四、关于“大量冷启动轨迹 SFT”到底有没有用
有，但前提是：
协议固定
工具真的可执行
轨迹可回放
最终答案可验证
tool/no-tool 分布合理
最近一篇专门做“从文本合成 tool-use trajectories”的工作报告了多轮工具使用 benchmark 的提升，说明大规模、可验证的工具轨迹 SFT 本身是有价值的；但前提不是“轨迹越多越好”，而是“格式稳定、质量可验证、和推理时协议一致”。
所以对你这里来说，重点不是 87K 全蒸馏，而是**18K 左右的高质量协议一致（可以直接交给 Codex）
如果只保留一句话，我的建议是：
把 AceCode-87K 当成“问题语义 + 隐藏测试源”，不要当成旧格式轨迹源；重新用 DeepSeek Chat + 本地 python_exec 工具，按 messages -> tool_calls -> tool result -> final code 协议蒸馏出一套 18K 左右、40/40/20 配比的 ShareGPT 冷启动数据。
最重要的几个决定是：
只保留一个模型可见工具：python_exec(code)
DeepSeek 用 strict function calling
teacher 采样分三路：none / auto / required
hidden tests 只做离线筛选，不暴露给学生
最终转成 LLaMA-Factory ShareGPT：human / function_call / observation / gpt
SFT 数据里 direct/no-tool 一定要占到 40% 左右
不要把旧 <code> 标签协议带进来
下一条我可以直接给你三份可以让 Codex 开写的东西：
 一份 DeepSeek 蒸馏脚本骨架，一份 LLaMA-Factory ShareGPT 转换脚本骨架，以及一份 AceCode 单条样本自动分桶与过滤规则的伪代码。

请结合我们本地已经实现好的代码执行本地server，来实现sft数据蒸馏并启动这个进程，24线程，监控20分钟，没有因为并发太多而导致异常就继续高并发蒸馏，并预估蒸馏结束时间


在指南中prompt/函数约束的部分只保留英文，去掉中文翻译