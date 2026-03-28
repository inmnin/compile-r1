下面给你一份只保留核心操作项的冷启动指南。目标只有一个：

Teacher 侧按 DeepSeek 官方 OpenAI-compatible tool-calling 格式蒸馏原始轨迹；后处理时不要转成 qwen-agent 风格，而是直接规范化成 Qwen3 的 OpenAI-compatible function-calling 原始格式。
因为这两者在原始轨迹层本来就是同一类协议：messages + tools + assistant.tool_calls + role=tool + tool_call_id + final assistant。真正需要做的是规范化字段和值，不是重新发明协议。Qwen3 官方还明确说明，function.arguments 在这层是 JSON-formatted string，工具结果通过 role="tool" 和 tool_call_id 回填。DeepSeek 官方也把 messages 定义为 system/user/assistant/tool 四类消息，并要求 tool message 带 tool_call_id。

一、你要落盘保存的“蒸馏原始格式”
1）顶层结构
{
  "id": "acecode_xxx",
  "bucket": "direct | single_tool | repair",
  "question": "...",
  "source": "...",
  "tools": [...],
  "messages": [...],
  "hidden_eval": {
    "pass_direct": 0.0,
    "pass_auto": 1.0,
    "pass_required": null
  }
}

核心只有两块：tools 和 messages。DeepSeek 的 tool_choice 支持 none / auto / required，正好对应你三条 teacher 路径。

2）tools 的标准格式
[
  {
    "type": "function",
    "function": {
      "name": "python_exec",
      "strict": true,
      "description": "Execute a short self-contained Python verification script to reduce uncertainty on a coding task. Use it only when execution can materially improve correctness, such as checking tricky edge cases, validating a candidate implementation on small examples, debugging suspected syntax/runtime issues, or comparing candidates when uncertain. Do not use it for trivial tasks when execution would not change the final answer. The code must be a complete runnable Python script. If you define helper functions, call them explicitly in the same script. Do not include Markdown fences. Do not use files, network access, subprocesses, or third-party packages.",
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
]

这里不能乱改。DeepSeek 官方规定：工具数组元素顶层必须有 type="function" 和 function；function.description 会被模型用于决定什么时候调用、怎么调用；strict=true 要配合 Beta base URL 使用；strict 模式下服务端会校验 schema。

3）messages 的标准格式
A. Direct / no-tool
[
  {"role": "system", "content": "SYSTEM_PROMPT"},
  {"role": "user", "content": "USER_PROMPT"},
  {"role": "assistant", "content": "def target_function(...):\n    ..."}
]
B. Single-tool
[
  {"role": "system", "content": "SYSTEM_PROMPT"},
  {"role": "user", "content": "USER_PROMPT"},
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_0001",
        "type": "function",
        "function": {
          "name": "python_exec",
          "arguments": "{\"code\":\"def candidate(...):\\n    ...\\n\\nprint(candidate(...))\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_0001",
    "content": "{\"status\":\"success\",\"stdout\":\"...\",\"error\":null,\"return_value\":null}"
  },
  {
    "role": "assistant",
    "content": "def target_function(...):\n    ..."
  }
]
C. Repair / multi-turn
[
  {"role": "system", "content": "SYSTEM_PROMPT"},
  {"role": "user", "content": "USER_PROMPT"},
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_0001",
        "type": "function",
        "function": {
          "name": "python_exec",
          "arguments": "{\"code\":\"...first check...\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_0001",
    "content": "{\"status\":\"error\",\"stdout\":\"\",\"error\":\"AssertionError\",\"return_value\":null}"
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_0002",
        "type": "function",
        "function": {
          "name": "python_exec",
          "arguments": "{\"code\":\"...fixed check...\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_0002",
    "content": "{\"status\":\"success\",\"stdout\":\"ok\\n\",\"error\":null,\"return_value\":null}"
  },
  {
    "role": "assistant",
    "content": "def target_function(...):\n    ..."
  }
]

这几条是硬规则：
assistant 的工具调用放在 tool_calls；function.arguments 必须是 JSON 字符串；tool 消息必须带 tool_call_id；最后的 assistant.content 是 final answer。Qwen3 官方对 OpenAI-compatible function calling 就是这样示例的，DeepSeek 的 message/body 定义也与此一致。

二、从 DeepSeek 原始轨迹到 Qwen3 原始格式：怎么转
结论

几乎不用“转协议”，只做规范化。
因为你蒸的是 DeepSeek 的 OpenAI-compatible 原始日志，而你要的 Qwen3 function-calling 原始格式也是 OpenAI-compatible 这一套。

规范化规则
tools 原样保留顶层 type/function 结构。
每个 tool-call assistant turn 统一成：
role="assistant"
content=null
tool_calls=[...]
tool_calls[*].function.arguments 必须规范成JSON 字符串。
每个 tool result 必须是：
role="tool"
tool_call_id=<对应call id>
content=<observation JSON 字符串>
最后一条 assistant.content 只保留最终 Python 代码。

所以不是“改几个字段名”那么简单；正确说法是：原始协议相同，但你要做强规范化。

三、后续如果要转成训练格式

如果你后面要喂给 LLaMA-Factory，推荐转成 ShareGPT function-calling 形式：

{
  "system": "SYSTEM_PROMPT",
  "tools": "[{\"name\":\"python_exec\",\"description\":\"...\",\"parameters\":{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\",\"description\":\"...\"}},\"required\":[\"code\"]}}]",
  "conversations": [
    {"from": "human", "value": "USER_PROMPT"},
    {"from": "function_call", "value": "{\"name\":\"python_exec\",\"arguments\":{\"code\":\"...\"}}"},
    {"from": "observation", "value": "{\"status\":\"success\",\"stdout\":\"...\",\"error\":null,\"return_value\":null}"},
    {"from": "gpt", "value": "def target_function(...):\n    ..."}
  ]
}

这里和原始日志的一个关键差异是：
ShareGPT 的 function_call.value 里，arguments 应该是对象，不再是字符串。
LLaMA-Factory 官方文档明确给了 human -> function_call -> observation -> gpt 的格式，并说明 tools、system 可放在样本顶层。

四、你的 prompt：可以用，但建议改成下面这个版本

你的原 prompt 大方向是对的，但为了更稳定地产生干净的原始工具轨迹，建议只保留下面这版。

System Prompt（英文）
You are a Python coding assistant solving function-implementation tasks.

You may use one tool, python_exec(code: string), when execution can materially reduce uncertainty.

Use the tool mainly for:
- checking tricky edge cases,
- validating a candidate implementation on small examples,
- debugging suspected syntax or runtime issues,
- comparing candidate implementations when you are unsure.

Do not use the tool for trivial tasks when execution would not change your final answer.

When you use the tool:
- make exactly one tool call at a time,
- do not include explanation in that assistant turn,
- put only a complete self-contained Python script in the code argument,
- if you define helper functions, call them explicitly in the same script,
- include any helper functions, assertions, or prints needed for verification,
- do not include Markdown fences,
- do not rely on files, network access, subprocesses, or third-party packages.

After receiving a tool result, either make another grounded tool call or provide the final answer.

Only the last assistant message may contain the final answer.
Your final answer must contain only the final Python solution implementing the required function or class.
Do not include tests.
Do not include explanation.
Do not include Markdown fences.
User Prompt（英文）
Solve the following Python coding task.

Return only the final Python solution in your last assistant message.

Task:
{question}

这版的目的只有三个：

让 teacher 更愿意把“何时调用工具”交给 tools[].function.description + tool_choice 去决定；
让 tool-call turn 尽量是纯协议消息，不要混自然语言解释；
让 final assistant turn 只产出最终代码。DeepSeek 官方明确说明 description 会影响调用时机与方式；同时 assistant message 的 content 在 schema 里是 nullable，所以工具调用 turn 令其为空是完全合法的。
五、teacher 蒸馏流程

三条路径共用同一套 system + user prompt，只改 API 参数：

Direct
{
  "tool_choice": "none",
  "thinking": {"type": "disabled"}
}
普通工具蒸馏
{
  "tool_choice": "auto",
  "thinking": {"type": "disabled"}
}
Tool 候选但 auto 没调时的 fallback
{
  "tool_choice": "required",
  "thinking": {"type": "disabled"}
}

DeepSeek 官方明确支持 tool_choice=none/auto/required，并支持 thinking.enabled/disabled 切换。strict mode 需要 Beta base URL。

六、反例数据格式：这些一律不要保留（**数据筛选与清洗的标准**）
反例 1：qwen-agent 风格原始日志
{
  "role": "assistant",
  "function_call": {
    "name": "python_exec",
    "arguments": {"code": "..."}
  }
}

不要。你现在要的是 Qwen3/OpenAI-compatible raw format，不是 qwen-agent 风格。

反例 2：原始日志里把 arguments 存成对象
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "python_exec",
        "arguments": {
          "code": "print(1+1)"
        }
      }
    }
  ]
}

不要。原始日志层 function.arguments 应该是 JSON 字符串。

反例 3：tool 消息没有 tool_call_id
{
  "role": "tool",
  "content": "{\"status\":\"success\",\"stdout\":\"2\\n\"}"
}

不要。tool 消息必须带 tool_call_id。

反例 4：tool-call turn 混 explanation
{
  "role": "assistant",
  "content": "I will first test a small example.",
  "tool_calls": [...]
}

不建议保留。虽然协议层未必非法，但这会污染你要蒸的“干净函数调用轨迹”。

反例 5：final answer 混测试 / 解释 / fence
{
  "role": "assistant",
  "content": "```python\ndef f(...): ...\n```\nThis solution works because ..."
}

不要。最后一条 assistant 只保留最终代码。

七、最短落地规则

你可以直接把这 6 条交给 Codex：

蒸馏原始日志统一存 DeepSeek/OpenAI-compatible 轨迹：messages + tools。
assistant 调工具时只存 tool_calls，content=null。
function.arguments 在原始日志层必须是 JSON 字符串。
tool 结果必须是 role=tool + tool_call_id + content(JSON字符串)。
final assistant 只保留最终 Python 代码。
后续转训练格式时，再把 raw log 转成 ShareGPT：human / function_call / observation / gpt，并把 arguments 从字符串解析回对象。

如果你要，我下一条直接给你两份可落盘的 JSONL 模板：

deepseek_raw_direct/single_tool/repair.jsonl 模板
llamafactory_sharegpt_direct/single_tool/repair.jsonl 模板