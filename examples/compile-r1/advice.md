请为 TIGER-Lab/AceCode-87K 编写蒸馏数据生成脚本，用 DeepSeek 生成“会正确使用代码执行工具”的冷启动轨迹。数据集中每条样本字段请按以下方式使用：

【1) 数据字段映射】
- id: 样本唯一 ID，用于日志/去重/追踪。
- source: 来源，仅用于元数据记录。
- question: 题面文本（用于构造用户输入）。
- context_messages: 题面对应的对话消息（role/content）。如果存在且你使用 ChatML/对话接口，则优先用它作为输入；若它是嵌套 list（List[List[dict]]），选择第一组 messages；否则直接用 question 作为用户消息。
- test_cases: List[str]，每个元素是一条 Python assert 单测（标准验证用例）。注意：test_cases 只给“判题器/验证器”，不要直接拼到模型 prompt 里（避免模型通过背断言投机）。
- inferences: 明确忽略，不读取、不用于 prompt、不用于过滤（我们只需要标准 test_cases）。

【2) 目标轨迹格式（同一轨迹可多次调用执行工具）】
每条蒸馏样本的 assistant 输出是一个完整 trace 字符串，包含若干轮：
  <think>…</think>
  <code>…</code>
  <result>…</result>
重复若干次后，以最终：
  <answer>…</answer>
结束。关键要求：
- <code>/<result> 可以出现多次：模型在解题过程中可多次请求执行 Python 代码，脚本必须逐次执行并回填 <result>。
- <result> 只能由蒸馏脚本生成（来自 sandbox 执行真实结果），模型不允许“臆造”result。
- <answer> 只出现一次且必须是最后输出：放最终可提交的完整 Python 解（只放解法代码，不附带测试）。

【3) 生成-执行-回填 的交互协议（工具式蒸馏）】
对每条样本，按以下循环生成轨迹：
A. 构造初始输入（messages）：
   - system 指令：要求 DeepSeek 遵守上述标签协议；当需要运行代码时输出一个 <code>…</code> 并停止，让系统执行后回填 <result>；当准备好最终解时输出 <answer>…</answer> 并停止。
   - user 内容：来自 context_messages（优先）或 question。
   - 严禁把 test_cases 作为用户可见上下文拼进去。

B. 循环最多 max_steps 轮：
   1) 调用 DeepSeek 生成下一段 assistant 文本（必须包含 <think>；可以包含 0 或 1 个 <code>；不允许一次输出多个 <code>，这样便于稳定解析和逐步工具调用。若模型输出多个 <code>，脚本应按出现顺序逐个执行并在每个 <code> 后插入对应 <result>，然后再把拼接后的内容作为该轮 assistant 输出。）
   2) 若检测到 <code>…</code>：
      - 将 code 发送给 Python 执行 sandbox 执行（这是“工具调用”）。
      - sandbox 返回 stdout/stderr/异常/耗时/是否超时等；蒸馏脚本生成 <result>…</result> 并插入到 trace 中紧跟在对应 <code> 后面。
      - 将该轮 assistant（含回填的 <result>）追加到对话历史，再进入下一轮，让模型根据 result 继续推理/修正。
   3) 若检测到 <answer>…</answer>：停止生成循环，进入最终判题步骤。
   4) 若既无 <code> 也无 <answer>：视为无效轮次，可追加一条 system 纠错提示要求输出 <code> 或 <answer>，继续；超过阈值则丢弃该样本轨迹。

【4) 最终正确性过滤（只保留全通过轨迹）】
生成结束后，必须用 test_cases 对最终 <answer> 中的代码做“干净环境”的最终判题：
- 在一个全新 sandbox 会话中 exec(answer_code)；
- 逐条执行 test_cases（assert 语句）；
- 全部通过 => 保留该样本（保存完整 trace：包含所有中间 <code>/<result> 以及最终 <answer>）；
- 任一失败/异常/超时 => 丢弃该样本轨迹（不要保存）。

注意：最终判题必须在“全新会话”运行，不能复用中间工具调用时产生的状态，避免模型靠中间状态/缓存投机通过。

【5) <result> 内容规范（中间工具输出，非最终答案）】
每个 <result> 建议写成紧凑 JSON（便于训练与解析），字段包括：
- ok / exception_type / exception_msg（截断）/ stdout_snippet / stderr_snippet / wall_time_ms
并强制限制长度（如每段最多 4KB）。
重要：<result> 不要回显完整 test_cases 文本；如果你允许在中间阶段也“跑测试”，只能回传通过数/失败索引，不回传断言原文（避免把期望值泄漏到可学习文本里）。

【6) 高并发/高吞吐的执行工具（sandbox）要求】
蒸馏吞吐量的瓶颈通常在代码执行与超时回收上，请按以下架构实现：
- 将 sandbox 执行做成“可并发的 worker 池”：
  - 单机：用 asyncio 调度 + ProcessPoolExecutor（或 multiprocessing）维护 N 个长期存活的执行 worker；
  - 每个 worker 内部运行隔离执行（建议 nsjail 或 AppArmor/CodeJail 之类），并设置超时/内存上限；
  - 用有界队列/信号量做 backpressure：限制同时在跑的代码执行数 = N（可配置），防止过载。
- 避免“每个 <code> 新起一个进程/容器”的高开销做法：优先复用 worker（但每次任务都必须在受控临时目录中运行，并在任务结束后清理环境）。
- LLM 生成与 sandbox 执行解耦：主协程批量发起 DeepSeek 请求；<code> 执行请求进入 sandbox 队列；两端流水线并行提高整体吞吐。
- 需要更大吞吐时：允许横向扩展多机 worker（例如把 sandbox 做成独立服务，主进程通过 RPC/HTTP 投递执行任务；或用 Ray/Celery 之类分布式队列），保证并发数随机器数线性扩展。

【7) 输出数据格式（建议）】
每条保留下来的蒸馏样本保存：
- id, source
- prompt_input（可保存 question 或 context_messages）
- trace（完整标签串）
- num_tool_calls（<code> 次数）
- final_pass_all=true
- （可选）test_cases_hash / test_case_count（审计用，不把 test_cases 本文写进 trace）

隔离/资源限制建议的依据：

nsjail 明确是基于 Linux namespaces/cgroups/rlimits/seccomp 的轻量隔离工具，适合跑不可信代码。

resource.setrlimit() 可用于设置进程资源上限（软/硬限制）。

CodeJail 提供基于 AppArmor 的不可信 Python 执行隔离思路。