# 模型请求的上下文归因

[English](context-attribution.md) | [简体中文](context-attribution.zh-CN.md)

上下文归因用于解释模型请求中的空间由哪些内容占用：对话历史、工具结果、运行指导或工具 schema。
它可以帮助分析请求为何变大，但不能用来推算计费用量或决定裁剪策略。
离线[运行报告](../evals/swe_bench_lite/README.zh-CN.md#分析已有运行)会从事件日志汇总这些估算。

## 测量对象

`MODEL_REQUESTED.data.context_attribution`（`schema_version: 1`）在 `RecoveryExecutor`
每次实际尝试调用模型服务前，测量最终要发送的消息和工具 schema。主 Agent、子 Agent 的轮次，
以及通过该执行器转发的摘要和记忆调用都在范围内；绕过 `RecoveryExecutor` 直接调用模型服务
则不会记录归因。临时错误重试时会再次测量；因上下文长度错误而压缩后，会先重建请求再测量。
归因过程不会修改请求或保存的对话历史。

## 内容分类

- system/developer 消息归入 `system_runtime_guidance`，包括未标记的运行状态、
  工具使用效率指导和收尾指导。
- user 消息归入 `user_task_messages`；assistant 消息归入 `assistant_history`，
  其中包括函数名称、参数和推理字段。
- tool 消息若位于最后一条 assistant 消息之后，归入 `fresh_tool_results`，否则归入
  `historical_tool_results`。“新结果”表示后面还没有被接受的 assistant 响应，并不意味着
  模型服务从未读取它：失败或重试的请求可能已经读取过。
- 对于非 tool 消息，已知的 `name` 标记精确匹配时，优先于角色分类。
  技能清单、持久化记忆清单及相关记忆、上下文摘要及归档、待办状态各有独立分类。
- 工具 schema 归入 `tool_schemas`；无法识别的角色归入 `other_messages`。

工具结果还会按工具名汇总到 `tool_results_by_name`。匹配依据是前一个 assistant 批次中的
`tool_call_id`，而不是全局 ID 表。匹配缺失、有歧义或名称不安全时，记为 `unknown`。
名称仅允许 ASCII 字母、数字、下划线和连字符，长度不超过 64。汇总不包含参数、ID、路径或正文。
没有条目的分类会省略，等同于零。

没有标记的指导内容无法可靠地按来源拆分。通过工具返回的技能正文仍归入工具结果。
已经被摘要替代的结果无法还原，也无法再归因到原工具。如果自定义消息冒用了保留标记，
在没有额外来源信息的情况下，归因过程无法区分。

## 如何理解估算值

每个分类包含 `count`、`estimated_tokens` 和 `serialized_chars`。
字符数按紧凑 JSON 计算，包含消息或 schema 的结构字符，单位是 Unicode 字符而非 UTF-8 字节。
Token 使用配置的 `TokenMeter` 估算；如果使用校准后的计量器，则采用其底层启发式方法。
每个分类先对条目单独估算，扣除空请求的估算值，再求和；整个请求的总量另外测量。
`envelope_and_rounding_tokens` 是使分类之和与 `estimated_tokens` 对齐的有符号差值。
对于不满足可加性的自定义计量器，这个差值还包含条目之间相互影响产生的部分。

存在 `context_tokens` 元数据时，`calibration_adjustment_tokens` 记录带符号的校准差值，
使归因总量与校准后的请求估算对齐。校准量不会分摊到分类中，归因也不会修改校准结果。
这里估算的是输入上下文，并非精确的服务计费、缓存 token 统计、输出用量，也不包括
`tool_choice` 等传输选项。模型服务实际报告的用量见 `MODEL_RESPONDED` 字段。
归因结果出现在 JSONL 事件和离线报告中，不显示在终端进度里。

### assistant 历史的细分

`assistant_history_breakdown` 将内容分为 `reasoning_content`、`visible_content`、
`tool_calls` 和 `envelope_and_other`。它们是 `assistant_history` 内部的细分，
不能再次加入请求总量。四项始终存在，没有内容时也会保留。

估算器在每条 assistant 消息的副本上，依次移除非空的推理内容、可见正文和完整工具调用数据。
每一步减少的估算量归入对应项，其中包含字段序列化开销；舍入也按这一顺序处理。
空字段和其余元数据归入 `envelope_and_other`。四项之和等于 `assistant_history`，
不会暴露消息正文。参见[实现](../tiny_harness/context/attribution.py)和
[测试](../tests/test_context_attribution.py)。

## 实验用途：限制历史结果大小的模拟

目前没有实现实验性的请求裁剪。若要离线研究可能节省多少上下文，需要使用返回预设响应的模型替身，
并在每次构建请求前保存对话历史快照。每轮从快照构建两份请求：基线保持现状；实验副本完整保留
新工具结果，只限制历史工具结果的大小。调用 ID、结果顺序、所有非结果消息、运行时加入请求的内容、
工具和收尾状态都必须保留。使用同一个未经校准的计量器估算两份完整请求，并针对几种限制值报告
逐轮总量、新结果与历史结果的总量，以及差值。不能从校准后的总量中直接减去原始文本长度来推算节省量。

尤其需要注意，当前 `model_context_messages()` 会缩短过大的 `read_file` 结果，即使它是新结果。
实验必须从保存的对话历史快照开始，而不能使用已经整理过的请求，否则无法取得完整的新结果。
如果保存的历史已经压缩过，其中可能只剩预览或摘要；要还原原始内容，需要压缩前的测试数据或保留的文件。

只含计数的 JSONL 事件无法重建正文，也无法精确估算任意保留首尾内容的方案。
应使用可控的离线测试数据，不要为此把正文加入事件。如果还要模拟压缩阈值，两条实验分支需要
各自维护历史，并使用预设的摘要结果。固定轨迹模拟无法预测模型实际会如何改变决策、后续工具调用
或计费 token；真实模型对比需要另行开展。
