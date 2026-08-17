# Phase 4：Minimal Context Guard

## 状态

- 状态：Accepted（2026-08-17）
- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API smoke test：通过

Phase 4 的实现和验收已经完成。提交当前改动并创建 Git tag 后，即可冻结为 baseline。

## 唯一目标

无论历史执行轨迹多长，TinyHarness 在每次调用模型前，都必须构造一个协议合法、保留当前任务和最近执行证据、且不超过配置字符预算的 context；无法做到时，必须在 API 调用前明确失败。

```text
完整内存 messages
→ 验证 assistant/tool 协议
→ 计算 messages + tools 字符数
→ 删除最旧的完整工具交互块
→ 再次验证字符预算
→ Model Provider
```

未配置字符预算时保持 Phase 3 行为，不进行额外协议验证或裁剪。

## 配置入口

CLI 新增：

```text
--max-context-chars <positive integer>
```

它也作为 `agent_loop()` 的可选 `max_context_chars` 参数。小于 1 的值在 CLI 参数解析或 Agent Loop 启动时直接拒绝。

Phase 4 不选择厂商相关的默认 context window，因此该参数默认不启用。

## 字符预算口径

实现位于 `tiny_harness/runtime/context.py`。

`context_char_count()` 对以下对象使用 `ensure_ascii=False` 和紧凑 separators 进行 JSON 序列化，然后计算 Python 字符数：

```json
{"messages": [], "tools": []}
```

预算包含：

- 发给 Provider 的 messages
- 完整 tool schemas

预算不包含 SDK 添加的 model、HTTP headers 或其他请求字段。它不是 token 数，也不保证与任意模型的真实 context window 精确对应。

## 协议合法性

配置预算后，`prepare_context()` 将当前 TinyHarness 历史解析为：

```text
pinned prefix
+ assistant(tool_calls) / tool results block 1
+ assistant(tool_calls) / tool results block 2
+ ...
```

### Pinned prefix

第一个 assistant tool-call message 之前的所有消息作为不可删除前缀。当前 CLI 中它就是：

- system message
- 初始 user task

### 完整工具交互块

一个块由以下内容组成：

1. 一个包含非空 `tool_calls` 的 assistant message；
2. 按 tool call 原始顺序排列的全部 tool messages；
3. 每个 tool result 的 `tool_call_id` 与对应 call ID 一致。

多个 tool calls 及其 results 是同一个原子块。Phase 4 不允许只保留其中一部分。

以下情况抛出 `ContextProtocolError`：

- 孤立的 tool message
- assistant tool call 缺少结果
- tool results 缺失、额外或顺序不一致
- tool call ID 缺失或重复
- context 无法序列化为 JSON

错误发生在 Provider 调用前。

## 裁剪算法

`prepare_context(messages, tools, max_chars)`：

1. 验证参数和 assistant/tool 协议。
2. 计算完整 context 的 `before_chars`。
3. 如果超限，从最旧工具交互块开始逐块删除。
4. 始终保留 pinned prefix。
5. 如果存在工具历史，始终保留最新一个完整工具交互块。
6. 保留的历史必须是连续的最新块后缀。
7. 计算最终 `after_chars`。
8. 返回深拷贝，不修改完整内存 messages。

如果 pinned prefix、最新块和 tool schemas 仍然超过预算，抛出：

```text
ContextLimitError
```

Agent Loop 不调用 Provider，并通过 Phase 3 Event Log 记录 `run_failed`。

## Event Log

实际删除历史块时，在 `model_requested` 之前记录：

```text
context_trimmed
```

metadata：

- `turn`
- `before_chars`
- `after_chars`
- `dropped_blocks`
- `dropped_messages`

事件不保存 task、message、reasoning、工具参数或 tool result 正文。

未发生裁剪时不产生 `context_trimmed`。

## 完整历史与 Provider context

Agent Loop 继续在原始 `messages` 列表中追加完整 assistant 和 tool messages。裁剪只决定当前 Provider 调用看到的副本：

```text
完整历史：保留在当前进程内
Provider：收到满足预算的深拷贝
```

Phase 4 不持久化完整历史。进程退出后无法恢复。

## 自动测试

命令：

```powershell
cd D:\learn-claude-code\tinyharness
python -m unittest discover -s tests -v
```

当前结果：

- 执行 80 项测试
- 79 项通过
- 1 项跳过
- 跳过项仍是 Windows 环境无法创建符号链接

新增测试覆盖：

- messages 和 tools 使用同一确定性字符计数
- 刚好等于预算时不裁剪
- 返回深拷贝且不修改原消息
- 删除最旧完整工具交互块
- 只保留连续的最新块后缀
- 多个 tool calls 及其 results 不被拆开
- pinned prefix 和最新交互块不可删除
- 必保留内容超限时抛出 `ContextLimitError`
- 孤立、缺失或顺序错误的 tool result 被拒绝
- context 错误发生在 Provider 调用前
- `context_trimmed` 位于 `model_requested` 之前
- Event metadata 不包含消息正文
- CLI 正确传入和验证字符预算
- 未配置预算时保持 Phase 3 行为

## 明确不做

Phase 4 不实现：

- 精确 token 计算
- tokenizer 依赖
- 自动探测模型 context window
- LLM 摘要或语义压缩
- 单个 ToolResult 截断
- ToolResult 落盘
- Artifact Store
- Session Resume
- Memory
- Hooks
- Provider-specific context policy
- Context overflow 自动重试
- 修改现有 message representation
- 并发或流式工具执行

## 真实 API 验收结果

使用足够大的显式预算验证 CLI、Context Guard 与真实模型链路：

```powershell
python -m tiny_harness `
  "必须调用 list_files 工具列出 workspace 根目录，然后报告结果。" `
  --workspace D:\learn-claude-code\tinyharness `
  --max-turns 5 `
  --max-context-chars 100000 `
  --event-log D:\tinyharness-logs\phase4-smoke.jsonl
```

运行后读取：

```powershell
Get-Content D:\tinyharness-logs\phase4-smoke.jsonl
```

2026-08-17 使用真实 DeepSeek API 执行通过。日志共 8 条事件，验收结果：

- 真实 API 调用和 `list_files` 工具回填成功。
- 所有事件使用同一 run ID，sequence 为 1 到 8。
- `run_started` 包含 `max_context_chars: 100000`。
- 两轮模型调用均产生 requested/responded 事件。
- 日志以 `run_finished` 结束。
- 日志不包含 prompt、文件列表正文或最终答案正文。
- 本次 context 未超过预算，因此没有产生 `context_trimmed`。

裁剪边界已由使用 fake provider 的确定性自动测试验收。真实模型无法稳定保证生成固定长度和固定轮数，因此“真实 API 必须触发裁剪”不是 Phase 4 条件。

## 参考来源

Phase 4 没有读取 `learn-claude-code-main/s08_context_compact`，也没有查看 Claude Code 产品源码。算法仅根据 TinyHarness 当前同步消息结构和已确认的 Phase 4 验收目标设计。
