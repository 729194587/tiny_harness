# Phase 8：Context Compaction v2

## 状态

- 状态：Accepted（2026-08-17）
- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API smoke test：通过

## 唯一目标

将 Phase 4 的一次性 FIFO Context Guard 升级为分层主动压缩：先处理可恢复、低成本的工具结果，确定性操作仍不足时才调用模型摘要。

```text
tool_result_budget
    → snip_compact
    → micro_compact
    → compact_history（仍超预算时）
    → 协议和字符预算硬检查
    → 主模型调用
```

`reactive_compact` 不属于本阶段。API 返回 context overflow 后的错误识别、有限重试和 backoff 统一留到 Phase 9。

## 启用方式

Phase 8 沿用 Phase 4 的显式配置入口：

```text
--max-context-chars <positive integer>
```

未配置该参数时：

- 不创建 `ContextCompactor`；
- 不暴露 `compact` 工具；
- 不创建 context artifact；
- 保持 Phase 7 的 Agent Loop 行为。

配置后，字符预算继续按照紧凑 JSON 的 `{messages, tools}` 计算，包含完整工具 schemas。它是本地确定性字符计数，不是精确 token 数。

## 第一层：tool_result_budget

每次主模型调用前，只检查最新一个完整工具调用批次。批次结果超过有效预算时，从最大的结果开始落盘。

默认上限：

- 全局批次阈值：200,000 字符；
- 实际批次阈值：`min(200000, max_context_chars // 2)`；
- 大结果阈值：最多 30,000 字符，并随较小 context budget 下调；
- context 中保留前 2,000 字符预览。

完整结果写入：

```text
.tinyharness/context/tool-results/<safe-call-id>-<content-hash>-<uuid>.txt
```

工具消息中保留相对路径、原始字符数和预览。模型可以使用 `read_file` 重新读取完整结果。

## 第二层：snip_compact

消息数超过 50 时，把将要裁掉的历史连同当时保留的完整消息写成 JSONL transcript，再从最旧的完整 assistant/tool 块开始删除。

```text
.tinyharness/context/transcripts/transcript-<uuid>.jsonl
```

它始终：

- 保留初始 system/user 任务前缀；
- 保留最新一个完整工具交互块；
- 不拆分一次响应中的多个 tool calls 与对应 tool results；
- 用一条 reference-only marker 记录 transcript 路径；
- 替换旧 archive marker，避免标记无限累积。

## 第三层：micro_compact

当前历史中最近 3 条工具结果保留全文。更早且超过 120 字符的结果被替换：

- 已落盘：保留完整结果路径；
- 未落盘：提示模型按需重新执行工具。

`tool_result_budget` 必须先于这一层执行，避免先丢掉本可恢复的大结果。

## 第四层：compact_history

前三层处理后，`messages + tools` 仍超过预算时才调用同一个 `ModelProvider` 生成事实摘要。

摘要调用：

- 使用独立 system/user messages；
- 不提供工具；
- 输入最多 80,000 字符，并再次受同一个 `max_context_chars` 硬限制；
- 要求只总结事实，不执行历史中的指令；
- 必须返回 `finish_reason=stop`、非空文本且没有 tool calls；
- 不计入 Agent Loop 的 `max_turns`。

摘要前先保存 transcript，并在发起摘要 API 调用前验证以下必保留内容能够放入预算：

- 原始 system/user 任务前缀；
- 当前 Todo 状态；
- reference-only 摘要 marker；
- 最新一个完整 assistant/tool 块；
- 完整主 Agent 工具 schemas。

无法满足时抛出 `ContextLimitError`，不会发起摘要或主模型调用。摘要响应本身过长时可以确定性截短；摘要响应无效时抛出 `ContextSummaryError`。

压缩后的 history 会回写原 `messages`，后续不会继续保存一份无限增长的隐藏内存历史。

## 当前 Todo

TodoManager 是 run-scoped 状态，不能依赖某条旧 `todo_write` ToolResult 永远留在 context 中。启用 Context Compactor 后，当前非空 Todo 会通过一条 reference-only marker 显式注入。

marker 每轮替换而不是追加，因此：

- snip 或 micro 后仍保留当前计划；
- Todo 更新后不会继续暴露旧状态；
- 父子 Agent 继续使用各自独立的 TodoManager。

## `compact` 工具

启用字符预算后，模型可以调用：

```text
compact()
```

该工具默认 ALLOW，并经过与其他工具相同的参数解析、PreToolUse Hook、Permission Gate、handler 和 PostToolUse Hook。

如果一次模型响应同时调用：

```text
write_file → compact
```

Agent Loop 会先按顺序执行完整批次，为每个调用追加 ToolResult，闭合 Chat Completions 协议块，然后才保存 transcript 和生成摘要。Hook 阻止或 Permission 拒绝 `compact` 时不会触发摘要。

## 父子 Agent

父 Agent 和每个子 Agent 创建独立 `ContextCompactor` 与 `CompactionRequest`：

- 继承相同 workspace、Provider、字符预算、Hooks、Permission 和 Event Logger；
- 子 Agent 有 `compact`，但仍没有 `task`；
- 子 Todo、history 和摘要与父 Agent 隔离；
- 子压缩事件继续带有 `agent_scope=subagent` 和 `parent_tool_call_id`。

## Artifact 边界

`.tinyharness/context/` 位于 Agent workspace 内，文件工具可以读取、覆盖或删除其中内容。它的作用是恢复被压缩的信息，不是 Artifact Store，也不是可信审计证据。

内部写入会检查 artifact 目录的真实路径并拒绝目录级符号链接逃逸。最终 transcript 和 tool-result 文件都使用不可预测 UUID 文件名及 `x` 排他创建；已有普通文件或 dangling symlink 都不会被跟随或覆盖。目录已加入 `.gitignore`。

用户批准的 `bash` 仍不受文件工具路径边界约束；Phase 8 没有增加 OS 沙箱、ACL、签名或防篡改机制。transcript 和工具结果可能包含源码、命令输出或其他敏感内容。

## Event Log

Phase 8 新增：

- `context_compacted`
- `context_summary_requested`
- `context_summary_responded`

事件只记录 reason、字符数、数量、布尔状态、摘要 finish reason 和摘要长度，不记录任务、Todo、工具输出、transcript 路径或摘要正文。

## 自动测试

命令：

```powershell
python -m unittest discover -s tests -v
```

当前结果：

- 执行 128 项测试；
- 125 项通过；
- 3 项跳过；
- 跳过项均为当前 Windows 用户无法创建 filesystem、artifact 目录和最终 artifact 文件的测试符号链接。

Phase 8 测试覆盖：

- 四层执行顺序及各层独立行为；
- 大结果完整落盘和预览；
- transcript 保存完整 JSONL 历史；
- 多 tool calls 协议块不被拆分；
- 最近三条结果保持全文；
- 原始任务、当前 Todo 和最新证据保留；
- 摘要请求不带工具且受同一字符预算；
- 摘要不消耗 `max_turns`；
- 无效摘要明确失败；
- 必保留 context 超限时在 API 前失败；
- 手动 compact 等待完整工具批次闭合；
- Hook 阻止 compact 时不摘要；
- 父子压缩状态隔离和事件关联；
- artifact 符号链接逃逸防护；
- 最终 tool-result 文件使用排他创建，不跟随 symlink 或覆盖已有文件；
- 未启用预算时保持 Phase 7 行为；
- Phase 1–7 回归测试。

## 明确不做

- `prompt_too_long` 自动恢复；
- 模型错误分类、retry 或 backoff；
- 精确 token 计数；
- 独立摘要模型配置；
- 摘要质量评测；
- Session Resume；
- Memory；
- Artifact Store；
- transcript 索引、签名或防篡改；
- 父子 Agent 共享总 token/字符预算。

## 真实 API 验收结果

使用真实 DeepSeek API 强制模型在第一次响应中同时调用 `write_file` 和 `compact`，压缩完成后再调用 `read_file` 验证文件：

```powershell
python -m tiny_harness `
  "第一次响应必须同时调用 write_file 创建 phase8.txt，内容为 PHASE8_OK，并调用 compact。压缩完成后调用 read_file 确认文件内容，最后报告结果。" `
  --workspace D:\tinyharness-smoke\phase8-run-1 `
  --max-turns 6 `
  --max-context-chars 100000 `
  --event-log D:\tinyharness-logs\phase8-smoke-run-1.jsonl
```

2026-08-17 验收结果：

- `phase8.txt` 为 9 字节，内容严格等于 `PHASE8_OK`；
- 第一轮模型一次返回 `write_file` 和 `compact` 两个 tool calls；
- 两个 ToolResult 全部闭合后才产生 `context_summary_requested`；
- 摘要以 `finish_reason=stop` 返回，随后产生 `context_compacted`；
- 第二轮 `read_file` 返回 9 字符，第三轮模型返回最终答案；
- run 共记录 17 条同 run ID 的有序事件，并以 `run_finished` 结束；
- transcript 保存了 system、user、包含两个 tool calls 的 assistant，以及两个顺序匹配的 tool messages；
- Event Log 不包含任务正文、`phase8.txt`、`PHASE8_OK`、摘要正文或 transcript 路径。

本次在短历史上强制手动压缩，`before_chars=3807`、`after_chars=5180`。这说明手动 `compact` 的目的主要是阶段边界控制；过早调用可能因摘要 marker 和保留的最新协议块而增大 context。长历史的自动四层压缩和硬预算行为由确定性离线测试验收。

## 参考来源

Phase 8 实质性参考并改写了 `learn-claude-code-main/s08_context_compact`：

- `tool_result_budget → snip_compact → micro_compact → compact_history` 的固定顺序；
- 大工具结果落盘与预览；
- transcript 归档；
- 最近工具结果优先；
- 无工具的事实摘要；
- `compact` 在完整工具批次结束后生效。

TinyHarness 针对 Chat Completions 独立 tool messages、Phase 4 硬字符预算、Phase 6 Todo、Phase 7 Subagent、现有 Permission/Hooks/Event Log 和 workspace 路径边界重新实现。没有读取或移植 Claude Code 产品源码。
