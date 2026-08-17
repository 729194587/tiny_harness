# Phase 3：Minimal Execution Event Log

## 状态

- 状态：Accepted（2026-08-17）
- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API smoke test：通过

Phase 3 的实现和验收已经完成。提交当前改动并创建 Git tag 后，即可冻结为 baseline。

## 唯一目标

为现有 Agent 链路增加可选、同步、持久化的控制流事件日志：

```text
Run
→ Model lifecycle
→ Tool / Permission lifecycle
→ Run outcome
→ JSONL
```

Event Log 的目标是回答“发生了什么以及顺序是什么”，不是保存会话内容或实现恢复。

## 启用方式

CLI 新增：

```text
--event-log <path>
```

未指定时使用 `NullEventLogger`，不创建文件，也不改变 Phase 2 行为。

指定时使用 `JsonlEventLogger`：

- append 模式
- 自动创建父目录
- 每个 logger 生成独立 run ID
- 每次 run 的 sequence 从 1 开始
- 每个事件写入一行 JSON
- 每次写入立即 flush
- 日志写入失败抛出 `EventLogError`

建议把日志写到 Agent workspace 外部，例如 `D:\tinyharness-logs\events.jsonl`。这样 workspace 内的 `write_file` 和 `edit_file` 无法修改日志。

Event Log 是独立于最终回答的 observability trace，不是不可篡改的 trusted audit log。Phase 2 没有 OS 级沙箱，用户批准的 `bash` 命令仍可能访问 workspace 外部路径。Phase 3 不为此引入 protected paths、日志签名、append-only ACL 或 sandbox。

## 数据结构

实现位于 `tiny_harness/runtime/events.py`。

### `Event`

```text
run_id
sequence
timestamp
event_type
data
```

- `run_id`：关联同一次运行中的所有事件
- `sequence`：从 1 开始严格递增
- `timestamp`：UTC ISO 8601 时间
- `event_type`：固定生命周期事件类型
- `data`：事件所需的最小 metadata

### `EventLogger`

同步 Protocol：

```python
emit(event_type, data=None) -> None
```

Phase 3 包含：

- `NullEventLogger`
- `JsonlEventLogger`

没有 Composite logger、异步队列、后台写线程或数据库。

## 事件类型

### `run_started`

记录：

- `max_turns`

### `model_requested`

记录：

- `turn`

### `model_responded`

记录：

- `turn`
- `finish_reason`
- `tool_call_count`
- `content_length`

### `tool_started`

handler 执行前记录：

- `tool_call_id`
- `tool_name`

如果这个事件无法写入，handler 不执行。

### `tool_denied`

Permission 最终结果为 DENY 时记录：

- `tool_call_id`
- `tool_name`

DENY 不产生 `tool_started` 或 `tool_finished`。

### `tool_finished`

记录：

- `tool_call_id`
- `tool_name`
- `outcome`：`returned` 或 `error`
- `content_length`

`returned` 表示 handler 正常返回，不表示 shell exit code 一定为 0。Phase 3 不重新设计结构化 ToolResult。

未知工具或参数解析失败可能直接产生 `tool_finished outcome=error`，此前没有 `tool_started`，因为 handler 从未开始。

### `run_finished`

正常最终回答时记录：

- `turns`
- `answer_length`

### `run_failed`

模型错误、异常停止或 `max_turns` 时记录：

- `turn`
- `error_type`

事件日志自身失败时不尝试继续写 `run_failed`，而是直接传播 `EventLogError`。

## 不记录的内容

Phase 3 默认不记录：

- API Key
- 完整 task / prompt
- messages
- `reasoning_content`
- 工具 arguments
- bash command
- 文件路径或文件内容
- 完整 ToolResult
- 最终答案正文

只记录控制流 metadata 和长度。这使日志可以证明工具是否开始、拒绝、完成或失败，同时避免提前变成 Context Store 或 Artifact Store。

## 失败语义

Event Log 只有显式启用后才可能影响运行。

- `run_started` 写入失败：模型不调用。
- `tool_started` 写入失败：handler 不执行。
- `tool_finished` 写入失败：handler 可能已经产生副作用，随后运行以 `EventLogError` 终止。
- JSON 序列化、目录创建或文件写入失败：不转成普通 ToolResult。
- Event Log 错误不被 registry 的工具异常处理吞掉。

Phase 3 不实现事务、回滚、日志防篡改或 exactly-once 副作用保证。

## 自动测试

命令：

```powershell
cd D:\learn-claude-code\tinyharness
python -m unittest discover -s tests -v
```

当前结果：

- 执行 65 项测试
- 64 项通过
- 1 项跳过
- 跳过项仍是 Windows 环境无法创建符号链接

新增测试覆盖：

- JSONL 可解析
- 同一 run ID
- sequence 严格递增
- 多次 run append 且 sequence 各自从 1 开始
- 自动创建父目录
- 不可序列化数据和非法目标路径抛出 `EventLogError`
- 未启用时 no-op
- 正常 run/model 生命周期
- ALLOW 工具产生 started / finished
- 同一次模型响应的多个工具调用按实际执行顺序产生事件
- DENY 工具只产生 denied
- 工具异常产生 error outcome
- 模型异常和 max_turns 产生 run_failed
- 日志失败发生在首次事件时，模型不调用
- prompt、reasoning、工具内容和最终答案不写入 JSONL
- CLI 正确注入 JSONL logger

## 明确不做

Phase 3 不实现：

- Hooks
- Event Replay
- Session Resume
- 完整消息持久化
- Event 查询 API
- 数据库
- 日志轮转或清理策略
- Artifact Store
- Context Management
- token budget
- 并发或流式工具执行
- OpenTelemetry
- Web UI

## 真实 API 验收结果

使用无副作用 bash 命令，同时启用事件日志：

```powershell
python -m tiny_harness `
  "必须调用 bash 工具执行命令 cd，并报告输出。" `
  --workspace D:\learn-claude-code\tinyharness `
  --max-turns 5 `
  --event-log D:\tinyharness-logs\phase3-smoke.jsonl
```

在 ASK 中输入 `y`。运行后读取：

```powershell
Get-Content D:\tinyharness-logs\phase3-smoke.jsonl
```

2026-08-17 使用真实 DeepSeek API 执行通过。日志共 8 条事件，验收结果：

- 每行均为合法 JSON。
- 所有事件使用同一 run ID。
- sequence 为 1 到 8，严格递增。
- 两轮调用均出现 `model_requested` / `model_responded`。
- bash 的 `tool_started` 位于 `tool_finished` 之前。
- 最后一个事件为 `run_finished`。
- 日志中没有 `cd` 命令正文、工具输出或模型最终答案正文。

## 参考来源

Phase 3 没有读取 learn-claude-code 后续章节实现，也没有查看 Claude Code 产品源码。事件类型和 JSONL 结构根据 TinyHarness Phase 1/2 smoke test 中缺少独立控制流证据的问题设计。
