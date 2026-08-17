# Phase 5：Minimal Tool Hooks

## 状态

- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API PreToolUse smoke test：通过
- 真实 API PostToolUse smoke test：通过
- 状态：Accepted（2026-08-17）

Phase 5 的设计、实现和验收已经完成。提交当前改动并创建 Git tag 后，即可冻结为 baseline。

## 唯一目标

为每个已经解析的工具调用提供同步、run-scoped、顺序确定的 PreToolUse 和 PostToolUse 扩展点，同时保持 Permission Gate、每个 tool call 对应一个 ToolResult、多工具调用顺序和现有 Event Log 语义不变。

```text
ToolCall
→ 工具查找和 JSON 参数解析
→ PreToolUse Hooks
→ Permission Gate
→ Handler
→ ToolResult
→ PostToolUse Hooks
→ Agent Loop
```

Phase 5 只实现 Tool Hooks，不实现 UserPromptSubmit 或 Stop Hook。

## 数据结构

实现位于 `tiny_harness/runtime/hooks.py`。

### `ToolHookContext`

```text
tool_call_id
tool_name
arguments
```

arguments 是解析后的只读顶层 Mapping，并且与实际传给 handler 的 dict 隔离。Pre Hook 不能修改实际执行参数。

### `HookBlock`

Pre Hook 只能返回：

- `None`：继续
- `HookBlock(reason)`：阻止

其他非空返回值违反 Hook contract，包装成 `HookExecutionError`。

### `ToolHooks`

一个具体的 run-scoped 注册表：

```python
hooks = ToolHooks()
hooks.register_pre(pre_hook)
hooks.register_post(post_hook)
```

- Pre/Post 各自按注册顺序运行。
- 不使用全局 Hook 注册表。
- `agent_loop()` 未传入 `tool_hooks` 时行为不变。
- callback 必须可调用。

### `HookExecutionError`

包含：

- `stage`：`pre` 或 `post`
- `hook_index`：从 1 开始的注册顺序
- `error_type`：原始异常类型

异常正文不写入 Event Log。

## PreToolUse

Pre Hooks 在以下步骤之后运行：

1. 工具存在性检查
2. `arguments_json` 解析
3. 参数顶层 object 检查

并在 Permission Gate 之前运行。

### Continue

返回 `None`，继续下一个 Pre Hook；全部通过后才进入 Permission。

### Block

第一个 `HookBlock` 会：

- 停止后续 Pre Hooks
- 不调用 Permission Policy 或 Prompt
- 不执行 handler
- 不执行 Post Hooks
- 生成保留原 `tool_call_id` 的 ToolResult
- 不影响同一次模型响应中的其他 tool calls

ToolResult：

```text
Error: Tool call blocked by PreToolUse hook: <reason>
```

reason 可能提供给模型，但不写入 Event Log。

### Exception

Pre Hook 普通异常包装成 `HookExecutionError`：

- handler 不执行
- 记录 `tool_hook_failed`
- Agent Loop 记录 `run_failed`
- 异常继续向调用方传播

`KeyboardInterrupt` 和 `SystemExit` 不作为普通 Hook 异常吞掉。

## Permission Gate

Phase 2 Permission 保持独立，不移动到 Hook：

```text
PreToolUse
→ Permission
→ Handler
```

Permission deny 不运行 handler 或 Post Hooks，继续产生原有 `tool_denied` 和拒绝 ToolResult。

## Handler 与 PostToolUse

handler 正常返回或普通异常被转换成 ToolResult 后：

1. 记录 `tool_finished`
2. 依注册顺序运行 Post Hooks
3. 返回原 ToolResult

因此 Post Hook 可以观察：

- 正常 ToolResult
- handler 错误 ToolResult

但不观察：

- 未知工具
- JSON 解析错误
- Permission deny
- Pre Hook block

每个 Post Hook 接收 ToolResult 的深拷贝。Hook 修改该对象不会改变实际回填模型的结果，也不会影响后续 Post Hooks。

Post Hook 非 `None` 返回值或普通异常包装成 `HookExecutionError`。此时 handler 可能已经产生副作用；Phase 5 不回滚，也不把 Hook 失败伪装成 handler 失败。

## Event Log

新增：

### `tool_hook_blocked`

```text
tool_call_id
tool_name
hook_index
```

不记录 reason 和 arguments。

### `tool_hook_failed`

```text
tool_call_id
tool_name
stage
hook_index
error_type
```

不记录异常正文和 arguments。

成功 Hook 不产生独立事件，避免正常日志过于嘈杂。

Post Hook 失败时顺序为：

```text
tool_started
→ tool_finished
→ tool_hook_failed
→ run_failed
```

## 多工具调用

同一模型响应中的每个 ToolCall 独立执行完整 Hook / Permission / Handler 流程。

一个调用被 HookBlock 阻止，不会跳过其后的调用。全部 ToolResults 继续按模型原始调用顺序回填，因此 Phase 4 Context Guard 仍能验证完整 assistant/tool block。

## CLI 与示例

主 CLI 不新增 Hook 配置参数。Hook 是 Python API 扩展点。

`examples/hooks_demo.py` 注册：

- 一个阻止 `write_file` 的 Pre Hook
- 一个报告 ToolResult 字符数的 Post Hook

该示例使用与主 CLI 相同的 DeepSeek 默认模型配置和环境变量。

## 自动测试

命令：

```powershell
cd D:\learn-claude-code\tinyharness
python -m unittest discover -s tests -v
```

当前结果：

- 执行 91 项测试
- 90 项通过
- 1 项跳过
- 跳过项仍是 Windows 环境无法创建符号链接

新增测试覆盖：

- Pre/Post 注册顺序
- 第一个 HookBlock 停止后续 Pre Hooks
- block 跳过 Permission Prompt 和 handler
- block 保留 tool call ID
- block reason 不进入 Event Log
- Permission deny 不运行 Post Hook
- Post Hook 观察正常和错误 ToolResult
- Post Hook 不能修改实际 ToolResult
- Pre Hook 不能修改实际 handler 参数
- Pre Hook 异常发生在 handler 前
- Post Hook 异常发生在 `tool_finished` 后
- 非法 Hook 返回值明确失败
- 一个 tool call 被阻止不影响同批其他调用
- block 后消息仍通过 Phase 4 协议验证
- Hook 失败产生 `run_failed`
- Phase 1–4 原有测试继续通过

## 明确不做

- UserPromptSubmit Hook
- Stop Hook
- Permission-as-Hook
- 参数修改
- ToolResult 修改
- Hook matcher
- Hook 配置文件
- Shell Hook
- 异步或并发 Hook
- 插件系统
- Hook 持久化
- Hook retry
- Claude Code 产品实现

## 真实 API 验收结果

### PreToolUse block

```powershell
python -m examples.hooks_demo `
  "必须调用 write_file 创建 phase5-blocked.txt；如果工具返回 blocked，不要重试，直接说明被 Hook 阻止。" `
  --workspace D:\learn-claude-code\tinyharness `
  --event-log D:\tinyharness-logs\phase5-pre-smoke.jsonl
```

检查：

```powershell
Test-Path D:\learn-claude-code\tinyharness\phase5-blocked.txt
Get-Content D:\tinyharness-logs\phase5-pre-smoke.jsonl
```

2026-08-17 验收结果：

- 模型确实请求了 `write_file`，ToolResult 告知模型调用被 Hook 阻止
- `phase5-blocked.txt` 未创建
- 同一 `run_id` 的事件序号为 1–7，包含一次 `tool_hook_blocked`
- 被阻止的调用没有 `tool_started` 或 `tool_finished`
- 日志不包含 block reason、文件内容或最终答案正文
- 模型没有重试，并以 `run_finished` 正常结束

### PostToolUse observe

```powershell
python -m examples.hooks_demo `
  "必须调用 list_files 列出 workspace 根目录，然后报告结果。" `
  --workspace D:\learn-claude-code\tinyharness `
  --event-log D:\tinyharness-logs\phase5-post-smoke.jsonl
```

2026-08-17 验收结果：

- 终端出现 `[PostToolUse] list_files returned 138 characters`
- 模型收到 list_files ToolResult 并正常回答
- 同一 `run_id` 的事件序号为 1–8，`tool_started` 先于 `tool_finished`
- `tool_finished` 的 outcome 为 `returned`，content length 为 138
- 日志不包含工具结果正文或最终答案正文，并以 `run_finished` 结束

## 参考来源

Phase 5 实现前只查看了：

- `learn-claude-code-main/s04_hooks/code.py`
- `learn-claude-code-main/s04_hooks/README.zh.md`

TinyHarness 借鉴了有序注册、PreToolUse 阻止和 PostToolUse 观察的核心控制流。没有复制教学版的全局 HOOKS、Permission Hook、UserPromptSubmit 或 Stop Hook，也没有查看 Claude Code 产品源码。
