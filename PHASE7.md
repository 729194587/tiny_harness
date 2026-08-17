# Phase 7：Minimal Subagent

## 状态

- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API smoke test：通过
- 状态：Accepted（2026-08-17）

Phase 7 的设计、实现和验收已经完成。提交当前改动并创建 `phase-7-baseline` Git tag 后，即可冻结为 baseline。

## 唯一目标

让主 Agent 通过 `task` 工具同步运行一个具有全新消息上下文的子 Agent，并且只把子 Agent 的最终文本作为 ToolResult 返回父对话：

```text
Parent Agent messages
→ task(prompt)
→ fresh child messages
→ Child Agent Loop
→ child final text
→ task ToolResult
→ Parent Agent messages
```

隔离的是消息和 Todo 状态，不是进程或文件系统。

## task 工具

工具实现位于 `tiny_harness/tools/task.py`，schema 注册在统一 Tool Registry。它只接收一个非空字符串 `prompt`。运行时会去除 prompt 首尾空白并再次验证，不能只依赖模型遵守 JSON Schema。

`task` 默认权限为 `ALLOW`，但不会绕过现有控制流：

```text
PreToolUse Hooks
→ Permission Gate
→ synchronous Subagent runner
→ child final text or error ToolResult
→ PostToolUse Hooks
```

Pre Hook 阻止 `task` 时不会启动子 Agent。Post Hook 可以观察返回父 Agent 的最终摘要或错误 ToolResult。

## 父子共享与隔离

父子共享：

- 同一个 ModelProvider
- 同一个 workspace
- 同一个 Permission Policy 和 Permission Prompt
- 同一组 Tool Hooks
- 同一个底层 Event Logger
- 相同的 `max_context_chars` 配置值

子 Agent 不继承：

- 父 system prompt
- 原始用户任务
- 父消息历史
- 父 reasoning
- 父 tool calls 和 tool results
- 父 TodoManager 状态

子 Agent 初始上下文只有独立 system message 和 `task.prompt` 对应的 user message。子 Agent 的完整中间历史只存在于嵌套 Agent Loop；父 Agent 最终只看到 `task` 调用和一个关联的 ToolResult。

文件修改是共享副作用。子 Agent 创建或编辑的 workspace 文件会立即对父 Agent 可见。

## 子工具与递归限制

父 Agent 可用七个工具：

```text
read_file
write_file
edit_file
list_files
bash
todo_write
task
```

子 Agent 可用前六个工具，不包含 `task`。

递归限制同时在两层执行：

1. 子 Agent 收到的 tool schemas 不包含 `task`
2. 子 Runtime 没有 Subagent runner，即使模型虚构 `task` 调用，也返回 `Unknown tool: task`

Phase 7 只允许一层委派。

## 同步执行与多 task 调用

`task` 是普通同步工具。父 Agent 等待子 Agent 完成后才得到 ToolResult。同一次父模型响应包含多个 `task` 调用时，沿用 TinyHarness 的顺序工具语义：

```text
task A child run
→ task A result
→ task B child run
→ task B result
```

每个子 Agent 都从独立的 fresh messages 和 TodoManager 开始。Phase 7 不并发执行子 Agent。

## 资源限制

CLI 新增：

```text
--subagent-max-turns 10
```

每个子 Agent 独立获得最多 10 次模型调用。父 Agent 的 `--max-turns` 不会被子调用扣减。

子 Agent 使用与父 Agent 相同的 `--max-context-chars` 数值，但 Context Guard 在子 Agent 自己的消息历史和不含 `task` 的子工具 schema 上单独计算。

Phase 7 没有父子共享总 token budget，也没有限制一次父 run 可以创建多少个顺序子 Agent。顶层 `max_turns` 间接限制父模型能发起的委派轮次，但不是全局成本上限。

## Event Log 关联

`ScopedEventLogger` 将所有子事件写入同一个底层有序日志，并固定附加：

```text
agent_scope: subagent
parent_tool_call_id: <task call id>
```

成功顺序：

```text
tool_started(task)                  # parent scope
→ run_started                      # subagent scope
→ model_requested / responded      # subagent scope
→ child tool events                # subagent scope
→ run_finished                     # subagent scope
→ tool_finished(task)              # parent scope
```

日志使用顶层 JsonlEventLogger 的同一个 `run_id` 和单调递增 sequence。`agent_scope` 用于区分嵌套的 `run_started` / `run_finished`。

Event Log 不记录 delegated prompt、子消息、工具参数、工具正文或子最终文本。

## 失败传播

子 Agent 的普通失败会先生成带 scope 的 `run_failed`，然后由父 `task` dispatch 转换成错误 ToolResult：

```text
run_failed(subagent)
→ tool_finished(task, outcome=error)
→ Error ToolResult returned to parent model
```

普通失败包括 Provider 异常、Context 错误、HookExecutionError 和达到 `subagent_max_turns`。子工具自己的普通 handler 异常仍先转换为子 ToolResult，由子模型决定如何继续。

`EventLogError` 保持 fail-closed：子日志写入失败不会被转换成普通 task 错误，而是终止整个父 run。`KeyboardInterrupt` 和 `SystemExit` 同样不会被吞掉。

## 终端行为

每次委派显示：

```text
[Subagent started]
[Subagent done]
```

普通失败显示 `[Subagent failed]`。子 Agent 使用 `todo_write` 时仍会显示自己的 `## Current Tasks`。Phase 7 不额外打印每个子工具结果正文，避免把大输出或文件内容复制到终端。

## 自动测试

命令：

```powershell
cd D:\learn-claude-code\tinyharness
python -m unittest discover -s tests -v
```

当前结果：

- 执行 121 项测试
- 120 项通过
- 1 项跳过
- 跳过项仍是 Windows 环境无法创建符号链接

新增覆盖：

- 父 schema 包含 `task`，子 schema 不包含
- task prompt 运行时验证
- 子上下文不包含父 system、task 或 reasoning
- 父上下文只接收子最终文本
- 子中间 tool calls、tool results 和 reasoning 不进入父历史
- 父子共享 workspace 副作用
- 父子 TodoManager 隔离
- 子工具继承 Permission 和 Hooks
- Pre Hook 在子启动前阻止 task
- 虚构的子 task 调用不能递归
- 子 max turns 失败转换为父 ToolResult
- 子失败事件先于父 task error result
- 子 EventLogError 终止父 run
- 所有子事件关联 `parent_tool_call_id`
- 多个 task 顺序运行且上下文互相隔离
- CLI 参数解析和默认值
- Phase 1–6 原有测试继续通过

## 明确不做

- 并行 Subagent
- Background Task
- 多层递归委派
- Agent Teams
- 子 Agent Session Resume
- 子 Agent 消息持久化
- 父子共享 Todo
- 单独 Provider、模型、API key 或 system role 配置
- 全局 token / cost / model-call budget
- 子 Agent 超时取消
- 子 Agent 工作区隔离、进程隔离或 OS sandbox
- Workflow / DAG scheduler

## 真实 API 验收结果

建议继续使用独立 smoke workspace 和 workspace 外 Event Log：

```powershell
$smokeWorkspace = "D:\tinyharness-smoke\phase7-run-1"
$smokeLog = "D:\tinyharness-logs\phase7-smoke-run-1.jsonl"

New-Item -ItemType Directory -Force $smokeWorkspace

cd D:\learn-claude-code\tinyharness
python -m tiny_harness `
  "必须调用 task，把以下完整任务委派给子 Agent：在 workspace 创建 phase7-child.txt，内容必须恰好为 SUBAGENT_OK，然后读取该文件确认内容并返回简短结论。父 Agent 不得直接调用文件工具；收到 task 结果后报告结论。" `
  --workspace $smokeWorkspace `
  --max-turns 8 `
  --subagent-max-turns 8 `
  --max-context-chars 100000 `
  --event-log $smokeLog
```

检查：

```powershell
Get-Content "$smokeWorkspace\phase7-child.txt"

Get-Content $smokeLog |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Format-Table sequence, event_type, @{
    Label = "scope"
    Expression = { $_.data.agent_scope }
  }, @{
    Label = "tool"
    Expression = { $_.data.tool_name }
  }, @{
    Label = "parent_call"
    Expression = { $_.data.parent_tool_call_id }
  }
```

2026-08-17 验收结果：

- 终端依次出现 `[Subagent started]` 和 `[Subagent done]`
- `phase7-child.txt` 恰好为 11 字节 `SUBAGENT_OK`，没有尾随换行或空格
- Event Log 共 20 条，属于同一个 `run_id`，sequence 从 1 连续递增到 20
- 父级 sequence 4 为未带 `agent_scope` 的 `tool_started(task)`
- sequence 5–16 是子生命周期，全部带 `agent_scope=subagent` 和相同 `parent_tool_call_id`
- 子 Agent 依次完成 `write_file` 和 `read_file`，两个 outcome 均为 `returned`
- sequence 16 的子 `run_finished` 先于 sequence 17 的父 `tool_finished(task)`
- 父 Agent 没有直接调用文件工具，最终 sequence 20 为父 `run_finished`
- 日志仅包含预期 metadata，不包含 delegated prompt、文件路径、`SUBAGENT_OK` 或最终回答正文

## 参考来源

Phase 7 实现前查看了：

- `learn-claude-code-main/s06_subagent/code.py`
- `learn-claude-code-main/s06_subagent/README.zh.md`

`task` schema、fresh child messages、同步嵌套 Agent Loop、共享 workspace、只返回最终文本以及子工具移除 `task` 的控制流实质性参考并改写自 s06。

TinyHarness 额外把 Permission、Hooks、Todo、Context Guard、独立 child max turns、错误 ToolResult 和 Event Log correlation 接入该控制流。没有查看或移植 Claude Code 产品源码。
