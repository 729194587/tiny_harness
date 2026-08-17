# Phase 6：Minimal Todo / Agent Plan

## 状态

- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API smoke test：通过
- 状态：Accepted（2026-08-17）

Phase 6 的设计、实现和验收已经完成。提交当前改动并创建 `phase-6-baseline` Git tag 后，即可冻结为 baseline。

## 唯一目标

让 Agent 在一次运行中通过 `todo_write` 维护结构化的任务列表，并在长时间未更新计划时收到 Harness 提醒：

```text
Model
→ todo_write
→ PreToolUse Hooks
→ Permission Gate
→ TodoManager.update
→ PostToolUse Hooks
→ rendered ToolResult
→ Model
```

Todo 增加规划和进度跟踪能力，不增加文件、Shell 或网络执行能力，也不验证用户目标是否真正完成。

## TodoManager

实现位于 `tiny_harness/runtime/todos.py`。

### TodoItem

每项包含：

```text
content
status: pending | in_progress | completed
```

### 更新规则

- 一次更新最多 20 项
- 每项必须是 object
- `content` 必须是字符串，去除首尾空白后不能为空
- 缺少 `status` 时兼容为 `pending`
- 同时最多一个 `in_progress`
- 全部项目验证成功后才替换当前列表
- 非法更新不会破坏旧列表或增加 revision
- 空列表合法，渲染为 `No todos.`

正常工具 schema 要求 `todos` 为 array。为保留 s05 的容错能力，运行时也接受 JSON array string 或 Python list literal string；后者使用 `ast.literal_eval`，不使用 `eval`。

### Run-scoped 状态

每次 `agent_loop()` 创建独立的 TodoManager。状态只存在于本次 Python 运行的内存中，不使用 s05 教学代码中的模块级全局 `TODO`。

因此：

- 两次 Agent run 不共享 Todo
- Todo 不写入 workspace
- Todo 不支持 Session Resume
- 未来每个 Subagent 调用独立 Agent Loop 时会自然获得隔离状态

## todo_write 工具

实现位于 `tiny_harness/tools/todo.py`，schema 注册在统一 Tool Registry 中。

成功更新后，工具同时：

1. 在终端打印 `## Current Tasks`
2. 返回相同的渲染结果作为 ToolResult

渲染格式沿用 s05：

```text
[ ] pending
[>] in progress
[x] completed

(1/3 completed)
```

`todo_write` 的默认权限是 `ALLOW`。它仍经过 Phase 5 的完整 Hook 管线：

- Pre Hook 可以阻止更新
- Permission deny 不更新
- 成功或 handler 错误结果可由 Post Hook 观察
- 一个 Todo 调用失败不影响同一模型响应中的后续 tool calls

## Agent Loop 提醒

Agent Loop 维护 `rounds_since_todo`。这里的一轮是“一次包含至少一个 tool call 的模型响应”，不是单个 tool call。

- 该轮至少有一次成功的 Todo 更新：计数归零
- Todo 被 Hook 阻止、权限拒绝或校验失败：不算成功更新
- 该轮没有成功更新：计数加一
- 连续三轮没有成功更新：注入 Reminder，然后计数归零

Reminder 包含当前 Todo 快照：

```text
<todo-reminder>
Update your todo list.

Current todos:
[>] Implement feature
[ ] Run tests
</todo-reminder>
```

## Chat Completions 协议与 Context Guard

s05 使用 Anthropic content blocks，可以在工具结果旁加入 text block。TinyHarness 使用 Chat Completions 消息，因此不能插入没有 `tool_call_id` 的额外 tool message。

Phase 6 将 Reminder 附加到第三轮最后一个合法 tool result 的 content：

- assistant tool calls 与 tool results 仍严格一一对应
- 同一响应中的多个 tool results 顺序不变
- Reminder 和当前 Todo 快照进入最新完整交互 block
- Phase 4 Context Guard 会把 Reminder 计入字符预算
- 最新完整 block 加 Reminder 后仍超出预算时，在下一次模型 API 调用前抛出 `ContextLimitError`

Post Hook 观察的是原始 handler ToolResult，不观察随后由 Agent Loop 添加的 Harness Reminder。

## Event Log

新增 `todo_reminder`：

```text
turn
rounds_since_todo
todo_count
```

事件不记录 Todo 正文或 Reminder 正文。成功的 `todo_write` 继续使用已有事件：

```text
tool_started
→ tool_finished
```

Reminder 的事件顺序为：

```text
tool_finished
→ todo_reminder
→ model_requested
```

## CLI 提示

系统提示增加：多步骤任务开始前使用 `todo_write` 规划，并在执行过程中更新状态。

这是行为引导，不是强制 Gate。模型仍可能：

- 不创建 Todo
- 创建后不更新
- Todo 未完成时直接给出最终回答

这些约束属于后续 Goal Loop / Verification Gate，不在 Phase 6 实现。

## 自动测试

命令：

```powershell
cd D:\learn-claude-code\tinyharness
python -m unittest discover -s tests -v
```

当前结果：

- 执行 107 项测试
- 106 项通过
- 1 项跳过
- 跳过项仍是 Windows 环境无法创建符号链接

新增覆盖：

- Todo 更新、渲染和清空
- JSON 与安全 Python literal string 兼容
- 非法结构、状态、空内容、超过 20 项
- 同时多个 `in_progress`
- 非法更新原子失败并保留旧状态
- Todo 状态在不同 Agent run 间隔离
- `todo_write` schema、默认权限和 ToolResult call ID
- `todo_write` 与其他工具保持模型原始顺序
- 成功更新重置计数
- 非法或被 Hook 阻止的 Todo 不重置计数
- 第三轮注入 Reminder 并携带当前 Todo 快照
- Reminder 不破坏 Chat Completions 工具消息协议
- Reminder 超出 context 预算时在 Provider 调用前失败
- `todo_reminder` 事件顺序与隐私边界
- Phase 1–5 原有测试继续通过

## 明确不做

- Todo ID、优先级、依赖关系或截止时间
- 增量 patch Todo；每次调用提交完整列表
- Todo 文件持久化
- Session Resume
- 自动任务拆解
- 强制所有任务先规划
- 强制最终回答前完成全部 Todo
- Goal Gate 或 Evidence Verification
- Todo CLI 配置参数
- 多 Agent 共享 Todo
- Planner / Executor 双模型

## 真实 API 验收结果

建议使用独立 smoke workspace，避免测试文件进入项目仓库：

```powershell
New-Item -ItemType Directory -Force D:\tinyharness-smoke\phase6

cd D:\learn-claude-code\tinyharness
python -m tiny_harness `
  "这是一个多步骤任务。必须先调用 todo_write 规划两个步骤：创建 phase6-a.txt 和 phase6-b.txt。执行每一步前后更新 Todo 状态，两个文件内容分别为 A 和 B，完成后报告结果。" `
  --workspace D:\tinyharness-smoke\phase6 `
  --max-turns 12 `
  --max-context-chars 100000 `
  --event-log D:\tinyharness-logs\phase6-smoke.jsonl
```

检查：

```powershell
Get-Content D:\tinyharness-smoke\phase6\phase6-a.txt
Get-Content D:\tinyharness-smoke\phase6\phase6-b.txt
Get-Content D:\tinyharness-logs\phase6-smoke.jsonl
```

2026-08-17 验收结果：

- 终端三次显示 `## Current Tasks`，状态依次为 `0/2`、`1/2`、`2/2 completed`
- `phase6-a.txt` 和 `phase6-b.txt` 分别包含 `A` 和 `B`
- 工具顺序严格为 `todo_write → write_file → todo_write → write_file → todo_write`
- Event Log 共 24 条事件，属于同一个 `run_id`，sequence 从 1 连续递增到 24
- 五次工具调用都具有相邻的 `tool_started` / `tool_finished`，outcome 均为 `returned`
- 六次模型调用中，前五次各返回一个 tool call，最后一次返回最终文本
- Event Log 只包含预期 metadata，不包含任务 prompt、Todo 正文、文件路径、文件内容或最终答案正文
- 运行以 `run_finished` 结束

## 参考来源

Phase 6 实现前查看了：

- `learn-claude-code-main/s05_todo_write/code.py`
- `learn-claude-code-main/s05_todo_write/README.zh.md`

`TodoManager.update/render`、`todo_write` 的终端展示、工具 schema 和三轮 Reminder 控制流实质性参考并改写自 s05。TinyHarness 的 run-scoped 状态、Chat Completions Reminder 表达、Permission / Hooks / Event Log / Context Guard 集成是针对当前项目的适配。

没有查看或移植 Claude Code 产品源码。
