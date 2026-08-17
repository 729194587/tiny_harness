# TinyHarness

TinyHarness 是一个面向 Coding Agent Harness Reliability 研究的轻量 Python 项目。

Phase 1 已冻结为最小可运行 Agent 基线，Phase 2 加入 Permission Gate，Phase 3、4 分别加入 Event Log 和 Context Guard，Phase 5 加入 run-scoped Tool Hooks，Phase 6 加入内存 Todo 规划状态。当前 Phase 7 加入同步 Subagent：

```text
CLI
→ Agent Loop
→ Context Guard
→ Chat Completions Model Provider
→ Tool Calls
→ PreToolUse Hooks
→ Permission Gate
→ Tool Registry / Runtime
  ├→ Base Tools / TodoManager
  └→ task → fresh Child Agent Loop → Child Final Text
→ PostToolUse Hooks
→ Tool Results
→ Todo Reminder（连续三轮未更新时）
→ Model
→ Final Answer
→ JSONL Event Log
```

DeepSeek 是当前默认模型服务，但模型适配器通过 `api_key`、`base_url` 和 `model` 配置，可用于使用相同 Chat Completions 工具调用格式的服务。`reasoning_content` 作为可选兼容字段保留。

## Phase 1 基线功能

- 一个同步的 `ChatCompletionsProvider`
- 一个短小、串行的 Agent Loop
- 五个基础工具：
  - `read_file`
  - `write_file`
  - `edit_file`
  - `list_files`
  - `bash`
- 单一工具注册表及 function-tool schemas
- 工具调用参数解析和统一分发
- 工具异常转换为 `ToolResult`
- 一次模型响应中的多个 tool calls
- 文件工具的 workspace 路径边界
- 模型调用次数限制 `max_turns`
- 一次性任务 CLI

Phase 1 的完整范围、验收记录和已知限制见 [PHASE1_BASELINE.md](PHASE1_BASELINE.md)。

## Phase 2：最小 Permission Gate

Phase 2 默认策略为：

| 工具 | 默认权限 |
|---|---|
| `read_file` | ALLOW |
| `list_files` | ALLOW |
| `write_file` | ALLOW |
| `edit_file` | ALLOW |
| `todo_write`（Phase 6） | ALLOW |
| `task`（Phase 7） | ALLOW |
| `bash` | ASK |

`bash` 执行前，CLI 会显示工具参数并询问：

```text
Allow this tool call? [y/N]:
```

只有 `y` 或 `yes` 会放行。空输入、其他输入、EOF、缺少交互 prompt 或权限组件异常都会默认拒绝。拒绝结果作为关联原 tool call ID 的 `ToolResult` 回填模型，Agent Loop 可以继续运行。

Phase 2 的设计、测试状态和待验收项目见 [PHASE2.md](PHASE2.md)。

## Phase 3：最小执行事件日志

指定 `--event-log` 后，TinyHarness 将控制流元数据追加到 JSON Lines 文件：

```powershell
python -m tiny_harness `
  "列出 workspace 中的文件" `
  --workspace D:\learn-claude-code\tinyharness `
  --event-log D:\tinyharness-logs\events.jsonl
```

事件包括：

- `run_started`
- `model_requested`
- `model_responded`
- `tool_started`
- `tool_denied`
- `tool_finished`
- `run_finished`
- `run_failed`

每条事件包含 run ID、递增 sequence、UTC 时间、事件类型和最小 metadata。日志不保存 prompt、`reasoning_content`、工具参数、文件内容、完整工具输出或最终答案正文。

为了让日志独立于 `write_file` / `edit_file` 的 workspace 文件访问范围，建议把日志路径放在 Agent workspace 外部。Event Log 是 observability trace，不是不可篡改的 audit log；Phase 2 没有 OS 级沙箱，用户批准的 `bash` 命令仍可能访问 workspace 外部路径。

未指定 `--event-log` 时使用 no-op logger，Phase 2 行为不变。指定日志后，写入失败会明确终止执行，不会静默丢失事件。

Phase 3 的设计、测试状态和真实 API 验收结果见 [PHASE3.md](PHASE3.md)。

## Phase 4：Minimal Context Guard

指定 `--max-context-chars` 后，每次模型调用前都会按照紧凑 JSON 的 `{messages, tools}` 计算字符数：

```powershell
python -m tiny_harness `
  "列出 workspace 中的文件" `
  --workspace D:\learn-claude-code\tinyharness `
  --max-context-chars 100000
```

预算内的 context 原样复制给 Provider。超出预算时，Context Guard 从最旧的完整 assistant/tool 交互块开始删除，同时始终保留初始消息和最新一个完整交互块。一次 assistant 返回的多个 tool calls 及其全部 tool results 不会被拆开。

如果必保留内容和 tool schemas 本身已经超过预算，会在调用模型 API 前抛出 `ContextLimitError`。协议中存在孤立、缺失或顺序错误的 tool result 时，会在 API 调用前抛出 `ContextProtocolError`。

字符预算是 TinyHarness 的确定性本地计数，不等于模型 token 数，也不保证与某个厂商的 context window 精确对应。未指定该参数时，Phase 3 行为不变。

发生裁剪时，Event Log 增加不含消息正文的 `context_trimmed` 事件。详细设计与真实 API 验收结果见 [PHASE4.md](PHASE4.md)。

## Phase 5：Minimal Tool Hooks

TinyHarness 的 Python API 可以为一次 Agent Run 注册有序的 PreToolUse 和 PostToolUse Hooks：

```python
from tiny_harness.runtime.hooks import HookBlock, ToolHooks

hooks = ToolHooks()
hooks.register_pre(
    lambda context: (
        HookBlock("write disabled")
        if context.tool_name == "write_file"
        else None
    )
)
hooks.register_post(
    lambda context, result: print(context.tool_name, len(result.content))
)
```

Pre Hook 位于参数解析之后、Permission Gate 之前。返回 `HookBlock` 会阻止本次调用，但仍生成关联原 call ID 的 ToolResult。Post Hook 在 handler 已经产生 ToolResult 后运行，只观察结果，不能修改实际回填内容。

Hook 按注册顺序同步执行。Hook 异常包装成 `HookExecutionError` 并明确终止 run；Pre Hook 异常发生在 handler 前，Post Hook 异常可能发生在工具已经产生副作用之后。

主 CLI 暂不增加 Hook 配置参数。真实 API 演示通过 `python -m examples.hooks_demo` 运行。完整设计和真实 API 验收结果见 [PHASE5.md](PHASE5.md)。

## Phase 6：Minimal Todo / Agent Plan

多步骤任务中，模型可以调用 `todo_write` 创建并更新当前运行的任务列表：

```text
[x] Inspect files
[>] Implement change
[ ] Run tests

(1/3 completed)
```

Todo 状态只存在于单次 `agent_loop()` 的内存中。一次最多 20 项，同时最多一个 `in_progress`；更新先完整校验再原子替换。成功更新会把当前列表打印到终端，同时作为 ToolResult 返回模型。

连续三个包含工具调用的模型轮次没有成功更新 Todo 时，Agent Loop 会向最新 tool result 附加当前 Todo 快照和 Reminder。该表达保持 Chat Completions 工具消息配对，并由 Phase 4 Context Guard 统一计算字符预算。Reminder 事件只记录轮次和条目数量，不记录 Todo 正文。

Todo 是规划提示，不是完成条件。TinyHarness 当前不会阻止模型在 Todo 未完成时返回最终答案。完整设计和真实 API 验收结果见 [PHASE6.md](PHASE6.md)。

## Phase 7：Minimal Subagent

主 Agent 可以调用 `task(prompt)` 同步运行一个 fresh-context 子 Agent。父子使用同一个模型 Provider 和 workspace，但子 Agent 不继承父消息、reasoning 或 Todo 状态。子 Agent 的中间工具历史不会进入父上下文，只有最终文本作为 `task` ToolResult 返回。

子 Agent 可使用六个已有工具，但没有 `task`，因此只允许一层委派。子工具继续经过相同 Permission 和 Hooks；`bash` 仍然逐次 ASK。多个 `task` 调用保持原始顺序串行执行。

每个子 Agent 默认最多调用模型 10 次，可通过 `--subagent-max-turns` 调整。子 Agent 使用独立消息历史执行 Context Guard，但继承相同字符预算值。

子生命周期事件写入同一有序 Event Log，并增加 `agent_scope=subagent` 和父 task call ID。详细边界和真实 API 验收结果见 [PHASE7.md](PHASE7.md)。

## 环境要求

- Python 3.10 或更高版本
- 一个兼容的模型 API Key

安装项目和依赖：

```powershell
cd D:\learn-claude-code\tinyharness
python -m pip install -e .
```

## 配置

API Key 通过环境变量提供，不要写入源码或提交到版本库。

```powershell
$env:TINYHARNESS_API_KEY="你的 API Key"
```

可选配置：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `TINYHARNESS_API_KEY` | 无 | 必需的 API Key |
| `TINYHARNESS_MODEL` | `deepseek-v4-flash` | 模型 ID |
| `TINYHARNESS_BASE_URL` | `https://api.deepseek.com` | Chat Completions API 地址 |

环境变量只需在运行 TinyHarness 的进程环境中存在。项目当前不自动读取 `.env` 文件。

## 运行

必须从项目根目录运行模块：

```powershell
cd D:\learn-claude-code\tinyharness

python -m tiny_harness `
  "列出 workspace 中的文件" `
  --workspace D:\learn-claude-code\tinyharness `
  --max-turns 20
```

执行 `python -m pip install -e .` 后，也可以使用命令入口：

```powershell
tinyharness "列出 workspace 中的文件" `
  --workspace D:\learn-claude-code\tinyharness
```

`--workspace` 默认为当前目录，`--max-turns` 默认为 20。`--max-context-chars` 默认不启用。`--subagent-max-turns` 默认为 10，并分别应用于每个同步子 Agent。

## 工具行为

| 工具 | 当前行为 |
|---|---|
| `read_file` | 读取 workspace 内的 UTF-8 文本文件 |
| `write_file` | 写入 UTF-8 文本，并自动创建 workspace 内的父目录 |
| `edit_file` | 仅当 `old_text` 恰好出现一次时替换 |
| `list_files` | 稳定排序并列出指定目录的直接子项 |
| `bash` | 在 workspace 中以 `cwd` 执行系统 shell，超时 120 秒 |
| `todo_write` | 原子替换当前 run 的内存 Todo，并在终端显示状态 |
| `task` | 同步运行 fresh-context 子 Agent，只返回其最终文本 |

文件工具会拒绝解析后位于 workspace 外的路径，包括 `..` 路径穿越、workspace 外的绝对路径和可解析的符号链接逃逸。

`bash` 不受这个文件路径边界约束。它只有 `cwd` 约束，不是安全沙箱。

## 测试

```powershell
cd D:\learn-claude-code\tinyharness
python -m unittest discover -s tests -v
```

Phase 1 冻结时的基线：

- 35 项测试被执行
- 34 项通过
- 1 项跳过：当前 Windows 用户没有创建符号链接的权限

当前 Phase 2 离线测试：

- 50 项测试被执行
- 49 项通过
- 1 项跳过：同一个 Windows 符号链接权限限制

当前 Phase 3 离线测试：

- 65 项测试被执行
- 64 项通过
- 1 项跳过：同一个 Windows 符号链接权限限制

当前 Phase 4 离线测试：

- 80 项测试被执行
- 79 项通过
- 1 项跳过：同一个 Windows 符号链接权限限制

当前 Phase 5 离线测试：

- 91 项测试被执行
- 90 项通过
- 1 项跳过：同一个 Windows 符号链接权限限制

当前 Phase 6 离线测试：

- 107 项测试被执行
- 106 项通过
- 1 项跳过：同一个 Windows 符号链接权限限制

当前 Phase 7 离线测试：

- 121 项测试被执行
- 120 项通过
- 1 项跳过：同一个 Windows 符号链接权限限制

## 项目结构

```text
tinyharness/
├─ tiny_harness/
│  ├─ __main__.py
│  ├─ agent/
│  │  ├─ loop.py
│  │  └─ messages.py
│  ├─ models/
│  │  ├─ base.py
│  │  └─ chat_completions.py
│  ├─ tools/
│  │  ├─ filesystem.py
│  │  ├─ registry.py
│  │  ├─ shell.py
│  │  ├─ task.py
│  │  └─ todo.py
│  └─ runtime/
│     ├─ context.py
│     ├─ events.py
│     ├─ hooks.py
│     ├─ permissions.py
│     └─ todos.py
├─ examples/
│  └─ hooks_demo.py
├─ tests/
├─ PHASE1_BASELINE.md
├─ PHASE2.md
├─ PHASE3.md
├─ PHASE4.md
├─ PHASE5.md
├─ PHASE6.md
├─ PHASE7.md
└─ pyproject.toml
```

`runtime/permissions.py`、`runtime/events.py`、`runtime/context.py`、`runtime/hooks.py` 和 `runtime/todos.py` 已分别在 Phase 2–7 接入。`runtime/goal.py` 和 `agent/session.py` 仍是空占位文件。

## 当前边界

当前实现了非持久化的 ALLOW / DENY / ASK、显式启用的控制流 metadata JSONL 日志、可选的确定性 context 字符预算、通过 Python API 注入的同步 Pre/Post Tool Hooks、run-scoped Todo 规划状态，以及单层同步 Subagent。仍没有 Hook 配置文件、Prompt/Stop Hooks、精确 token 预算、摘要压缩、Session Resume、Goal Gate、Artifact Store、Memory、MCP、并行 Subagent、Agent Teams 或 Workflow。工具和 Subagent 顺序执行；主 CLI 会显示 ASK 交互、Todo 更新、Subagent 状态和最终答案。

这些限制是后续可靠性研究的基线，不应被误认为已经实现但未启用的功能。

## 参考来源

Phase 1 选择性参考了：

- `learn-claude-code-main/s01_agent_loop`
- `learn-claude-code-main/s02_tool_use`

参考内容仅限核心控制流、工具 schema、分发和 workspace 路径边界。TinyHarness 根据自身 Phase 1 目标重新实现，没有直接移植 integrated harness 或后续阶段机制。

Phase 2 选择性参考了 `s03_permission` 的执行前权限控制流。Phase 5 选择性参考了 `s04_hooks` 的有序注册、PreToolUse 阻止和 PostToolUse 观察概念，但保留了 TinyHarness 独立的 Permission Gate，只实现 Tool Hooks。Phase 6 实质性参考并改写了 `s05_todo_write` 的 TodoManager、工具 schema、终端渲染和三轮 Reminder 控制流。Phase 7 实质性参考并改写了 `s06_subagent` 的 task schema、fresh child context、同步嵌套 Loop、共享 workspace 和单层委派控制流。Phase 3、4 是 TinyHarness 的可靠性扩展。没有查看 Claude Code 产品源码。
