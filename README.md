# TinyHarness

TinyHarness 是一个面向 Coding Agent Harness Reliability 研究的轻量 Python 项目。

当前完成的是 Phase 1：以尽可能少的机制打通一个可运行的 Agent 链路：

```text
CLI
→ Chat Completions Model Provider
→ Agent Loop
→ Tool Calls
→ Tool Registry / Runtime
→ Tool Results
→ Model
→ Final Answer
```

DeepSeek 是当前默认模型服务，但模型适配器通过 `api_key`、`base_url` 和 `model` 配置，可用于使用相同 Chat Completions 工具调用格式的服务。`reasoning_content` 作为可选兼容字段保留。

## Phase 1 功能

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

`--workspace` 默认为当前目录，`--max-turns` 默认为 20。

## 工具行为

| 工具 | Phase 1 行为 |
|---|---|
| `read_file` | 读取 workspace 内的 UTF-8 文本文件 |
| `write_file` | 写入 UTF-8 文本，并自动创建 workspace 内的父目录 |
| `edit_file` | 仅当 `old_text` 恰好出现一次时替换 |
| `list_files` | 稳定排序并列出指定目录的直接子项 |
| `bash` | 在 workspace 中以 `cwd` 执行系统 shell，超时 120 秒 |

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
│  │  └─ shell.py
│  └─ runtime/
├─ tests/
├─ PHASE1_BASELINE.md
└─ pyproject.toml
```

`runtime/` 和 `agent/session.py` 目前仍是空占位文件，没有接入 Phase 1。

## 当前边界

Phase 1 没有 Permission、Context Management、Event Log、Session Resume、Artifact Store、Memory、MCP、Subagent、Agent Teams 或 Workflow。工具顺序执行，CLI 只打印最终答案。

这些限制是后续可靠性研究的基线，不应被误认为已经实现但未启用的功能。

## 参考来源

Phase 1 选择性参考了：

- `learn-claude-code-main/s01_agent_loop`
- `learn-claude-code-main/s02_tool_use`

参考内容仅限核心控制流、工具 schema、分发和 workspace 路径边界。TinyHarness 根据自身 Phase 1 目标重新实现，没有直接移植 integrated harness 或后续阶段机制。
