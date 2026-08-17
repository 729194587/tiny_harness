# Phase 1 Baseline

## 状态

- 状态：Accepted
- 验收日期：2026-08-17
- 阶段目标：最小可运行 Coding Agent
- 当前默认服务：DeepSeek
- 模型接口：Chat Completions-compatible API
- 默认模型：`deepseek-v4-flash`
- 默认 Base URL：`https://api.deepseek.com`

## 目标链路

Phase 1 验收的最小链路为：

```text
CLI
→ ChatCompletionsProvider
→ Agent Loop
→ Tool Calls
→ Tool Registry
→ Tool Runtime
→ Tool Results
→ Model
→ Final Answer
```

## 已实现范围

1. 可配置的 `ChatCompletionsProvider`
2. 一个尽可能短小的同步 Agent Loop
3. 五个基础工具：
   - `read_file`
   - `write_file`
   - `edit_file`
   - `list_files`
   - `bash`
4. 单一工具注册、function-tool schemas 和统一分发
5. 一次性任务 CLI
6. 简单的 `max_turns`
7. 文件工具的 workspace 路径边界
8. 工具异常转换成 `ToolResult`
9. 支持模型一次返回多个 tool calls
10. 可选保留和回填 `reasoning_content`

DeepSeek 是当前 CLI 默认配置，不是 Provider 的硬依赖。Provider 的 `api_key`、`base_url` 和 `model` 均由外部注入。

## 最小数据契约

定义位于 `tiny_harness/agent/messages.py` 和 `tiny_harness/models/base.py`。

### `ToolCall`

- `id`：模型生成的工具调用 ID
- `name`：工具名称
- `arguments_json`：模型返回的原始 JSON 参数字符串

参数保留为原始 JSON，是为了让工具运行时能够把非法 JSON 转换为与原调用 ID 对应的 `ToolResult`。

### `ToolResult`

- `tool_call_id`：原始工具调用 ID
- `content`：返回给模型的文本结果

Phase 1 不定义结构化错误字段。普通工具错误统一表示为以 `Error:` 开头的文本。

### `ModelResponse`

- `content`
- `reasoning_content`
- `tool_calls`
- `finish_reason`

### `ModelProvider`

Provider 实现同步的：

```python
complete(messages, tools) -> ModelResponse
```

Phase 1 不定义完整的 provider-neutral message hierarchy。Agent Loop 使用 Chat Completions 消息字典。

## Agent Loop 不变量

- `max_turns` 统计模型调用次数，不统计单个工具调用数量。
- 同一次模型响应中的多个工具调用按原始顺序串行执行。
- 每个 `ToolCall` 恰好生成一个具有相同 ID 的 `ToolResult`。
- 一个 assistant 消息中的多个 tool calls 后面跟随各自独立的 `role="tool"` 消息。
- 工具错误作为 tool result 回填模型，不直接终止 Agent Loop。
- `reasoning_content` 存在时随 assistant 工具调用消息回填。
- 只有无工具调用且 `finish_reason == "stop"` 时才作为正常最终回答返回。
- `length` 等其他停止原因不会被误认为成功的最终回答。
- 如果最后一个允许的 model turn 仍返回工具调用，工具会执行并回填到内存消息，随后循环以 `Maximum model turns reached` 终止，不会再调用模型。

## 工具语义

### 文件工具

- 所有路径都相对于显式传入的 workspace 解析。
- 拒绝解析后位于 workspace 外的路径。
- 覆盖 `..` 穿越、workspace 外绝对路径和可解析的符号链接逃逸。
- 文件以 UTF-8 文本处理。

`read_file`：读取完整文件内容，不截断。

`write_file`：写入完整内容，并自动创建 workspace 内的父目录；成功消息报告 UTF-8 字节数。

`edit_file`：

- `old_text` 为空：失败
- 0 次匹配：`Text not found`
- 1 次匹配：执行替换
- 多次匹配：`Text is not unique`
- 失败时不写入文件

`list_files`：只列指定目录的直接子项；结果稳定排序；目录名以 `/` 结尾；空目录返回 `(no files)`。

### `bash`

- 使用系统 shell。
- `cwd` 设置为 workspace。
- 超时为 120 秒。
- stdout 和 stderr 合并为文本结果。
- 成功且无输出时返回 `(no output)`。
- 非零退出时返回退出码和输出。
- 不实现命令权限控制或进程沙箱。

### 工具运行时

统一 dispatcher 处理：

- 未知工具
- 非法 `arguments_json`
- 参数顶层不是 JSON object
- 缺少或多余参数
- handler 抛出的普通异常
- workspace 路径越界

这些失败都会成为关联原始 tool call ID 的 `ToolResult`。

JSON Schema 当前用于向模型描述工具参数，但没有独立的运行时 JSON Schema validator。类型和参数错误最终由 JSON 解析、Python 调用或 handler 转换为 tool result。

## CLI 与配置

入口位于 `tiny_harness/__main__.py`。

CLI 接收：

- 一个位置参数 `task`
- `--workspace`，默认为当前目录
- `--max-turns`，默认为 20

环境变量：

| 名称 | 必需 | 默认值 |
|---|---|---|
| `TINYHARNESS_API_KEY` | 是 | 无 |
| `TINYHARNESS_MODEL` | 否 | `deepseek-v4-flash` |
| `TINYHARNESS_BASE_URL` | 否 | `https://api.deepseek.com` |

标准运行命令：

```powershell
cd D:\learn-claude-code\tinyharness

$env:TINYHARNESS_API_KEY="<set locally>"

python -m tiny_harness `
  "列出 workspace 中的文件" `
  --workspace D:\learn-claude-code\tinyharness `
  --max-turns 20
```

API Key 不写入项目文件。

## 自动测试基线

命令：

```powershell
cd D:\learn-claude-code\tinyharness
python -m unittest discover -s tests -v
```

2026-08-17 验证结果：

- 执行 35 项测试
- 34 项通过
- 1 项跳过
- 跳过项：当前 Windows 用户缺少创建符号链接所需的权限

其余路径穿越、绝对路径越界、工具分发、异常转换、多工具调用、Agent Loop、Provider 和 CLI 测试均通过。

## 真实 API Smoke Test

- 日期：2026-08-17
- 模型服务：DeepSeek
- Prompt：`列出 workspace 中的文件`
- Workspace：`D:\learn-claude-code\tinyharness`

观察到的最终结果准确包含：

- `evals/`
- `examples/`
- `pyproject.toml`
- `tests/`
- `tiny_harness/`

CLI 成功收到最终自然语言回答，返回内容与实际 workspace 相符。

当前 CLI 不打印工具调用或事件日志，因此 smoke test 没有独立保存模型调用 `list_files` 的运行轨迹。Phase 1 的功能验收由实际输出和已覆盖消息回填的自动测试共同构成。

结论：PASS。

## 明确不做

Phase 1 没有实现：

- Permission ALLOW / DENY / ASK
- token budget
- Context Management
- Event Log
- Session Resume
- Goal Gate
- Evidence Collector
- Eval framework
- Artifact store
- Background Task
- MCP
- Memory
- Subagent
- Agent Teams
- Workflow
- Worktree

`tiny_harness/runtime/` 和 `tiny_harness/agent/session.py` 是空占位文件，没有连接到当前执行链路。

## 已知限制

- `bash` 不是沙箱，可以访问或修改 workspace 外部资源。
- 文件路径边界只保护四个文件工具，不保护 shell 命令。
- 没有权限确认、命令 allowlist 或危险命令过滤。
- 没有工具事件日志；CLI 只打印最终模型回答。
- 工具调用串行执行，没有并发或流式工具执行。
- 工具输出没有大小限制或持久化机制。
- 巨大的 `read_file` 或 `bash` 输出可能造成较高内存和上下文消耗。
- 没有 token budget、上下文裁剪或消息压缩。
- 没有 Harness 自己控制的模型重试、退避或错误恢复策略。
- `list_files` 不递归，也不支持 glob pattern。
- 文件工具只处理 UTF-8 文本。
- CLI 每个进程只处理一个任务，不保留会话。
- Provider 只读取 API 响应的第一个 choice。
- Provider 面向 Chat Completions 消息和工具调用格式，不支持任意模型协议。
- `reasoning_content` 仅作为可选兼容扩展；其他厂商私有推理字段未处理。
- 工具错误是普通文本，没有结构化错误类型。
- 符号链接逃逸的实现存在测试，但当前 Windows 环境因权限限制跳过了实际创建符号链接的测试。

## 参考与代码来源

Phase 1 选择性查看了：

- `learn-claude-code-main/s01_agent_loop/code.py`
- `learn-claude-code-main/s02_tool_use/code.py`

借鉴内容是核心循环形状、工具 schema、工具分发和 workspace 路径检查思路。TinyHarness 按当前需求重新实现；没有直接复制 integrated harness，也没有读取或移植 s08、s15、s16、s17 等后续机制。

## 冻结规则

该文档描述 Phase 1 验收时的基线。进入后续阶段后：

- 不覆盖这里记录的历史验收结果。
- 行为发生变化时，在新阶段文档中记录差异。
- 自动测试应继续保护 Phase 1 的关键不变量。
- 如果使用 Git，建议在首次提交后创建 `phase-1-baseline` tag。
