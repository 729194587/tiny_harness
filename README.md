# TinyHarness

TinyHarness 是一个轻量、可读的 Coding Agent Harness Runtime。它围绕一个很小的 Agent Loop 组织模型调用、工具执行和最终回答，同时为权限、Hooks、事件、恢复、上下文、Skills、Memory、Subagent 与 Todo 提供明确边界。

```text
LLM response
  ├─ tool calls → validate batch → execute tools → append results → next turn
  └─ final text → final-answer hooks → return
```

模型响应在写入 canonical history 前完成协议校验。同一响应中的 Tool Calls 作为完整 batch 顺序执行；暂时性 Provider 错误可在同一 logical turn 内重试；达到 turn 上限但没有自然结束时会明确失败。

### Tool Batch 生命周期

整个 Tool Call batch 在提交 assistant message 和执行任何工具前验证：调用 ID 必须是非空字符串且在 batch 内唯一，调用名称与 arguments 的传输类型必须合法，`finish_reason` 必须与调用列表一致。非法 batch 抛出 `ModelProtocolError`，不会执行其中任何 handler。arguments 字符串的 JSON 解码或工具参数错误仍作为普通 Tool Result 返回。

`AgentSession` 采用保守的失败契约：一次非空提交开始后，只有正常返回才保持可用；任何异常或中断均原样传播，并使 `session.failed` 为 `True`。已有 assistant/tool 历史保留，但 fatal Hook 或事件输出异常可能使 batch 未闭合；缺失结果不表示工具未执行。Runtime 不补造结果、不重放调用，也不回滚文件或进程副作用。后续 `submit()` 在改动历史或调用模型之前抛出 `SessionFailedError`，要求检查 workspace 后调用 `clear()`。

`clear()` 显式丢弃会话历史并解除 failed 状态，不撤销工具副作用；空任务的输入校验失败不会使 Session 进入 failed 状态。直接使用 `run_agent()` / `agent_loop()` 的调用方应在异常后自行丢弃运行或重建会话，不能把可能未闭合的 messages 直接用于下一次运行。

## Tools

每个 Tool module 通过 `build_tools(context)` 返回零个或多个 `ToolDefinition`。一个 definition 同时拥有 model-facing 元数据与可执行 handler：

```text
name + description + parameters + execute
```

Run 初始化时，`tiny_harness.tools.discovery` 按模块名扫描 `tiny_harness/tools/`，调用存在的 `build_tools(context)`，并把实际可用的 definitions 注册到 `ToolRegistry`。Factory 可依据当前 runtime capability 返回空集合，例如没有 Subagent runner 时不注册 `task`，没有 Skill catalog 内容时不注册 `load_skill`。

```text
Tool module
  → build_tools(run context)
  → discovery
  → ToolRegistry.register
  → ToolDefinition.model_schema()
  → model API tools=[...]
```

`ToolRegistry` 负责注册、重复名称校验、查找、列举和 schema 投影，不保存 built-in Tool 的第二份描述或 schema。执行仍经过统一管线：

```text
tool call
  → registry lookup / argument decoding
  → pre hooks
  → permission
  → started event
  → ToolDefinition.execute
  → finished event
  → post hooks
  → tool result
```

这只是 TinyHarness 内置 Tool 的目录式扩展 seam，不是通用 Plugin Framework，也不支持 entry points 或 hot reload。新增普通 Tool 通常只需在 `tiny_harness/tools/` 增加一个包含 `build_tools(context)` 的模块；如果默认权限属于新的安全策略，还应显式更新 permission policy。

## Skills

TinyHarness 从三个固定 root 发现 Skills：

```text
tiny_harness/skills/                  # bundled
~/.tinyharness/skills/                # user
<workspace>/.tinyharness/skills/      # workspace
```

每个 Skill 使用 `<name>/SKILL.md`，文件以 YAML frontmatter 提供 `name` 和 `description`。`AgentSession` 创建时生成不可变 catalog snapshot；模型起初只收到有界 metadata，需要时通过 `load_skill` 加载完整正文。正文始终作为不可信指导数据，不能覆盖 system/user 指令、Permission、Hooks 或 workspace boundary。Session 期间文件发生变化时，需要新建 Session 才会重新发现。

## Memory

Memory 是 opt-in 子系统，源码位于 `tiny_harness/memory/`，运行数据位于：

```text
<workspace>/.tinyharness/memory/
```

CLI 使用 `--memory` 启用。Run 开始时，Memory 会发现 catalog、为当前请求选择相关条目、按需加载并以 run-scoped markers 注入上下文。Root agent 产生 final answer 时，final-answer hook 会提取新的 durable memory、写入存储，并在达到阈值时可选地执行 consolidation。提取与 consolidation 失败不会阻止返回已经生成的最终回答。

Memory 用于跨 Session 的稳定偏好、反馈、项目事实和参考信息；它不替代当前计划、Todo、执行状态或会话历史。

## 运行时边界

- **Context**：按完整消息块裁剪或压缩，不拆散 Tool Call/Result batch；大型 Tool Result 可落盘并保留有界 preview。
- **Recovery**：暂时性模型错误按策略重试，fatal error 与 max-turn exhaustion 保持明确语义。
- **Permission**：每次 Tool Call 得到 `ALLOW`、`DENY` 或 `ASK`；重复拒绝会向模型返回可执行的恢复提示。
- **Hooks 与 Events**：Pre/Post Tool Hooks 和生命周期事件位于统一执行边界，Tool handler 不重复实现横切逻辑。
- **Subagent**：使用独立 messages、共享 workspace，并继承 Permission、Hooks、Recovery、Context、Skills、Memory 与测试 capability；不允许嵌套 Subagent。
- **Todo**：维护 run-scoped 任务状态，通过 `todo_write` 显式更新。

## Quick Start

TinyHarness 要求 Python 3.10+：

```powershell
python -m pip install -e .
```

CLI 使用兼容 Chat Completions 的 Provider。API key 必填，model 和 base URL 可选：

```powershell
$env:TINYHARNESS_API_KEY = "..."
$env:TINYHARNESS_MODEL = "deepseek-v4-flash"
$env:TINYHARNESS_BASE_URL = "https://api.deepseek.com"
```

运行单次任务：

```powershell
python -m tiny_harness "修复测试并说明结果" --workspace .
```

省略 task 会进入复用同一个 `AgentSession` 的交互模式：

```powershell
python -m tiny_harness --workspace .
```

当前 CLI 控制项包括：

- `--workspace`
- `--max-turns`
- `--max-model-retries`
- `--max-context-tokens`
- `--no-context-compaction`
- `--subagent-max-turns`
- `--event-log`
- `--memory`

TinyHarness 使用 token-aware context budgeting：上下文预算覆盖 system
prompt、messages 和 tool schemas。默认的 `HeuristicTokenMeter` 按紧凑 JSON
序列化后约每 4 个字符估算 1 token，因此无需绑定具体模型 tokenizer；未来可注入
provider-specific tokenizer 或基于 API usage 校准的实现。超过 token 预算的软阈值时，
仍沿用既有的工具结果裁剪、历史截断和摘要回退策略。
CLI 默认 `--max-context-tokens 125000`；Python 入口使用
`max_context_tokens`，并可通过 `token_meter` 注入替代实现。

## 项目结构

```text
tiny_harness/
  agent/       # Agent Loop、Session、run composition、tool batch、Subagent
  memory/      # Memory discovery/store、lifecycle、consolidation
  models/      # Provider contract 与 Chat Completions adapter
  runtime/     # Context、events、Hooks、Permission、Recovery、Skills、Todo、tests
  skills/      # bundled Skills
  tools/       # ToolDefinition、discovery、registry 与 built-in Tool modules
examples/
  hooks_demo.py
tests/         # scripted providers 与 deterministic regression tests
```

## Hooks 示例

`examples/hooks_demo.py` 展示一个阻止 `write_file` 的 Pre hook 和一个观察结果的 Post hook：

```powershell
python examples/hooks_demo.py "检查项目并给出建议" --workspace .
```

## 测试

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

测试使用 scripted provider 或本地 fixture，不需要真实 API。
