# TinyHarness

TinyHarness 是一个轻量、可测试的 Coding Agent Harness Runtime。它把模型调用、工具执行、权限、恢复、上下文与验证能力组合成一个同步 Agent Loop，同时保持控制流显式且易于做 deterministic regression。

## Agent Loop

```text
Model Response
    ├─ 有 tool calls → 校验 → 权限与 hooks → 执行 → 下一轮
    └─ 无 tool calls → Final Answer → return
```

Final Answer 是普通 Agent Loop 的自然退出条件。Runtime 不自动生成答案，也不会在工具或测试成功后强制停止。

## 核心能力

### 执行控制

- 严格校验 model response、tool calls 与 finish reason，非法响应不会触发副作用。
- `ALLOW` / `DENY` / `ASK` 权限策略，以及可执行的 DENY feedback。
- 有序、run-scoped 的 Pre/Post Tool Hooks。
- 文件工具与 shell 的 workspace boundary。
- 同一模型响应中的 tool batch 按顺序完整执行后再进入下一轮。

### 恢复与验证

- 区分 Logical Turn 与 Physical Model Attempt。
- 对暂时性 Provider 错误做 bounded retry；致命错误和重试耗尽显式失败。
- 重复权限拒绝会提供重新规划提示，但不会伪造完成状态。
- 可选、无参数的 `run_tests()` capability 使用固定 argv、workspace cwd、`shell=False`、timeout 和 `PYTHONDONTWRITEBYTECODE=1`。
- `run_tests` 的 PASS/FAIL 都只是普通 Tool Result；模型必须在后续调用中自行返回 Final Answer。
- 如果模型持续返回 tool calls，耗尽 `max_turns` 后抛出 `MaxTurnsExceededError`。

### Context 控制

- 可配置 context character budget。
- 大 Tool Result 落盘并在消息中保留受限 preview。
- 保持 tool batch 完整性的 history trimming 与 compaction。
- 可选 LLM summary、手动 `compact` 与 context-length 错误后的 reactive compaction。
- 压缩失败不会提交半完成的 canonical history。

### 扩展能力

- 同步 Subagent：隔离消息、共享 workspace，并继承 Provider、Permission、Hooks、Event Logger、Context Policy、Recovery Policy 与 `run_tests` capability；child 不能再创建嵌套 Subagent。
- Skills：先注入有界 metadata catalog，需要时再通过工具加载正文。
- Memory：可选的 workspace 持久记忆、选择、提取与 consolidation；失败默认 fail-open，Event Log 故障除外。
- Todo：run-scoped 状态与周期性 reminder。

## 安装与运行

要求 Python 3.10+，并安装项目依赖：

```powershell
python -m pip install -e .
```

设置兼容 Chat Completions 的 API 配置后运行单次任务：

```powershell
$env:TINYHARNESS_API_KEY = "..."
python -m tiny_harness "修复测试并说明结果" --workspace .
```

省略 task 会进入复用同一个 `AgentSession` 的交互模式：

```powershell
python -m tiny_harness --workspace .
```

常用选项：

```text
--max-turns N
--max-model-retries N
--max-context-chars N
--no-context-compaction
--subagent-max-turns N
--event-log PATH
--memory
```

## Reliability Eval

Eval 保留外部 hidden grading、post-run grading、Permission diagnostics 与 `run_tests` diagnostics。当前 Real Coding/Pilot 只使用普通 `harness` profile。

离线 deterministic eval 不调用真实 API：

```powershell
python -m evals.run --suite offline --results-dir .\eval-results
```

Pilot 会调用配置的真实 Provider：

```powershell
python -m evals.pilot run `
  --repetitions 2 `
  --results-dir .\pilot-results
```

结果包含 Verified、Premature terminal completion、Explicit failure、MaxTurnsExceeded、Tool denied、`run_tests` 调用数、turn/tool call 数以及 final workspace grade。完整协议见 [evals/README.md](evals/README.md)。

## 测试

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

测试全部使用 scripted provider 或本地 fixture，不需要真实 API。

Historical Goal Gate experiments are preserved in the `reliability-eval-v3-goal-gate` tag.
