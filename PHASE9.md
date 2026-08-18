# Phase 9：Model Failure Recovery

## 状态

- 状态：Accepted（2026-08-18）
- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API smoke test：通过

## 唯一目标

为物理模型 API 调用增加小而有界的失败恢复层：暂时性失败按统一预算重试，API 拒绝 context 时最多执行一次 reactive compaction；不能安全恢复的错误立即失败。

```text
logical model request
    → physical provider attempt
        ├→ success → continue Agent Loop
        ├→ transient → bounded backoff → retry
        ├→ context_length → reactive compact once → retry
        └→ fatal / exhausted → fail run
```

Phase 9 不重试工具调用，也不改变 `max_turns` 的含义。一个 Agent Loop turn 可以包含多次物理 Provider attempt；重试和摘要调用不额外消耗 `max_turns`。

## Provider 错误契约

`ModelProviderError` 将 SDK 或服务错误归一化为五类：

| kind | 默认处理 |
|---|---|
| `rate_limit` | 有界重试 |
| `server_unavailable` | 有界重试 |
| `connection` | 有界重试 |
| `context_length` | Agent Loop 最多编排一次 reactive compaction |
| `fatal` | 立即失败 |

Chat Completions Adapter 当前采用以下保守规则：

- HTTP 429 → `rate_limit`；
- HTTP 500、502、503、504、529 → `server_unavailable`；
- SDK 连接或超时异常 → `connection`；
- 仅在无状态码或 400、413、422 响应中识别少量 context 关键词；
- 其余错误 → `fatal`。

DeepSeek 没有公开保证一个稳定、专用的 context overflow error code，因此关键词识别只是兼容 heuristic，不是协议保证。认证、余额、参数校验等错误不会因为文本偶然包含 context 关键词而自动重试。

Chat Completions SDK 自身的自动重试被设置为 `max_retries=0`，避免 SDK retry 与 Harness retry 叠加，保证 TinyHarness 的次数和事件可观察。

DeepSeek 返回 `finish_reason=insufficient_system_resource` 时，Adapter 在解析 `content`、`reasoning_content` 或 tool calls 之前把它转换成 `server_unavailable`。该 choice 中的任何内容和工具调用全部丢弃，不能进入 Agent Loop 或触发工具。

`finish_reason=length` 和 `content_filter` 在 Adapter 读取 choice payload 前转换为 `fatal`，因此不会读取或执行响应中可能存在的不完整 tool calls，也不会重试或被当作最终答案。

Agent Loop 对任意第三方 `ModelProvider` 再执行一次防御性验证，并且验证发生在追加 assistant message 和 dispatch 工具之前：

- `stop` 必须没有 tool calls；
- `tool_calls` 必须至少包含一个调用；
- 其他 finish reason 一律不可执行。

因此即使自定义 Provider 错误地返回 `length + write_file(...)` 或 `stop + bash(...)`，响应也不会进入 canonical history，工具不会产生副作用。

## Bounded Retry

`RecoveryExecutor` 只负责一次 Provider 调用的 attempt、暂时性错误重试、backoff 和事件，不依赖 `ContextCompactor`。

默认配置：

- 每个逻辑模型请求最多重试 2 次，即暂时性失败路径最多 3 次物理 attempt；
- 指数 backoff 从 1 秒开始；
- delay 上限 8 秒；
- 默认增加最多 25% 的正向 jitter；
- 服务返回数字形式 `Retry-After` 时优先采用，但仍受 8 秒上限约束。

CLI 可以调整重试次数：

```text
--max-model-retries <non-negative integer>
```

传入 `0` 会关闭暂时性重试。Phase 9 暂不开放 delay 和 jitter 的 CLI 参数；Python API 测试可以通过 `RecoveryPolicy` 注入确定性配置。

同一个主模型逻辑请求只创建一个 `RecoveryState`。如果执行顺序为：

```text
transient failure
→ context_length
→ reactive compact
→ transient failure
```

第二个 transient failure 继续使用原计数，不会因为 compaction 重置 retry budget。摘要模型调用是独立的逻辑请求，有自己有界的 `RecoveryState`。

## Reactive Compaction

Reactive recovery 只在同时满足以下条件时启用：

- Provider 返回归一化的 `context_length`；
- CLI/Python API 已配置 `max_context_chars`，因此存在 `ContextCompactor`；
- 当前逻辑请求尚未使用 reactive compaction；
- 历史中存在可压缩的旧状态。

Agent Loop 负责捕获 `context_length`、调用 compactor、提交压缩后的 canonical history，再使用原 `RecoveryState` 重试。`RecoveryExecutor` 不导入或调用 `ContextCompactor`。

Phase 9 使用明确的 shrink margin：

```text
reactive target = min(max_context_chars, floor(failed_request_chars × 0.75))
```

摘要请求本身和压缩后的主请求都必须不超过这个 target。必保留任务、Todo、工具 schemas、summary marker 和最新完整工具证据无法满足 target 时，在再次调用 API 前抛出 `ContextLimitError`。

Reactive compaction：

- 保存完整 transcript；
- 只总结较旧历史；
- 保留初始任务、当前 Todo 和最新完整 assistant/tool 协议块；
- 不向摘要调用提供工具；
- 最多执行一次；
- 第二次 `context_length` 立即失败；
- 摘要调用自身发生 `context_length` 时不会递归 compaction。

因为字符数不是 token 数，即使满足 25% shrink margin，服务仍可能拒绝重试；此时 Harness 明确失败，不进入无界循环。

## Event Log

Phase 9 新增：

- `model_request_failed`
- `model_retry_scheduled`
- `model_retry_exhausted`

每次物理 Provider attempt 都单独产生 `model_requested`，并带有：

- `purpose`：`main` 或 `summary`；
- Agent Loop `turn`；
- 当前逻辑请求内从 1 开始的 `attempt`。

成功的 attempt 产生带相同 metadata 的 `model_responded`。失败事件只记录规范化 error kind、HTTP status、是否计划恢复等控制流信息，不记录异常正文、响应 body、prompt、摘要或工具 payload。

父子 Agent 各自创建 `RecoveryExecutor` 和 retry state。子 Agent 的恢复事件继续通过 `ScopedEventLogger` 带有 `agent_scope=subagent` 与 `parent_tool_call_id`。

## 自动测试

命令：

```powershell
python -m unittest discover -s tests -v
```

当前结果：

- 执行 148 项测试；
- 145 项通过；
- 3 项跳过；
- 跳过项均为当前 Windows 用户无法创建 filesystem、artifact 目录和最终 artifact 文件的符号链接测试。

Phase 9 新增测试覆盖：

- Adapter 对 429、5xx、连接、context 和 fatal 错误的归一化；
- SDK 内建 retry 已关闭；
- `insufficient_system_resource` 不读取或保留 choice payload；
- `length/content_filter` 不读取 choice payload，也不执行其中的工具；
- 第三方 Provider 的 finish reason/tool calls 矛盾组合在 commit 前失败；
- 指数 backoff、jitter-free 确定性序列和 `Retry-After` 上限；
- fatal 立即失败和 transient retry exhaustion；
- 每个物理 attempt 的独立事件 metadata；
- transient counter 跨 reactive compact 不重置；
- reactive 输出和摘要请求都满足 75% shrink target；
- 无旧历史时在摘要 API 前失败；
- 第二次 context rejection 不再次 compact；
- 主模型与摘要模型的 attempt 状态独立；
- 子 Agent retry 独立且事件保持父 task correlation；
- Phase 1–8 全量回归。

## 真实 API smoke test

真实服务很难稳定制造 429、连接中断或 context overflow，因此故障分支以确定性离线测试为主要验收。真实 API smoke test 只确认正常 DeepSeek 请求、工具副作用、物理 attempt 事件和 Subagent lineage 没有回归。

每次使用时间戳创建新的 workspace 和日志，避免 JSONL append 或旧文件影响计数。

### Smoke A：主 Agent 正常工具路径

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$smokeWorkspace = "D:\tinyharness-smoke\phase9-main-$stamp"
$smokeLog = "D:\tinyharness-logs\phase9-main-$stamp.jsonl"

New-Item -ItemType Directory -Force $smokeWorkspace

cd D:\learn-claude-code\tinyharness

python -m tiny_harness `
  "PHASE9_MAIN_PROMPT_PRIVATE。必须只调用一次 write_file 创建 phase9.txt，内容严格为 PHASE9_TOOL_OUTPUT_PRIVATE；然后只调用一次 read_file 验证，最后报告结果。" `
  --workspace $smokeWorkspace `
  --max-turns 6 `
  --max-model-retries 2 `
  --max-context-chars 100000 `
  --event-log $smokeLog
```

运行后执行确定性检查：

```powershell
$events = @(Get-Content -LiteralPath $smokeLog | ForEach-Object {
  $_ | ConvertFrom-Json
})
$modelEvents = @($events | Where-Object {
  $_.event_type -in @("model_requested", "model_responded")
})
$retryEvents = @($events | Where-Object {
  $_.event_type -in @(
    "model_request_failed",
    "model_retry_scheduled",
    "model_retry_exhausted"
  )
})
$writeEvents = @($events | Where-Object {
  $_.event_type -eq "tool_finished" -and
  $_.data.tool_name -eq "write_file" -and
  $_.data.outcome -eq "returned"
})
$readEvents = @($events | Where-Object {
  $_.event_type -eq "tool_finished" -and
  $_.data.tool_name -eq "read_file" -and
  $_.data.outcome -eq "returned"
})
$rawLog = Get-Content -LiteralPath $smokeLog -Raw

if ($modelEvents.Count -eq 0) { throw "No model lifecycle events" }
if (@($modelEvents | Where-Object {
  $_.data.purpose -ne "main" -or $_.data.attempt -ne 1
}).Count -ne 0) { throw "Unexpected purpose or attempt" }
if ($retryEvents.Count -ne 0) { throw "Unexpected retry event" }
if ($writeEvents.Count -ne 1) { throw "write_file did not finish exactly once" }
if ($readEvents.Count -ne 1) { throw "read_file did not finish exactly once" }
if ((Get-Content -LiteralPath "$smokeWorkspace\phase9.txt" -Raw) -ne
    "PHASE9_TOOL_OUTPUT_PRIVATE") { throw "Unexpected file content" }
if ($events[-1].event_type -ne "run_finished") {
  throw "Run did not finish successfully"
}
if ($rawLog.Contains("PHASE9_MAIN_PROMPT_PRIVATE") -or
    $rawLog.Contains("PHASE9_TOOL_OUTPUT_PRIVATE")) {
  throw "Event Log leaked prompt or tool content"
}

$modelEvents | Select-Object event_type, @{
  Name="purpose"; Expression={$_.data.purpose}
}, @{
  Name="turn"; Expression={$_.data.turn}
}, @{
  Name="attempt"; Expression={$_.data.attempt}
} | Format-Table
```

### Smoke B：Subagent lineage

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$childWorkspace = "D:\tinyharness-smoke\phase9-child-$stamp"
$childLog = "D:\tinyharness-logs\phase9-child-$stamp.jsonl"

New-Item -ItemType Directory -Force $childWorkspace

cd D:\learn-claude-code\tinyharness

python -m tiny_harness `
  "PHASE9_PARENT_PROMPT_PRIVATE。父 Agent 必须只调用一次 task，把下面工作全部委派给子 Agent，父 Agent 不得直接调用文件工具：创建 phase9-child.txt，内容严格为 PHASE9_CHILD_OUTPUT_PRIVATE，并读取确认。子 Agent 完成后父 Agent 报告结果。" `
  --workspace $childWorkspace `
  --max-turns 6 `
  --subagent-max-turns 6 `
  --max-model-retries 2 `
  --max-context-chars 100000 `
  --event-log $childLog
```

检查父子 attempt、scope 和 task call ID：

```powershell
$events = @(Get-Content -LiteralPath $childLog | ForEach-Object {
  $_ | ConvertFrom-Json
})
$taskStarted = @($events | Where-Object {
  $_.event_type -eq "tool_started" -and $_.data.tool_name -eq "task"
})
if ($taskStarted.Count -ne 1) { throw "task did not start exactly once" }
$taskCallId = $taskStarted[0].data.tool_call_id

$parentModel = @($events | Where-Object {
  $_.event_type -in @("model_requested", "model_responded") -and
  $null -eq $_.data.agent_scope
})
$childModel = @($events | Where-Object {
  $_.event_type -in @("model_requested", "model_responded") -and
  $_.data.agent_scope -eq "subagent"
})
$retryEvents = @($events | Where-Object {
  $_.event_type -in @(
    "model_request_failed",
    "model_retry_scheduled",
    "model_retry_exhausted"
  )
})
$childWrites = @($events | Where-Object {
  $_.event_type -eq "tool_finished" -and
  $_.data.tool_name -eq "write_file" -and
  $_.data.agent_scope -eq "subagent" -and
  $_.data.outcome -eq "returned"
})
$childReads = @($events | Where-Object {
  $_.event_type -eq "tool_finished" -and
  $_.data.tool_name -eq "read_file" -and
  $_.data.agent_scope -eq "subagent" -and
  $_.data.outcome -eq "returned"
})
$taskFinished = @($events | Where-Object {
  $_.event_type -eq "tool_finished" -and
  $_.data.tool_name -eq "task" -and
  $_.data.outcome -eq "returned"
})
$parentFileTools = @($events | Where-Object {
  $_.event_type -eq "tool_finished" -and
  $_.data.tool_name -in @(
    "read_file", "write_file", "edit_file", "list_files"
  ) -and
  $null -eq $_.data.agent_scope
})
$rawLog = Get-Content -LiteralPath $childLog -Raw

if ($parentModel.Count -eq 0 -or $childModel.Count -eq 0) {
  throw "Missing parent or child model events"
}
if (@($parentModel | Where-Object {
  $_.data.purpose -ne "main" -or $_.data.attempt -ne 1
}).Count -ne 0) { throw "Unexpected parent model attempt" }
if (@($childModel | Where-Object {
  $_.data.purpose -ne "main" -or
  $_.data.attempt -ne 1 -or
  $_.data.parent_tool_call_id -ne $taskCallId
}).Count -ne 0) { throw "Invalid child scope lineage or attempt" }
if ($retryEvents.Count -ne 0) { throw "Unexpected retry event" }
if ($childWrites.Count -ne 1) { throw "Child write did not finish exactly once" }
if ($childReads.Count -ne 1) { throw "Child read did not finish exactly once" }
if ($taskFinished.Count -ne 1) { throw "task did not finish exactly once" }
if ($parentFileTools.Count -ne 0) { throw "Parent executed a file tool" }
if ((Get-Content -LiteralPath "$childWorkspace\phase9-child.txt" -Raw) -ne
    "PHASE9_CHILD_OUTPUT_PRIVATE") { throw "Unexpected child file content" }
if ($events[-1].event_type -ne "run_finished" -or
    $null -ne $events[-1].data.agent_scope) {
  throw "Parent run did not finish successfully"
}
if ($rawLog.Contains("PHASE9_PARENT_PROMPT_PRIVATE") -or
    $rawLog.Contains("PHASE9_CHILD_OUTPUT_PRIVATE")) {
  throw "Event Log leaked parent prompt or child tool content"
}

$childModel | Select-Object event_type, @{
  Name="scope"; Expression={$_.data.agent_scope}
}, @{
  Name="parent_call"; Expression={$_.data.parent_tool_call_id}
}, @{
  Name="turn"; Expression={$_.data.turn}
}, @{
  Name="attempt"; Expression={$_.data.attempt}
} | Format-Table
```

正常 smoke 没有异常 body，不能凭空验证异常日志脱敏；该 invariant 由离线故障测试使用私有异常 marker 验证。真实 smoke 能直接确认 prompt、工具内容和最终回答正文没有进入 Event Log。

### 真实 API 验收结果

2026-08-18 使用真实 DeepSeek API 完成两组验收。

Smoke A（主 Agent）：

- 12 条同 run ID 的有序事件；
- 3 次主模型逻辑请求，均为 `purpose=main, attempt=1`；
- finish reason 顺序为 `tool_calls → tool_calls → stop`；
- `write_file` 和 `read_file` 分别成功完成 1 次；
- 文件内容严格等于 `PHASE9_TOOL_OUTPUT_PRIVATE`；
- retry 事件为 0；
- Event Log 不包含 prompt marker 或工具内容 marker；
- 最后一条事件为父级 `run_finished`。

Smoke B（Subagent）：

- 20 条同 run ID 的有序事件；
- 父 Agent 两次逻辑请求均为 `purpose=main, attempt=1`；
- 子 Agent 三次逻辑请求均为 `purpose=main, attempt=1`；
- 所有子模型事件均带 `agent_scope=subagent`，并关联同一个真实父 task call ID；
- `task` 启动和成功完成各 1 次；
- 子 Agent 的 `write_file` 和 `read_file` 分别成功完成 1 次；
- 父 Agent 文件工具调用为 0；
- retry 事件为 0；
- 文件内容严格等于 `PHASE9_CHILD_OUTPUT_PRIVATE`；
- Event Log 不包含父 prompt marker 或子工具内容 marker；
- 最后一条事件为无 child scope 的父级 `run_finished`。

## 明确不做

- 工具重试或幂等性判断；
- fallback model；
- circuit breaker；
- 全局跨 Agent retry/token/time budget；
- adaptive concurrency 或请求队列；
- `max_tokens` 自动扩张；
- 对 `length`、`content_filter` 自动恢复；
- 多次 recursive reactive compaction；
- 独立摘要模型；
- OS 网络故障注入；
- 精确 token 计数。

## 参考来源

Phase 9 选择性参考并重新实现：

- `learn-claude-code-main/s08_context_compact` 的 API 拒绝后 reactive compaction 概念；
- `learn-claude-code-main/s15_integrated_harness` 中 429/overload 的 bounded backoff 与 prompt-too-long recovery 控制流。

TinyHarness 没有复制 integrated harness，也没有引入 fallback model、`max_tokens` escalation 或其他 Phase 9 范围外机制。Provider-neutral 错误契约、统一 `RecoveryExecutor`、物理 attempt 事件、明确 75% shrink margin，以及与现有 Event Log、Todo、Subagent 和字符预算的组合均按当前项目重新实现。没有读取或移植 Claude Code 产品源码。
