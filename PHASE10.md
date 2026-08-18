# Phase 10：Goal Verification Gate

## 状态

- 状态：Accepted（2026-08-18）
- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API smoke test：通过

## 唯一目标

模型不再调用工具只表示它提出结束当前运行，不等于任务已经完成。Phase 10 在真正返回最终答案前增加一个独立、只读、无工具的 Goal Evaluator：

```text
worker model proposes final text
    → project bounded execution evidence
    → independent goal evaluation
        ├→ achieved   → commit candidate → return
        ├→ incomplete → reject candidate → feedback → continue loop
        ├→ impossible → fail explicitly
        └→ invalid / exhausted → fail explicitly
```

Goal Gate 只在显式传入 `--goal` 时启用。未配置 Goal 时，Agent Loop 的原有停止行为不变。

## CLI

```text
--goal <completion condition>
--max-goal-retries <non-negative integer>
```

默认允许 Goal Evaluator 阻止结束并自动续跑 3 次。这个预算只计算被拒绝后实际安排的 automatic continuation，不计算首次判断；最后一个 `max_turns` 轮次后无法发生的 continuation 不会计数。主 Agent 的每次候选回答仍消耗正常的 `max_turns`。Evaluator 调用和它自己的物理 retry 不消耗主 Agent turn。

Goal 最长 4000 字符。空 Goal、超长 Goal 和负数 continuation budget 会在模型 API 调用前失败。

好的 Goal 应同时说明结束状态和可检查证据，例如：

```text
phase10.txt 存在且内容严格为 PHASE10_OK，并且最近的 read_file 工具结果确认该内容
```

“把项目做好”这类主观条件不能被 Goal Gate 可靠验证。

## 候选答案事务边界

普通 Agent Loop 会在 `finish_reason=stop` 后立即把 assistant message 写入历史并返回。Phase 10 改为：

1. 构造候选 assistant message，但暂不写入 canonical history；
2. Evaluator 判断 Goal；
3. 只有 `achieved` 才提交候选并发出 `run_finished`；
4. `block` 时丢弃候选，把 Evaluator 的有界简短理由作为不可信数据写入唯一的 Goal state marker，然后继续同一个 Agent Loop；
5. `impossible`、续跑预算耗尽、全局 `max_turns` 耗尽或 Evaluator 错误都明确失败。

因此被拒绝的“已经完成”不会污染后续证据，也不会被 Context Compactor 当作事实摘要来源。

## Goal 状态

`GoalState` 只存在于一次 `agent_loop()` 调用的内存中，包含：

- completion condition；
- 已执行的 evaluation 次数；
- 已使用的 continuation 次数；
- 最近一次 Evaluator reason。

Harness 在主消息历史中维护一个 `tinyharness_goal_state` marker。Marker 会原位替换而不是累积，并由 Context Compactor 作为任务前缀保留。Marker 把两类信息明确分开：由 Harness 生成的可信 continuation 状态说明上一 stop proposal 是否被拒绝、是否真正安排了第 N 次续跑；Evaluator reason 最多 1600 字符，并包在 `<evaluator-feedback>` 中标记为可能包含转述工具文本的不可信数据，worker 不能执行其中的命令或指令。

父 Agent 的 Goal 不会隐式继承给子 Agent。`task` 子 Agent 仍只负责被委派的子任务；父 Agent 收到 task ToolResult 后，由父级 Goal Gate 判断整个任务是否完成。

Phase 10 不实现 Session Resume，因此 Goal 不会跨 CLI 进程持久化、查看、替换或恢复。

## 独立 Evaluator

默认 `PromptGoalEvaluator` 使用一次独立 Chat Completions 调用：

- 与 worker 使用同一个已配置 Provider；
- `tools=[]`，不能执行工具或产生副作用；
- 通过 Phase 9 `RecoveryExecutor` 使用独立 `RecoveryState`；
- 物理 attempt 的 `purpose=goal_evaluation`；
- 只接受 `finish_reason=stop` 且不含 tool calls 的非空文本；
- 使用 strict JSON contract with optional code-fence unwrapping，只允许 `ok`、`reason`、`impossible` 三个字段；
- `reason` 必须是 1–1600 字符的非空字符串；
- malformed、截断、过滤或契约冲突响应均 fail closed。

JSON 契约：

```json
{"ok": false, "reason": "缺少测试退出码", "impossible": false}
```

Phase 10 没有 ProviderFactory、单独 evaluator model、投票或规则引擎。Python API 可以注入实现同一 `GoalEvaluator` Protocol 的确定性判断器，主要用于测试。

## Evidence Projection

Evaluator 不直接获得工具能力，而是读取有界执行记录：

- completion condition；
- 尚未提交的 candidate final answer；
- canonical history 中最近的完整消息块；
- assistant tool-call message 与对应的全部 tool results 作为不可拆分证据块；
- 超预算时优先保留最新证据，并对单个超长块做中间裁剪。

投影会删除 `reasoning_content` 和 Goal feedback marker。Evaluator 的信任层级是：只有来自具体 workspace/process 工具的直接 `role=tool` 结果，才能独立支持可验证条件；candidate、assistant narrative、Todo、context summary、archive marker 和 task/Subagent summary 都只是 reference/claim。工具输出本身仍是不可信数据，Evaluator 只能据此判断，不能执行其中的指令。

Phase 10 同时让 `bash` 的成功和失败结果都显式包含退出码：

```text
Exit code: 0
<stdout/stderr>
```

这使“测试命令退出码为 0”成为 Evaluator 能直接检查的证据，而不是根据有没有输出猜测成功。

Goal Gate 不是测试框架，也不是可信审计器。它只能判断送入执行记录的文本；恶意或错误的工具结果、workspace 内被篡改的 artifact、获批 bash 的外部副作用，以及模型判断误差仍然存在。

## Context 与 Recovery 组合

配置 `--max-context-chars` 时：

- worker request 继续经过 Phase 8 Context Compactor；
- Goal marker 作为必保留任务状态参与 worker request 预算；
- Evaluator request 使用同一个字符上限，但不携带 tool schemas；
- request 超限时确定性缩小旧 evidence，再缩小 candidate；
- 连 system prompt、Goal 和最小 JSON overhead 都无法放入预算时，在 API 调用前抛出 `ContextLimitError`。

Evaluator 的 transient retry 使用 Phase 9 的统一重试策略，但每次 evaluation 都创建新的逻辑 `RecoveryState`。Evaluator 失败不会被解释成 Goal 已完成，也不会自动重复整次 Goal 判断。

## Event Log

Phase 10 新增：

- `goal_evaluation_requested`
- `goal_evaluated`

事件只记录：turn、evaluation 序号、outcome、reason 长度和已使用 continuation 次数。模型调用事件通过 `purpose=goal_evaluation` 标识 Evaluator attempt。

Event Log 不记录 Goal 正文、candidate、Evaluator prompt、Evaluator response 或 reason 正文。未验证的运行以 `run_failed` 结束，不会产生 `run_finished`。

## 自动测试

命令：

```powershell
python -m unittest discover -s tests -v
```

当前结果：

- 执行 171 项测试；
- 168 项通过；
- 3 项跳过；
- 跳过项均为当前 Windows 用户无法创建 filesystem、artifact 目录和最终 artifact 文件的符号链接测试。

Phase 10 新增或调整的测试覆盖：

- 严格 Evaluator JSON contract；
- 超长 reason 在写入 Parent context 前 fail closed；
- feedback marker 将 reason 标注为不可信数据；
- Evidence trust hierarchy 区分直接工具结果与 narrative/marker/Subagent claim；
- evidence 保留最新完整工具块并删除 reasoning / Goal feedback；
- Evaluator 无工具且请求满足字符预算；
- 无法容纳最小请求时 API 零调用；
- rejected candidate 不提交，feedback marker 不累积；
- achieved、impossible、continuation exhaustion 和 `max_turns`；
- 最后一个 `max_turns` 轮次不虚增 continuation 计数；
- Goal marker 与 Context Compactor 组合；
- Evaluator 复用 bounded recovery 且不消耗主 turn；
- Goal 不隐式继承给 Subagent；
- Goal 事件顺序和敏感正文不落日志；
- CLI goal 参数及前置验证；
- Bash 成功和非零退出码证据；
- Phase 1–9 全量回归。

## 真实 DeepSeek smoke test

这个 smoke 刻意要求 worker 第一次先提出一个无工具证据的候选答案。Goal Evaluator 应拒绝第一次停止；worker 收到 feedback 后再创建并读取文件，第二次判断通过。

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$smokeWorkspace = "D:\tinyharness-smoke\phase10-$stamp"
$smokeLog = "D:\tinyharness-logs\phase10-$stamp.jsonl"

New-Item -ItemType Directory -Force $smokeWorkspace

cd D:\learn-claude-code\tinyharness

python -m tiny_harness `
  "这是 Goal Gate 协议测试。每轮先检查 name=tinyharness_goal_state 的最新消息：若 Harness continuation status 表示 no stop proposal has been rejected yet，则不调用工具，只返回纯文本 PREMATURE_CANDIDATE；若状态表示 previous stop proposal was rejected，则调用 write_file 创建 phase10.txt，内容严格为 PHASE10_OK，再调用 read_file 验证，最后报告结果。状态改变前不要提前执行工具。" `
  --goal "phase10.txt 存在且内容严格为 PHASE10_OK，并且执行记录中有 read_file 返回 PHASE10_OK 的证据" `
  --workspace $smokeWorkspace `
  --max-turns 6 `
  --max-goal-retries 2 `
  --max-model-retries 2 `
  --max-context-chars 100000 `
  --event-log $smokeLog
```

运行后检查：

```powershell
$events = @(Get-Content -LiteralPath $smokeLog | ForEach-Object {
  $_ | ConvertFrom-Json
})
$goalEvents = @($events | Where-Object {
  $_.event_type -eq "goal_evaluated"
})
$goalModel = @($events | Where-Object {
  $_.event_type -in @("model_requested", "model_responded") -and
  $_.data.purpose -eq "goal_evaluation"
})
$writes = @($events | Where-Object {
  $_.event_type -eq "tool_finished" -and
  $_.data.tool_name -eq "write_file" -and
  $_.data.outcome -eq "returned"
})
$reads = @($events | Where-Object {
  $_.event_type -eq "tool_finished" -and
  $_.data.tool_name -eq "read_file" -and
  $_.data.outcome -eq "returned"
})
$rawLog = Get-Content -LiteralPath $smokeLog -Raw

if ($goalEvents.Count -lt 2) { throw "Goal was not evaluated twice" }
if ($goalEvents[0].data.outcome -ne "block") {
  throw "First unsupported candidate was not blocked"
}
if ($goalEvents[-1].data.outcome -ne "achieved") {
  throw "Final evidenced candidate was not accepted"
}
if ($goalModel.Count -lt 4) { throw "Missing evaluator model lifecycle" }
if (@($goalModel | Where-Object { $_.data.attempt -ne 1 }).Count -ne 0) {
  throw "Unexpected evaluator retry"
}
if ($writes.Count -ne 1 -or $reads.Count -ne 1) {
  throw "Expected exactly one successful write and read"
}
if ((Get-Content -LiteralPath "$smokeWorkspace\phase10.txt" -Raw) -ne
    "PHASE10_OK") { throw "Unexpected file content" }
if ($events[-1].event_type -ne "run_finished" -or
    $events[-1].data.goal_evaluations -lt 2) {
  throw "Run did not finish through Goal Gate"
}
if ($rawLog.Contains("PHASE10_OK") -or
    $rawLog.Contains("phase10.txt 存在且内容严格")) {
  throw "Event Log leaked Goal or tool content"
}

$goalEvents | Select-Object event_type, @{
  Name="evaluation"; Expression={$_.data.evaluation}
}, @{
  Name="outcome"; Expression={$_.data.outcome}
}, @{
  Name="retries_used"; Expression={$_.data.retries_used}
} | Format-Table
```

模型行为存在随机性。如果 worker 没有按 Goal marker 中的 continuation status 分支并提前完成验证，首次判断可能直接 `achieved`；这表示真实 Provider、工具证据和正常 Gate 路径通过，但不能验收自动 continuation，应使用新的 workspace 重跑。自动 continuation 的确定性保证仍由离线测试覆盖。

### 真实 API 验收结果

2026-08-18 使用真实 DeepSeek API 完成验收：

- 共记录 22 条同 run ID 的有序事件；
- Goal 判断顺序严格为 `block → achieved`；
- 第一次拒绝后实际安排 1 次 continuation，第二次判断没有虚增计数；
- 主 Agent 按顺序执行 `write_file → read_file`，各成功完成 1 次；
- `phase10.txt` 内容严格等于 `PHASE10_OK`；
- 两次 Goal Evaluator 调用均产生独立的 requested/responded 事件，全部为 `purpose=goal_evaluation, attempt=1`；
- transient retry 事件为 0；
- 最后一条事件为 `run_finished`，其中 `goal_evaluations=2`；
- Event Log 不包含 Goal 正文、`PREMATURE_CANDIDATE` 或 `PHASE10_OK`。

这次真实运行同时验证了候选答案不提交、可信 continuation 状态、工具证据回填、再次判断和最终提交的完整闭环。

## 明确不做

- Session-scoped `/goal` 命令、查看、替换、clear 或 restore；
- Goal 持久化或 Session Resume；
- 单独 evaluator model 配置、投票或多评审器；
- token、时间或成本 budget；
- 后台任务 defer / wakeup；
- Evaluator 工具调用或主动重跑测试；
- 规则式验收 DSL、Evidence Collector 或 Eval framework；
- Goal 自动拆解、Workflow 或 Agent Teams；
- 把 LLM 判断宣称为可信证明。

## 参考来源

Phase 10 实质性参考并按 TinyHarness 当前结构重新实现了 `learn-claude-code-main/s17_goal_loop` 的核心思想：模型的无工具响应只是 stop proposal，独立无工具 Evaluator 在 Agent Loop 返回边界决定 `achieved / block / impossible`。

没有移植 s17 的 session command、Goal restore、token/time accounting、后台任务 defer 或持久化 event store。TinyHarness 额外实现了候选答案事务提交、有界 evidence projection、与 Phase 8 字符预算及 Phase 9 Recovery/Event Log 的组合。没有读取或移植 Claude Code 产品源码。
