# TinyHarness

TinyHarness 是一个轻量、可读的 Coding Agent Harness Runtime。它不是完整的 Agent application，也不试图重新实现聊天产品；它关注的是模型产生的工具调用轨迹：哪些动作可以执行、失败后如何恢复、验证如何受控，以及长任务中的 Context 如何保持可用。

核心循环仍然很小：

```text
LLM
 ↓
Tool Calls
 ↓
response / tool validation
 ↓
Pre Tool Hooks → Permission
 ↓
Tool execution
 ↓
Post Tool Hooks
 ↓
Tool Results ───────────────┐
 ↓                         │
LLM ←──────────────────────┘
 ↓
Final Answer
```

Harness 的价值不在让循环本身变复杂，而是在循环周围建立明确的控制边界。模型提出一个动作，不意味着 Runtime 必须执行它。

## 核心设计

### 1. 执行与权限控制

模型响应会先经过协议校验，包括 Tool Call、参数形状和 Finish Reason；非法响应不会被提交到会话历史，也不会触发工具副作用。同一响应中的 Tool Calls 作为完整 batch 顺序执行，结果全部写回后才进入下一轮。

工具执行边界由三部分组成：

| 控制点 | 作用 |
|---|---|
| Permission | 每个 Tool Call 得到 `ALLOW`、`DENY` 或 `ASK` 决策 |
| Pre/Post Tool Hooks | 执行前阻断或执行后观察，不绕过 Permission |
| Workspace Boundary | 文件工具拒绝 workspace 逃逸；进程以 workspace 为 cwd 并继续受 Permission 约束 |

DENY 不只是一个布尔结果。Runtime 会把可执行的反馈作为普通 Tool Result 返回模型；连续出现同类权限拒绝时，还会要求模型停止重复尝试并重新规划。

### 2. 恢复与验证

TinyHarness 区分 Logical Turn 与 Physical Model Attempt。一次逻辑请求遇到暂时性 Provider 故障时，可以在不消耗新 Agent turn 的前提下重试；重试有固定上限，并支持指数退避、jitter 和受上限约束的 `Retry-After`。致命错误直接失败；如果模型持续返回工具调用而没有自然结束，Runtime 抛出 `MaxTurnsExceededError`。

验证使用可选的无参数 `run_tests()` capability：

```text
run_tests()
  → configured fixed argv
  → shell=False
  → workspace cwd
  → bounded timeout
  → PYTHONDONTWRITEBYTECODE=1
```

它来自一次具体的 Pilot 教训：只允许很窄的 Bash 命令会形成 verification loop；放宽 Bash 后，模型又可能自行拼装 verify script 或创建临时测试文件。将项目测试抽成 `run_tests()` 后，验证动作既可用又有明确边界。

`run_tests` 不是 completion detector。PASS 或 FAIL 都只是普通 Tool Result；模型仍需在下一轮自行决定继续工作还是返回 Final Answer。

### 3. Context 控制

Context 缩减不能破坏 Tool Call / Tool Result 的协议完整性。TinyHarness 因而把 Context 处理放在明确的 batch 边界上：

- 通过 character budget 控制发送给模型的消息与 tool schemas；
- 将大型 Tool Result 落盘，只在消息中保留有界 preview；
- 按完整历史块执行 trimming 和 compaction，不拆散 tool batch；
- 必要时请求 LLM summary，同时保留当前任务、运行状态和最新执行证据；
- Provider 返回 context-length 错误时，可压缩一次并重试同一 Logical Turn；
- 压缩流程未完整成功时，不提交半完成的 canonical history。

## 可选扩展

这些能力复用同一套 Runtime 边界，但不改变主循环的自然结束语义。

| 能力 | 作用 |
|---|---|
| Subagent | 使用独立 messages 执行子任务，共享 workspace，并继承 Permission、Hooks、Recovery、Context 与 `run_tests`；禁止嵌套 Subagent |
| Skills | 启动时只发现并注入有界 metadata，需要时再通过工具加载正文 |
| Memory | 从 workspace 持久记忆中按需选择；在 root Final Answer 路径提取新记忆，并在达到阈值时 consolidation |
| Todo | 保存 run-scoped 任务状态，并在多轮未更新时给出协议安全的 reminder |

## Reliability Eval：为什么重做评测

Reliability Eval 不只看最后的 workspace 是否正确，还检查 Agent 是怎样到达终点的。当前 Pilot 冻结了 6 个小型 Coding tasks；每个任务都有 visible tests、Agent workspace 外的 hidden grader、终局快照评分和 post-run final workspace grade。

下面的数字只描述这些冻结任务及当时的配置，不代表通用 Coding Agent 成功率。

| 阶段 | Verification 设计 | Bash DENY | 正确 workspace 但未结束 | 结果 |
| --- | --- | ---: | ---: | --- |
| v1 | 精确 Bash 命令 | 95 | 3 | 暴露 verification loop |
| v2 | 有限语义解析的 unittest 权限 + Recovery | 48 | 1 | 问题明显缓解 |
| v3 | `run_tests()` | 31 | 0 | 24 / 24 Verified |
| 当前 `main` | `run_tests()`，无 Goal Gate | 16（12 runs） | 0 | 12 / 12 Verified |

### 第一轮：最终代码正确，不等于轨迹可靠

第一轮共观察到 95 次 Bash Permission DENY。3 个 run 以 explicit failure 结束，但它们停止后的 workspace hidden grader 全部 PASS。

Trajectory audit 显示了共同模式：

```text
代码已经完成
 → Agent 继续验证
 → Bash DENY
 → 更换命令或编写 verify script
 → 再次 DENY
 → 没有返回 Final Answer
 → max_turns
```

这说明一部分 failure-to-stop 是 Harness 自己的 Permission / Recovery protocol 制造的，而不是代码任务没有完成。

### 第二轮：修复 Permission / Recovery protocol

随后加入有限语义解析的 unittest 权限、可执行的 DENY 反馈、连续拒绝后的重新规划提示，并明确 `MaxTurnsExceededError` 的终止语义。

```text
Bash DENY                         95 → 48
workspace 已正确但未自然结束       3 → 1
```

轨迹仍暴露出两个问题：测试成功后模型可能继续验证；通用 Bash 也允许模型创建 `test_edge_tmp.py`、sanity script 等额外验证副作用。

### 第三轮：将验证收敛为受控能力

把项目测试从通用 Bash 中抽成 `run_tests()` 后，24 个 run 的结果为：

| 指标 | 结果 |
|---|---:|
| Verified | 24 / 24 |
| Premature terminal completion | 0 |
| Explicit failure | 0 |
| Final hidden PASS | 24 / 24 |
| `run_tests` calls | 25 |

其中 23 个 run 只调用了一次 `run_tests`；另一个 run 调用了两次，但两次之间确实发生了代码修改。轨迹中没有再发现 verification loop 或临时验证文件。

### 当前公开版确认

移除当前 Pilot 中未观察到有效干预的 Goal Gate 后，当前 `main` 使用普通 Agent Loop 又做了 12-run 确认实验：

| 指标 | 结果 |
|---|---:|
| Verified | 12 / 12 |
| Premature terminal completion | 0 |
| Explicit failure | 0 |
| `MaxTurnsExceededError` | 0 |
| Final hidden PASS | 12 / 12 |
| `run_tests` calls | 12 |

当前公开 Eval 因此只保留一个普通 `harness` profile。Runner 仍记录 Permission DENY、Tool sequence、turn 数、`run_tests` 调用和 final workspace grade，便于继续审计轨迹。

## 设计结论

1. **Harness 的控制机制本身也可能制造失败。** Permission 太严格时，Agent 可能完成代码却无法自然收敛。
2. **更复杂的完成控制器不一定是第一答案。** 对“workspace 已正确但 Agent 停不下来”，这轮实验最终通过改善验证动作空间解决。
3. **Verification 更适合作为受控能力。** `run_tests()` 比让 Agent 在通用 Bash 中临时设计验证流程更清楚，也更容易审计。
4. **评测必须包含轨迹。** 最终成功率之外，还需要观察 Permission DENY、MaxTurns、临时文件、工具顺序和最终 hidden workspace grade。

## Quick Start

### 安装

TinyHarness 要求 Python 3.10+：

```powershell
python -m pip install -e .
```

### 配置 Provider

CLI 使用兼容 Chat Completions 的 Provider。API key 必填；model 和 base URL 可选，默认值来自源码配置。

```powershell
$env:TINYHARNESS_API_KEY = "..."
$env:TINYHARNESS_MODEL = "deepseek-v4-flash"       # 可选
$env:TINYHARNESS_BASE_URL = "https://api.deepseek.com"  # 可选
```

### 运行任务

```powershell
python -m tiny_harness "修复测试并说明结果" --workspace .
```

省略 task 会进入复用同一个 `AgentSession` 的交互模式：

```powershell
python -m tiny_harness --workspace .
```

常用控制项包括 `--max-turns`、`--max-model-retries`、`--max-context-chars`、`--no-context-compaction`、`--subagent-max-turns`、`--event-log` 和 `--memory`。

### 运行 Reliability Pilot

Pilot 会调用配置的真实 Provider：

```powershell
python -m evals.pilot run `
  --repetitions 2 `
  --results-dir evals/results/pilot
```

离线 deterministic recovery 与 safety checks 不调用真实 API：

```powershell
python -m evals.run --suite offline --results-dir evals/results/offline
```

Eval 产物与指标定义见 [`evals/README.md`](evals/README.md)。

## 项目结构

```text
tiny_harness/
  agent/       # Agent Loop、run context、session、subagent orchestration
  models/      # Provider contract 与 Chat Completions adapter
  runtime/     # Permission、Recovery、Context、Hooks、Skills、Memory、events
  tools/       # filesystem、shell、run_tests、task、todo、compact 等工具
evals/         # Pilot cases、hidden grading、runner 与 metrics
tests/         # scripted providers 和 deterministic regression tests
```

## 测试

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

测试使用 scripted provider 或本地 fixture，不需要真实 API。

早期 Goal Gate / Stop Proposal 实验版本保存在 `reliability-eval-v3-goal-gate` tag。
