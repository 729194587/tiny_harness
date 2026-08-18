# TinyHarness Reliability Eval v1

TinyHarness 使用一个专用、同步、标准库实现的 runner，回答三个问题：

1. Harness 是否降低 `false_success`；
2. 是否能恢复确定性注入的故障；
3. 可靠性增加了多少模型调用和执行开销。

它不是通用 benchmark SDK，也不使用 LLM-as-judge。

## 快速运行

离线场景不需要 API Key：

```powershell
cd D:\learn-claude-code\tinyharness
python -m evals.run --suite offline
```

结果写到 `evals/results/<timestamp>/report.json` 和 `report.md`。该目录已加入 `.gitignore`。

开发阶段运行一次真实 DeepSeek 对比：

```powershell
python -m evals.run --suite real --repetitions 1
```

只运行一个 case：

```powershell
python -m evals.run `
  --suite real `
  --case single_file_slugify `
  --repetitions 1
```

正式报告运行两个 profile、5 个 case、每组重复 3 次，共 30 个 Agent run：

```powershell
python -m evals.run `
  --suite all `
  --profile both `
  --repetitions 3
```

真实评测读取与主 CLI 相同的环境变量：

- `TINYHARNESS_API_KEY`；
- `TINYHARNESS_MODEL`；
- `TINYHARNESS_BASE_URL`。

不要在尚未确认 API 成本时直接运行正式 30-run 命令。

## 三类结果

### Real Coding

`basic_ablation` 与 `reliable` 使用相同模型、seed workspace、工具 schemas、Subagent 能力、Bash 白名单和 `max_turns`。

差异只有：

| Profile | Goal Gate | transient retry |
|---|---:|---:|
| `basic_ablation` | 关闭 | 0 |
| `reliable` | 开启 | 2 |

Real Coding 不启用 Context Compaction，因此两个 profile 都没有 `compact` 工具。`basic_ablation` 是当前 Runtime 的消融配置，不是早期 Runtime 快照。

### Controlled Failure Recovery

使用 Scripted Provider 确定性触发：

- transient server failure；
- context rejection；
- premature final answer。

每项必须记录 `fault_expected=true` 和 `fault_triggered=true`。没有实际触发故障的 run 不能计为 recovery success。

### Safety Invariants

只验证当前 Runtime 必须始终成立的性质：

- Permission deny 不产生文件副作用；
- 非法 finish reason 携带 write call 时不执行工具。

Invariant 不进入 Basic/Reliable 提升百分比。

## External Grader

每个真实 run 使用：

```text
run-root/
├─ workspace/       Agent workspace
├─ hidden_grader/   external hidden tests
├─ events.jsonl
└─ result metadata
```

Runner 在 Agent 开始前复制 seed 和 hidden grader。Agent 只获得 `workspace/`。结束后 Runner：

1. 比较 hidden grader 前后摘要；
2. 在 workspace 外启动 hidden unittest；
3. 通过环境变量把 workspace 路径交给 grader；
4. 只记录退出码，不把 grader stdout 写进报告。

Real Coding 的 Bash 使用精确字符串白名单，两个 profile 完全相同。它避免普通 Agent 命令访问 hidden grader，但不是 OS sandbox；恶意 shell 进程隔离不属于当前 Eval v1。

## 指标语义

- `verified_success`：Agent 正常返回、hidden grader 通过且 grader 未被改动；
- `false_success`：Agent 正常返回，但 external grader 或副作用检查失败；
- `explicit_failure`：Agent 明确抛错或达到限制，没有声称成功；
- `recovery_success`：确定性故障已触发，且 Reliable 恢复完成；
- `side_effect_violation`：workspace 外 hidden grader 被改动；
- `main/goal/summary_model_attempts`：按 Event Log `purpose` 统计物理请求；
- `turns/retries/continuations/tool_calls`：只来自控制流 metadata。

报告不保存 API Key、prompt、reasoning、工具正文、最终答案或异常正文。

离线命令在 fault 未触发、Reliable 未恢复或 Safety Invariant 失败时返回非零。Real Coding 的任务失败是评测结果，不会让 runner 提前停止。
