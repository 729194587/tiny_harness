# TinyHarness Reliability Eval

当前 Eval 使用单一普通 `harness` profile，评估 Runtime 的自然 Final Answer 路径以及 Permission、Recovery、Context 和验证能力。

## Pilot

Pilot 保留 6 个 coding tasks、seed workspace、visible tests 和 workspace 外的 hidden grader。Runner 为每次运行创建独立目录，Agent 只能访问 `workspace/`；hidden grader 在 root Agent 自然结束时对物理快照评分，并在运行完全结束后再做一次 final workspace grade。

```powershell
python -m evals.pilot run `
  --repetitions 2 `
  --results-dir .\pilot-results
```

选择部分任务时重复 `--task`：

```powershell
python -m evals.pilot run `
  --task free_shipping_policy `
  --task contact_name_order `
  --results-dir .\pilot-results
```

离线重建报告不会调用模型：

```powershell
python -m evals.pilot summarize .\pilot-results
```

## 结果与诊断

每个 run 保存：

- `events.jsonl`：不含 prompt、tool payload 或 grader output 的 Runtime metadata。
- `terminal_grade.json`：自然 Final Answer 时的外部 hidden grade。
- `post_run_grade.json`：Agent 完全停止后的 final workspace diagnostic。
- `run.json`：归一化 ledger。

根目录报告为 `summary.json`、`summary.md` 与 `runs.csv`，包含：

- Verified
- Premature terminal completion
- Explicit failure
- MaxTurnsExceeded
- Tool denied 与 Bash verification denial
- `run_tests` calls
- turn、tool call 与 model attempt 数
- final workspace grade

grader 的 stdout/stderr、failure reason 与 private fixture 内容不会进入 Agent messages、Runtime events 或 workspace。grading error 记为 `invalid_run`，不会伪装成任务失败。

## `run_tests` 协议

Eval 从 case 的受限 unittest 配置创建 `SubprocessTestRunner`。测试执行通过无参数 `run_tests()` capability 完成；相同 unittest 命令经 Bash 调用会被 DENY，并得到指向 `run_tests` 的 actionable feedback。

`run_tests` 使用固定 argv、workspace cwd、`shell=False`、timeout 与 `PYTHONDONTWRITEBYTECODE=1`。它的结果始终是普通 Tool Result，不会自动结束 Agent Loop。

## Offline suite

不调用真实 API 的 deterministic recovery 与 safety checks：

```powershell
python -m evals.run --suite offline --results-dir .\eval-results
```
