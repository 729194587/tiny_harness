# Phase 11：Reliability Eval v1

## 状态

- 状态：Completed
- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 确定性 offline eval：通过
- 真实 API 评测：通过（30 个 Real Coding run）

## 唯一目标

Phase 11 不再增加 Agent Runtime 功能，而是为 Phase 1–10 建立可复现证据，回答：

1. TinyHarness 是否降低错误成功声明；
2. 是否能够恢复指定且确认发生的故障；
3. 可靠性机制增加了多少物理模型调用和执行轮次。

```text
Eval Case
    → fresh run root
        ├→ workspace/       Agent 可访问
        ├→ hidden_grader/   Agent workspace 外
        └→ events.jsonl     Agent workspace 外
    → Agent Run
    → external deterministic grader
    → normalized metrics
    → JSON + Markdown report
```

正确性只由 hidden test、文件状态、退出码和副作用检查决定。Agent 最终回答和 Goal Evaluator 结论都不能直接判定 correctness。

## 报告分区

### 1. Real Coding：Basic vs Reliable

5 个小型 coding fixtures：

| Case | 主要风险 |
|---|---|
| `single_file_slugify` | 单文件逻辑与隐藏边界输入 |
| `multi_file_currency` | 多文件一致性和消除硬编码 |
| `duplicate_edit_anchor` | 非唯一 edit anchor 的错误位置修改 |
| `wrong_path_recovery` | 初始路径假设错误后的工具恢复 |
| `verified_palindrome` | 必须执行验证，不能只口头声称完成 |

每个 case 有 visible workspace tests 和 workspace 外 hidden tests。Seed 必须被 hidden grader 拒绝，仓库内参考修复必须通过。

`basic_ablation` 不是历史 Phase 1，而是当前 Runtime 关闭可选可靠性机制后的消融配置。

| 配置 | basic_ablation | reliable |
|---|---:|---:|
| Goal Gate | 关闭 | 开启 |
| transient retry | 0 | 2 |
| Context Compaction | 关闭 | 关闭 |
| Subagent | 开启 | 开启 |
| 工具 schemas | 相同 | 相同 |
| Permission/Bash 白名单 | 相同 | 相同 |

Real Coding 故意不触发 Context Compaction，避免 `compact` 工具造成能力差异。Subagent 在两个 profile 中都可用，不能把能力增强混成可靠性提升。

开发运行默认每组 1 次；正式报告按每 case/profile 重复 3 次，共执行 30 个真实 Agent run。由于样本仍小，报告展示计数和均值，不把结果包装成通用模型能力基准。

### 2. Controlled Failure Recovery

Scripted Provider 和注入式 Goal Evaluator 保证故障确定发生：

| Scenario | basic_ablation | reliable |
|---|---|---|
| transient Provider failure | 明确失败 | retry 后返回 |
| context rejection | 明确失败 | reactive compact 后返回 |
| premature final answer | false success | Goal block 后写入证据并完成 |

每个结果都必须满足：

```json
{
  "fault_expected": true,
  "fault_triggered": true
}
```

`fault_triggered=false` 时 scenario 无效，runner 返回非零。Reliable 未按预期恢复时也返回非零。故障注入不依赖 DeepSeek 随机地产生 5xx、context overflow 或 premature stop。

当前确定性结果：

| Scenario | Basic recovered | Reliable recovered |
|---|---:|---:|
| transient retry | 0 | 1 |
| reactive context | 0 | 1 |
| premature goal | 0 | 1 |

### 3. Safety Invariants

Invariant 只报告通过或失败，不进入 Basic/Reliable 提升数字：

| Invariant | 结果 | Side effect |
|---|---:|---:|
| Permission deny 不创建文件 | PASS | 0 |
| `length + write_file` 不执行工具 | PASS | 0 |

这两项通过 Fake Provider 和真实 Agent Loop 确定性执行。

## External Grader 边界

Runner 为每次 Real Coding run 创建互不复用的目录：

```text
results/<run>/runs/<case>-<profile>-<repetition>/
├─ workspace/
├─ hidden_grader/
└─ events.jsonl
```

hidden grader 不复制进 Agent workspace。Runner 在 Agent 运行前计算其文件摘要，运行后先检查摘要，再启动独立 Python unittest 进程。`verified_success` 同时要求：

- Agent 正常返回；
- hidden grader 退出码为 0；
- hidden grader 没有被修改。

文件工具的 workspace boundary 无法访问 grader。Real Coding 额外使用 eval-only Bash exact allowlist，只允许 case 声明的 visible test 命令；其他 bash 调用作为 Permission denied ToolResult 返回。

这不是恶意代码沙箱。TinyHarness 仍然没有 OS 级文件、进程或网络隔离；case 文件和 Bash 白名单被视为仓库维护者提供的可信评测配置。

## 指标

每个 run 的 `EvalResult` 记录：

- `verified_success`；
- `false_success`；
- `explicit_failure`；
- `recovery_success`；
- `invariant_passed`；
- `side_effect_violation`；
- `fault_expected / fault_triggered`；
- `main / goal / summary / total_model_attempts`；
- `turns / retries / continuations / tool_calls`；
- grader exit code、elapsed milliseconds 和安全的 error type。

指标只读取 Event Log metadata，不保存 prompt、reasoning、工具正文、最终答案或异常正文。Phase 11 没有 token/cost 计数，因此开销结论只使用模型调用次数、turn 和 wall-clock 辅助数据。

`false_success` 的定义是 Agent 正常返回但 external grader 未验证通过，包括 hidden tests 失败或 grader 被修改。Agent 修改了代码但最终明确失败，不计 verified success，也不计 false success。

## 报告

每次命令生成：

- `report.json`：逐 run 结构化结果；
- `report.md`：三块可读报告。

Markdown 包含：

1. Real Coding profile 汇总和逐 run 表；
2. Controlled Failure Recovery 的 fault/recovery 表；
3. Safety Invariants 表。

原始结果写入 `evals/results/` 并被 gitignore。正式真实结果通过人工核查后，把汇总结论写回本文件，而不是提交可能庞大的每次运行 workspace。

## 运行

不需要 API 的离线验收：

```powershell
cd D:\learn-claude-code\tinyharness
python -m evals.run --suite offline
```

一次低成本真实开发运行：

```powershell
python -m evals.run --suite real --repetitions 1
```

正式 30-run 对比：

```powershell
python -m evals.run `
  --suite all `
  --profile both `
  --repetitions 3
```

可以用 `--case <id>` 重复指定 case，或用 `--results-dir <path>` 指定全新的输出目录。已存在的单 run 目录会明确失败，防止覆盖以前结果。

## 自动测试

Phase 11 定向测试覆盖：

- case JSON 严格加载、唯一且路径安全的 ID；
- Event Log purpose/turn/retry/continuation/tool metrics；
- Bash exact allowlist；
- workspace 与 hidden grader sibling 隔离；
- hidden grader mutation 检测；
- 5 个 seed 全部失败、5 个参考修复全部通过；
- 3 个受控故障的 Basic/Reliable 确定结果；
- 2 个 Safety Invariant 无副作用；
- 三段式 Markdown 和 JSON 输出；
- Basic/Reliable 的工具、Subagent、Permission 和 context 能力相同。

命令：

```powershell
python -m unittest discover -s tests -v
```

当前完整结果：

- 执行 181 项测试；
- 178 项通过；
- 3 项跳过；
- 跳过项仍是当前 Windows 用户无法创建 filesystem、artifact 目录和最终 artifact 文件的符号链接测试。

## 真实 API Pilot

2026-08-18 使用 `deepseek-v4-flash` 对 `single_file_slugify` 执行一次双 Profile pilot：

| Profile | Hidden grader | False success | Main attempts | Goal attempts | Total attempts | Turns | Tool calls | Elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `basic_ablation` | PASS | 0 | 7 | 0 | 7 | 7 | 7 | 11.8s |
| `reliable` | PASS | 0 | 8 | 1 | 9 | 8 | 8 | 18.9s |

两个 run 的 grader exit code 都为 0，均无 explicit failure、side-effect violation、retry、continuation 或 summary call。`report.json` 不包含任务 prompt、reasoning 或 API key。

这个 pilot 验证了 fresh workspace、双 Profile、Bash 白名单、hidden grader、Event Log 指标和报告生成的真实服务链路。单 case、单次重复不能回答总体 false-success 降幅，也不能把本次 `+2` 模型调用泛化为稳定开销。

### 5-case Development Run（pre-freeze）

随后使用 5 个 case、两个 profile、各重复 1 次执行 10-run 开发评测：

| Profile | Verified | False success | Explicit failure | Avg main | Avg goal | Avg total |
|---|---:|---:|---:|---:|---:|---:|
| `basic_ablation` | 4/5 | 0 | 1 | 6.6 | 0.0 | 6.6 |
| `reliable` | 5/5 | 0 | 0 | 7.2 | 1.0 | 8.2 |

全部 10 个 run 的 hidden grader 都返回 exit code 0，没有 side-effect violation、retry、continuation、summary call 或报告正文泄露。唯一未计 verified success 的 run 是 `verified_palindrome/basic_ablation`：实现和 hidden tests 已通过，但 Agent 在第 8 个主模型 turn 更新 Todo 后达到当时的 `max_turns=8`，没有机会提交 final answer，因此被正确分类为 `explicit_failure`，而不是 false success。

这个结果说明初始 8/9 turn cap 会把正常的读取、修复、测试、Todo 更新流程截断，是 eval 配置 artifact。正式重复评测前，将 5 个 case 的 `max_turns` 统一冻结为 12；pre-freeze 的 4/5 vs 5/5 不作为最终提升结论。正式结果必须使用更新后的同一 suite 配置，不能与本次数字直接合并。

## 正式评测结果

2026-08-18 使用 `deepseek-v4-flash`，在冻结后的同一 suite 配置上运行 5 个 case、两个 profile、各重复 3 次，共 30 个真实 Agent run。报告位于本地忽略目录 `evals/results/20260818-110303/`。

### Real Coding

| Profile | Verified | False success | Explicit failure | Avg main | Avg goal | Avg total | Avg turns | Avg elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `basic_ablation` | 15/15 | 0 | 0 | 6.9 | 0.0 | 6.9 | 6.9 | 12.3s |
| `reliable` | 15/15 | 0 | 0 | 7.2 | 1.0 | 8.2 | 7.2 | 17.1s |

全部 30 个 external hidden grader 的退出码均为 0。两个 profile 都达到 100% verified success，因而这组小样本真实任务没有观察到 false success 差异，不能据此宣称 Reliable 降低了真实 coding task 的失败率。

Reliable 平均增加 1.3 次物理模型调用（约 18.8%），其中每个 run 固定增加一次 Goal Evaluator 调用，main attempts 平均增加 0.3；平均 turns 增加 0.3。平均 wall-clock 从 12.3s 增至 17.1s（约 39.7%），但它受网络和服务端延迟影响，只作为辅助观测，不作为稳定性能结论。

### Controlled Failure Recovery

- 三种故障的 `fault_triggered` 均为 true；
- Basic 对 transient Provider failure 和 context rejection 明确失败，对 premature final answer 产生 false success；
- Reliable 对 transient retry、reactive context compact 和 Goal continuation 均恢复成功，3/3；
- 这些结果来自确定性故障注入，证明的是指定恢复路径，不代表未知线上故障的总体恢复率。

### Safety Invariants

- Permission deny 不创建文件：PASS；
- 非法 `length + write_file` 响应不执行工具：PASS；
- 两项均确认故障已触发，side-effect violation 为 0。

30 份真实运行 Event Log 只包含控制流 metadata；未发现 prompt、reasoning、API key、模型响应正文、异常正文、工具参数或工具结果正文字段。

### Phase 11 结论

本阶段能够回答：Reliable profile 可以恢复三种指定且确定发生的故障，并保持两项安全不变量；代价是在本次 30-run 样本中平均增加 1.3 次模型调用和约 4.9s wall-clock。Real Coding 两组均为 15/15，因此当前样本不能回答“是否降低真实任务 false success”，该问题需要更难的任务、更多重复或新的故障分布，但不继续扩大 Phase 11 v1 范围。

## 明确不做

- 通用 Eval SDK、插件系统或数据库；
- SWE-bench 等外部大型 benchmark；
- LLM-as-judge；
- 并行或分布式评测；
- token、价格或供应商账单统计；
- 自动调参、显著性检验或模型排名；
- OS sandbox 或恶意代码执行环境；
- 把 5 个任务的结果泛化为通用 Coding 能力结论；
- 为评测修改 Phase 1–10 Runtime 语义。

## 参考来源

Phase 11 是 TinyHarness 自己的可靠性评测扩展，没有移植 `learn-claude-code` 或 Claude Code 产品的 Eval Runtime。它直接复用现有 Agent Loop、Provider、Event Log、Permission、Context Recovery 和 Goal Gate 公共接口；生产 Runtime 没有为 Eval 增加特殊分支。
