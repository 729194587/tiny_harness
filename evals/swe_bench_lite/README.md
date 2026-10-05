# SWE-bench Lite evaluation

The evaluation pipeline separates environment calibration, model rollout, and
patch grading. Calibration runs the baseline and reference patch through the
official evaluator: expected failing tests must fail on baseline and pass with
the reference patch, while regression tests must pass in both. See
[calibration.py](calibration.py) and its
[tests](../../tests/test_swe_batch_calibration.py).

From the repository root, install the evaluation extra and make Docker available:

```powershell
python -m pip install -e ".[swe-bench]"
python -m evals.swe_bench_lite calibrate --all-selected
python -m evals.swe_bench_lite run --help
```

`selected_tasks.jsonl` supplies the default smoke selection; `dev.jsonl` supplies
the candidate pool for `calibrate --all-candidates`. These checked-in selections
are not a published benchmark score or a record of every historical experiment.
Model rollout requires the provider configuration described in the
[root README](../../README.md).

## Inspect existing runs

These commands read existing event artifacts without Docker or a model API call:

```powershell
python -m evals.swe_bench_lite report <run-directory>
python -m evals.swe_bench_lite compare <run-a> <run-b>
```

The JSON reports cover logical turns, retries, token/cache usage, tool batching,
context compaction, request attribution, and workspace mutation observations.
Comparison is B minus A; unknown values remain null. Aggregate cache hit rate is
summed hits divided by summed hits plus misses, not an average of request rates.
It should not be compared across systems without matching workloads and usage
accounting. See [report.py](report.py) and
[request attribution](../../docs/context-attribution.md).

Workspace observation hashes content at tool boundaries, excluding `.git` and
`.tinyharness`. It can detect shell writes, but cannot see changes restored within
one tool or reliably attribute background writes. Missing observations leave the
first mutation unknown; a tool's return alone proves neither a write nor a passing
test. See [observability.py](observability.py).

Long stretches of reads before the first mutation are useful evidence when
investigating over-exploration. Counts alone do not establish that the agent had
enough evidence, or that a summary caused a failure; that requires inspecting the
trajectory and controlled comparisons.
## Single-run finalize

```bash
python -m evals.swe_bench_lite finalize <run-directory>
python -m evals.swe_bench_lite finalize <run-directory> --dataset <dataset.jsonl>
```

输入为包含 `predictions.jsonl` 和 rollout `events.jsonl` 的单任务 run
目录。数据集默认取 run metadata 的 `selected_tasks_path`，旧 run 未记录时
使用 bundled `selected_tasks.jsonl`；迁移目录后可用 `--dataset` 覆盖。
metadata 中的相对路径以 run 目录为基准。

命令复用现有 official evaluator，因此首次评测需要 SWE-bench harness 和
Docker 环境；不会调用 rollout 模型。它检查 run 内及上一级 results 目录的
`official_evaluation/*`，仅复用关联当前 predictions、退出成功且有明确 grader
结论的结果。新结果记录 predictions SHA-256；旧结果无摘要时要求 predictions
修改时间不晚于评测 metadata。修改或移动历史 artifacts 后可能需要重新评测。

输出 `summary.json`、`summary.md`，终端打印与 Markdown 相同的核心结果。
`Resolved` 和 `Unresolved` 表示有效评测结论，退出码均为 0；评测异常、非零
退出码、缺失或冲突的 grader 结论为 `Evaluation Error`，退出码 2。
输入不合法（例如多个 instance）直接报错，不启动评测。

metrics 复用 offline report：逻辑 turns 去除重试重复，usage 包含辅助模型调用，
overall cache hit rate 为累计 hit / (累计 hit + 累计 miss)，不是逐请求比例的平均。
缺失字段用 JSON null / “未知” 表示，部分 usage 的观测总量和覆盖数保留在 JSON。
checkpoint 仅统计 `llm_task_state_checkpoint` 事件，普通历史结果清理不计入。
summary cache hit rate 来自 `purpose="summary"` 的模型响应。
workspace mutation 缺少完整观测时保持未知，不从文件工具返回值推断。
