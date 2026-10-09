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

## Evidence and provenance

| Evidence | Available in Git | Meaning |
| --- | --- | --- |
| [Candidate pool](dev.jsonl), [catalog](task_catalog.csv), [source data](raw/dev.parquet) | Yes | Inputs for selecting and calibrating tasks; not model results. |
| [Default smoke selection](selected_tasks.jsonl) | Yes | Four fixed tasks, selected by [this script](select_swe_tasks.py). Membership alone does not prove calibration passed. |
| Historical calibrated 15-task development set | No complete selection or calibration bundle | Its exact membership cannot be recovered from the smoke set or inferred from candidates. |
| Historical model results and official verdicts | No published run bundle | No verifiable aggregate score or cache improvement is claimed here. |
| [Pipeline](../../tests/test_swe_bench_pipeline.py), [report](../../tests/test_swe_bench_report.py), and [finalizer](../../tests/test_swe_bench_finalizer.py) tests | Yes | Offline regression evidence using controlled fixtures, not real-model benchmark results. |

The default smoke IDs are `marshmallow-code__marshmallow-1343`,
`pylint-dev__astroid-1196`, `pydicom__pydicom-1139`, and
`sqlfluff__sqlfluff-1517`. Keep this selection unchanged when creating a separate
development set; pass the new JSONL file with `--selected`.

Local, Git-ignored `evals/results/` files may exist in an author's checkout.
They are not public evidence. The checkout reviewed for this documentation had
a single-task Marshmallow rollout and evaluator artifacts, but no complete
15-task bundle; the old rollout metadata also lacked source-commit provenance.
These files cannot establish a reproducible historical aggregate result.

To restore historical evidence, supply the exact selection JSONL (including
base commits and evaluator fields), calibration records, run/task metadata,
predictions, events, and matching official reports/logs. Retain the original
source revision and configuration if known; mark missing fields unknown rather
than assigning today's values. If those artifacts cannot be recovered, calibrate
an explicitly saved new selection and label subsequent runs as new experiments,
not a reconstruction of the historical 15-task set.

### Reproduce and retain a run

Use a clean checkout at a recorded commit. Save the exact selection file outside
the source checkout before running, record its SHA-256, and pass that same file
to rollout and grading. A path alone is not an immutable record of its contents.
For example, in Bash, after installing the evaluation extra and setting provider
variables as in the root README:

```bash
# This section runs real evaluations: Docker/images and model API usage apply.
mkdir -p ../tinyharness-evidence
cp evals/swe_bench_lite/selected_tasks.jsonl ../tinyharness-evidence/selection.jsonl
python -c "import hashlib,pathlib; p=pathlib.Path('../tinyharness-evidence/selection.jsonl'); print(hashlib.sha256(p.read_bytes()).hexdigest())" > ../tinyharness-evidence/selection.sha256
git rev-parse HEAD > ../tinyharness-evidence/source-commit.txt
python -m pip freeze > ../tinyharness-evidence/packages.txt
python -m evals.swe_bench_lite run \
  --selected ../tinyharness-evidence/selection.jsonl \
  --results-root ../tinyharness-evidence/results --run-id smoke-example \
  --max-turns 20 --subagent-max-turns 10 --max-context-tokens 125000
python -m evals.swe_bench_lite report ../tinyharness-evidence/results/smoke-example > ../tinyharness-evidence/metrics.json
python -m evals.swe_bench_lite evaluate \
  ../tinyharness-evidence/results/smoke-example/predictions.jsonl \
  --dataset ../tinyharness-evidence/selection.jsonl \
  --results-root ../tinyharness-evidence/results/smoke-example
```

`run` calibrates each selected task before rollout and skips calibration failures.
Preserve those failures and rollout errors in the denominator when describing
the selected workload; predictions contain only completed rollouts. A completed
rollout is not a resolved task. Check official per-instance reports and aggregate
verdicts; the evaluator process exiting successfully alone does not prove a fix.
For a single-task run, add `--instance-id` to `run` and use
`finalize <run-directory> --dataset <saved-selection.jsonl>` instead of `evaluate`
to produce `summary.json` and `summary.md`. `finalize` rejects multi-task runs.

Keep these artifacts together when publishing a result:

| Question | Artifact and limitations |
| --- | --- |
| Which tasks and source? | Saved selection + checksum; root/task `metadata.json` contains `tinyharness_git_commit` and `tinyharness_git_dirty`. A dirty flag does not capture edits: retain the diff and untracked source, or rerun from a clean commit. |
| Which model/configuration? | Metadata records model, turn limits, context budget, memory and workspace observation settings. Separately record provider endpoint (without credentials), date, exact command, Python/dependency versions, and Docker/image versions or digests; these are not all captured automatically. |
| What did the agent produce? | `predictions.jsonl`, task `model.patch`, `final_answer.txt`, and `events.jsonl`; context transcripts/summaries where emitted. |
| Did the patch resolve the task? | `official_evaluation/<id>/metadata.json`, `harness.log`, and official JSON reports/test logs tied to those predictions. Single-run finalization also records a predictions checksum. |
| What resources were observed? | `report` output and source events: token/cache usage with coverage, context estimates and compaction transitions, attribution, logical turns, and called/started tools. Missing usage remains unknown, not zero. |

Compare identical task selections and document configuration differences with
`compare`; a curated Dev evaluation graded by the official harness remains a
curated Dev evaluation, not an official full-benchmark score. Do not infer cost
savings from cache rate alone or compare totals from different task populations.

### Publishing artifacts

`evals/results/` is intentionally ignored. Publish a reviewed evidence bundle as
a release attachment or in a dedicated evidence directory and link it from the
root README. Generated metadata, summaries, logs, and transcripts can contain
absolute machine paths, tool output, or provider details. Remove credentials and
private data from a publication copy; document redactions and preserve artifact
associations. Prefer portable relative paths in that copy and pass `--dataset`
after relocation. Editing or moving old artifacts may prevent finalizer reuse
and require regrading; never relabel an unverified result as a successful reuse.

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
