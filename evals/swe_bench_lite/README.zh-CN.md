# SWE-bench Lite 评测

[English](README.md) | [简体中文](README.zh-CN.md)

评测流程分为环境校准、模型运行和补丁评分。校准会用官方评测器分别测试基线和参考补丁：
预期失败的测试必须在基线上失败、应用参考补丁后通过；回归测试则必须在两者上都通过。
参见 [calibration.py](calibration.py) 和[测试](../../tests/test_swe_batch_calibration.py)。

运行评测可查看[复现与保存运行](#复现与保存运行)；分析已有轨迹可使用[离线报告](#分析已有运行)；
为单个任务评分并汇总结果可使用[单次运行收尾](#单次运行收尾)。
[证据表](#证据与来源)区分了已提交的输入数据和未公开的结果。

在仓库根目录安装评测扩展依赖，并确保 Docker 可用：

```powershell
python -m pip install -e ".[swe-bench]"
python -m evals.swe_bench_lite calibrate --all-selected
python -m evals.swe_bench_lite run --help
```

`selected_tasks.jsonl` 是默认冒烟测试任务集；`dev.jsonl` 是
`calibrate --all-candidates` 使用的候选池。提交这些任务集不代表发布了基准成绩，
也不代表记录了所有历史实验。模型运行需要按[根 README](../../README.zh-CN.md)配置模型服务。

## 证据与来源

| 证据 | Git 中是否提供 | 含义 |
| --- | --- | --- |
| [候选池](dev.jsonl)、[目录](task_catalog.csv)、[源数据](raw/dev.parquet) | 是 | 用于任务选择和校准的输入，不是模型结果。 |
| [默认冒烟测试集](selected_tasks.jsonl) | 是 | 由[此脚本](select_swe_tasks.py)选出的四个固定任务。被选中不代表已经通过校准。 |
| 历史上校准过的 15 任务开发集 | 否 | 存在历史本地运行记录，但完整的任务选择和校准记录包未提交到 Git。 |
| 历史模型结果和官方判定 | 没有公开运行记录包 | 已恢复的 `pressure-only-15` 摘要记录了 **14/15 Resolved** 及本轮 token/缓存指标；该结果不是官方完整基准成绩。 |
| [流程](../../tests/test_swe_bench_pipeline.py)、[报告](../../tests/test_swe_bench_report.py)和[收尾](../../tests/test_swe_bench_finalizer.py)测试 | 是 | 基于可控测试数据的离线回归证据，不是真实模型基准结果。 |

### 历史 `pressure-only-15` 运行

目前已恢复一份历史本地摘要，记录了一个 15 任务 Dev 评测 `pressure-only-15`。该摘要包含以下运行事实：

- 任务集：来自 `dev.jsonl` 的 15 个任务
- 结果：**14 / 15 Resolved（93.33%）**
- `max_turns`：`40`
- 源码提交：`9a7635b5ebba3a921ce74f51523f18fad6075ef5`
- 总轮次：`443`
- 工具调用：`575`
- Prompt tokens：`7,499,628`
- Completion tokens：`161,213`
- Cache hit tokens：`7,162,880`
- Cache miss tokens：`336,748`
- 加权缓存命中率：**95.51%**
- 峰值活跃上下文：`53,424` tokens
- Pressure compaction / reactive compaction：`0 / 0`

这些是历史本地实验结果。目前完整证据包尚未提交到 Git，因此应将其表述为**精选的 15 任务 SWE-bench Lite Dev 评测**，而不是官方完整基准成绩。

默认冒烟任务 ID 为 `marshmallow-code__marshmallow-1343`、`pylint-dev__astroid-1196`、
`pydicom__pydicom-1139` 和 `sqlfluff__sqlfluff-1517`。创建独立开发集时，应保留默认任务集不变，
通过 `--selected` 传入新的 JSONL 文件。

本地、被 Git 忽略的 `evals/results/` 文件不属于公开证据。

目前已在本地恢复历史 `pressure-only-15` 摘要。因此，不能仅因为结果尚未提交 Git，就将该历史实验表述为“没有记录”。如果要将该结果发布为可移植的证据包，应保留确切的任务清单 JSONL（含基线提交和评测器字段）、校准记录、运行及任务元数据、预测文件、事件，以及对应的官方报告和日志。应保留原始源码版本和配置；真正缺失的字段标记为未知，不能填入今天的值。在这些材料公开之前，应将其表述为历史本地评测，而不是公开可复现的基准结果。

## 复现与保存运行

从一个干净、提交版本已记录的工作区开始。运行前，把确切的任务清单保存到源码目录之外，
记录其 SHA-256，并将同一文件交给模型运行和评分步骤。只有路径，不能保证文件内容不变。
例如，安装评测扩展依赖并按根 README 配置模型服务变量后，在 Bash 中运行：

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

`run` 会在模型运行前校准每个选定任务，跳过校准失败的任务。描述整个选定工作量时，
分母应保留校准失败和模型运行出错的任务；预测文件只包含完成运行的任务。运行完成并不等于问题
已解决。应检查官方逐任务报告和汇总判定，仅评测器进程成功退出不能证明修复有效。
单任务运行可在 `run` 中加入 `--instance-id`，并用
`finalize <run-directory> --dataset <saved-selection.jsonl>` 替代 `evaluate`，
生成 `summary.json` 和 `summary.md`。`finalize` 会拒绝多任务运行。

发布结果时，应将以下文件一起保留：

| 要回答的问题 | 文件及其限制 |
| --- | --- |
| 使用了哪些任务和源码？ | 保存的任务清单及校验和；根目录和任务目录的 `metadata.json` 包含 `tinyharness_git_commit` 与 `tinyharness_git_dirty`。dirty 标记不保存修改内容：需另存 diff 和未跟踪源码，或从干净提交重新运行。 |
| 使用了哪个模型和哪些配置？ | 元数据记录模型、轮次上限、上下文预算、记忆和工作区观测设置。服务端点（不含凭据）、日期、完整命令、Python 与依赖版本、Docker 和镜像版本或摘要需另行记录，这些信息不会全部自动保存。 |
| Agent 产出了什么？ | `predictions.jsonl`，任务目录的 `model.patch`、`final_answer.txt` 和 `events.jsonl`，以及生成时保存的上下文记录和摘要。 |
| 补丁是否解决了任务？ | 与预测文件对应的 `official_evaluation/<id>/metadata.json`、`harness.log`、官方 JSON 报告和测试日志。单次运行收尾还会记录预测文件的校验和。 |
| 观测到了哪些资源使用情况？ | `report` 输出和原始事件：带覆盖情况的 token/缓存用量、上下文估算及压缩前后变化、归因、逻辑轮次、已请求及已开始执行的工具。缺失用量仍是未知，不是零。 |

使用 `compare` 比较相同任务集，并说明配置差异。使用官方评测器评分的精选 Dev 任务集，
仍然只是精选 Dev 评测，不是官方完整基准成绩。不能仅凭缓存命中率推断成本节省，
也不能直接比较不同任务群体的总量。

### 发布结果文件

`evals/results/` 有意被 Git 忽略。应将审阅过的证据包作为发布附件，或放入专门的证据目录，
再从根 README 链接过去。生成的元数据、摘要、日志和对话记录可能包含本机绝对路径、工具输出
或模型服务信息。发布副本中应移除凭据和私密数据，说明删改情况，并保持文件之间的对应关系。
副本中优先使用可移植的相对路径；移动后通过 `--dataset` 指定数据集。编辑或移动旧文件可能导致
收尾工具无法复用旧评测，需要重新评分；不能把未经验证的结果改称为成功复用。

## 分析已有运行

以下命令只读取已有事件文件，不需要 Docker，也不调用模型 API：

```powershell
python -m evals.swe_bench_lite report <run-directory>
python -m evals.swe_bench_lite compare <run-a> <run-b>
```

JSON 报告涵盖逻辑轮次、重试、token/缓存用量、工具批次、上下文压缩、请求归因和工作区修改观测。
比较值为 B 减 A，未知值保留为 null。汇总缓存命中率为命中量之和除以命中量与未命中量之和，
不是逐请求命中率的平均值。跨系统比较时，需要匹配工作量和用量统计口径。
参见 [report.py](report.py) 和[请求归因](../../docs/context-attribution.zh-CN.md)。

工作区观测会在工具执行前后，对已跟踪文件和未被忽略的未跟踪文件计算内容哈希，排除 `.git`
和 `.tinyharness`。符号链接按其目标路径计算，不跟随链接读取内容；权限和空目录不在观测范围内。
这种方式可以发现 Shell 写入，但看不到同一次工具执行中先修改再恢复的变化，也无法可靠地归因
后台写入。缺少观测时，首次修改时间仍是未知；工具返回本身不能证明发生了写入或测试通过。
参见 [observability.py](observability.py)。

首次修改之前的大量读取，是调查过度探索时的有用线索。但计数本身不能证明 Agent 当时已经获得
充分证据，也不能证明某次摘要导致了失败；这些判断需要检查运行轨迹并进行受控对比。

## 单次运行收尾

```bash
python -m evals.swe_bench_lite finalize <run-directory>
python -m evals.swe_bench_lite finalize <run-directory> --dataset <dataset.jsonl>
```

输入是单任务运行目录，其中包含 `predictions.jsonl` 和模型运行的 `events.jsonl`。
数据集默认取自运行元数据中的 `selected_tasks_path`；旧运行没有该字段时，回退到内置的
`selected_tasks.jsonl`。目录移动后应使用 `--dataset`。元数据中的相对路径以运行目录为基准解析。

该命令调用现有官方评测器，因此首次评分需要 SWE-bench 评测框架和 Docker；不会调用执行任务的模型。
它会检查运行目录及其父级结果目录中的 `official_evaluation/*`，只复用与当前预测文件对应、
进程成功退出且具有明确评分判定的结果。新结果记录预测文件的 SHA-256；没有校验和的旧结果，
要求预测文件的修改时间不晚于评测元数据。编辑或移动历史文件可能需要重新评分。

命令写入 `summary.json` 和 `summary.md`，并打印与 Markdown 摘要相同的核心结果。
`Resolved` 和 `Unresolved` 都是有效判定，退出码均为 0。评测异常、评测器非零退出，
以及评分判定缺失或冲突，会产生 `Evaluation Error`，退出码为 2。
多实例等无效输入会在启动评测前被拒绝。

指标复用[离线报告](#分析已有运行)：逻辑轮次不重复计算重试，用量包含辅助模型调用。
缺失字段使用 JSON null 或显示为 `未知`；用量只覆盖部分调用时，JSON 仍保留已观测总量和覆盖计数。
检查点只统计 `llm_task_state_checkpoint` 事件，不包括普通历史结果裁剪。
摘要缓存命中率取自 `purpose="summary"` 的模型响应。上文说明的工作区观测限制同样适用于这些摘要。
