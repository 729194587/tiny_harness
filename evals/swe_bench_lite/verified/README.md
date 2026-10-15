# Verified P1：固定子集与无模型环境准备

这是 30 个任务的准备子集，**不是完整 SWE-bench Verified 基准成绩**。
不使用任务答案、难度、测试数量、模型表现或校准结果筛选任务。
本入口不导入 rollout、不构造模型 provider、不读取 API Key，也没有模型运行命令。
校准使用的 gold patch 仅进入评测侧，不能作为模型预测或模型成绩。

## 流程审查与复用

| 组件 | 结论 |
| --- | --- |
| 数据 | 官方新版 `SWE-bench/SWE-bench_Verified` 包含 `image`、`eval_script`、`log_parser`、`eval_type`，可直接使用 `SweTask` / `SweEvaluationBundle`。原始 Princeton 版本缺少这些环境字段，不混用。 |
| Calibration | 复用 baseline/gold 两个独立 Docker 环境和官方 `get_eval_report`；保存两份原始评分报告。新增入口区分环境预检失败、官方标记的基础设施失败、校准失败、可运行。 |
| Docker | 复用 `/testbed` 复制、bind mount、base commit reset、清理。数据中的镜像标签不等于不可变镜像，成功预检后记录 RepoDigests；此阶段不宣称已验证所有镜像。 |
| Rollout | 原有 `run --selected <jsonl>` 与数据隔离边界可复用，本次未执行、未修改。 |
| Official Evaluation | 复用现有官方 CLI 包装器（串行、`max_workers=1`），不复制评分逻辑。包装器的 `--split dev` 对本地 JSONL 无作用，5.0.2 的 loader 直接读取文件；真实来源是 manifest 中的 `test`。冒烟校准通过后额外运行 gold CLI，并要求明确的 `report.json` 成功结论。 |
| Report | 原有 report/compare 仍分析 rollout events；准备阶段没有模型事件，使用独立 `summary.json`，不伪造 token、成功率或成绩。 |

保持 `swebench==5.0.2`，未升级依赖，未改变 Agent Runtime、Prompt、Context 或 Tool 行为。
现有 Dev Smoke 文件和历史命令保持原样。旧 `calibrate --instance-id` 会选择 Dev pool，
因此 Verified 使用独立入口，避免改变已有命令的含义。

## 来源、抽样和复现

来源：[官方 Verified 数据](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/tree/78f471bf655a3137b2e8a75af1501690ec009ec3)，
固定 revision `78f471bf655a3137b2e8a75af1501690ec009ec3`，`test` split，共 500 个任务。
Parquet SHA-256：`030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25`。
完整任务记录保存在 `selected_tasks.jsonl`；所有 ID、选择文件校验和、来源校验和和排除清单见 `manifest.json`。

将现有 `dev.jsonl` 的全部 23 个候选任务和 Dev Smoke 的任务 ID 合并排除，
涵盖仓库中可追溯的开发任务；与此次 Verified 来源的交集为 0。
历史 15 任务完整清单未公开，不能验证未知的、来自其他来源的历史任务。
如有额外历史清单，可用 `--exclude <历史.jsonl>` 创建新的、有独立 manifest 的选择版本。

种子固定为 `20261014`。对排除后的每个 ID 计算
`SHA-256(UTF-8(str(seed) + "\n" + instance_id))`，按十六进制摘要升序排列，
ID 为同摘要的次排序键，取前 30。该方法不依赖源文件行顺序或 Python RNG 版本。
前三个固定为冒烟任务：`sympy__sympy-24539`、`django__django-11265`、`sphinx-doc__sphinx-10323`。
失败不补抽、不删除、不修改分母。

在仓库根目录运行，准备命令只读公开数据，不需要任何 Key：

```powershell
python -m pip install -e ".[swe-bench]"
New-Item -ItemType Directory -Force evals/results/verified-source
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/resolve/78f471bf655a3137b2e8a75af1501690ec009ec3/data/test-00000-of-00001.parquet' -OutFile evals/results/verified-source/test.parquet
python -m evals.swe_bench_lite.verified prepare evals/results/verified-source/test.parquet evals/results/verified-recreated
```

输出目录必须不存在，以免覆盖旧证据。准备过程拒绝来源 SHA-256 不匹配的文件。
重新生成的 `selected_tasks.jsonl` SHA-256 应等于已提交 manifest；JSON ASCII 转义
解决官方 5.0.2 Windows 本地 loader 的系统编码问题，解码后原文不变。
选择文件固定 LF；排除文件校验和按 CRLF 转 LF 后计算，避免 Git 换行配置造成差异。

## 独立校准

需要可用的 Linux Docker 引擎及公开镜像下载网络。默认仅尝试固定三个任务：

```powershell
python -m evals.swe_bench_lite.verified calibrate evals/results/verified-smoke-new
# 后续明确要校准全部 30 个任务时才运行；本次未执行：
python -m evals.swe_bench_lite.verified calibrate evals/results/verified-calibration-30 --all
```

两条命令均无模型调用路径。全量校准顺序执行 30 个任务的环境预检和 baseline/gold；
只有默认三任务冒烟模式在校准通过后额外执行官方 gold CLI。
环境预检调用 Docker 限时 30 秒；容器操作沿用 1200 秒超时；官方 CLI 限时 1800 秒。
这些是每次操作的超时，不能作为总运行时长估计。

`summary.json` 从开始即列出全部 30 个任务，并逐任务写回。状态为：

- `NOT_RUN`：未尝试，绝不计为通过或失败。
- `ENVIRONMENT_FAILED`：环境预检失败，或官方 grader 明确返回 `infra_failure`。
- `CALIBRATION_FAILED`：预检成功，但 baseline/gold 行为未复现或后续执行报错。
- `RUNNABLE`：baseline 的预期失败与回归通过、gold 的两类测试通过均得到官方评分支持。

`official_cli` 独立记录 `NOT_RUN` / `PASSED` / `FAILED`；进程退出 0 而没有明确成功报告不算通过。
Smoke 退出非零表示有失败，不能据此删除样本。`--all` 的 RUNNABLE 表示校准通过，
不表示有模型运行成绩。环境在预检后发生故障可能表现为校准失败，需结合日志诊断。
保留 `error.txt`、进程日志、baseline/gold 日志及报告、镜像摘要和官方 CLI 目录。
中断时未完成任务可能仍是 NOT_RUN，应结合是否存在任务目录识别未完成尝试。

## 实际验证与限制

2026-10-14，Windows / Python 3.11 / SWE-bench 5.0.2：

- 读取官方 500 行 Parquet，生成 30 行选择；两种 TinyHarness 数据视图可加载。
- 官方 loader 与 TestSpec 对全部 30 行的兼容性由自动化测试检查。
- 前三个任务调用真实官方 grader 读取空日志，验证不会把缺失测试输出判为 resolved。
  这是评分接口负向冒烟，不是真实容器测试或任务成绩。
- Docker 客户端 28.4.0 存在，但 Linux 引擎未正常响应；启动已安装 Docker Desktop
  后限时检查仍超时。三任务环境尝试均失败，未进行 baseline/gold 测试或官方 CLI 真实评分。
- 未调用模型 API，未使用真实 Key，未执行 30 任务全量校准。
- SWE 相关回归测试共 69 项，68 项通过、1 项因 Windows 符号链接权限跳过。
  最初沙箱执行无法创建临时 Git 仓库，使用正常权限重跑后通过。
- 从同一固定 Parquet 再次生成选择，manifest 和 JSONL 均逐字节一致。

见 `smoke-evidence.json` 的实际尝试记录。真实 Docker 执行及官方 CLI gold 评分仍待在
可用 Linux Docker 主机上复验；不得把本 PR 的离线测试当成这部分已通过。

```powershell
python -m unittest discover -s tests -p 'test_swe*.py' -v
```

## 30 个任务的资源预算

[官方建议](https://github.com/SWE-bench/SWE-bench/blob/main/README.md)：x86_64 主机，
至少 8 CPU、16 GB RAM、120 GB 可用磁盘。这里以该建议作为规划下限，不是本子集实测。
建议单 worker 起步，磁盘预留约 150–250 GB 作为镜像缓存和工作副本的工程余量；
实际空间取决于镜像共享层，未测量，120 GB 不保证足够。无需 GPU，无模型 API 费用。
网络会下载公开镜像；实际下载量和运行时间未知，先在上述三个固定任务上测量，
再估算 30 个任务。完整校准至少有 60 次测试阶段（30 baseline + 30 gold），
另有每任务环境预检与临时工作区；不要用离线测试时长推算真实环境时长。
