# TinyHarness

[English](README.md) | [简体中文](README.zh-CN.md)

TinyHarness 是一个面向 Chat Completions 服务的小型同步编码 Agent Runtime。
它力求保持精简，让开发者能够理解其实现、观测运行过程并进行修改。

这个项目通过真实编码任务和 SWE-bench 检验 Runtime 设计：引入机制、观察行为、
对比运行结果，再据此调整或移除机制。重点在于设计如何随着这些检验不断演进。

目前已实现顺序工具调用、权限检查与 Hooks、Context Compaction 与恢复、Skills、
可选的持久化 Memory、Todos，以及同步 Subagents。统一的事件流支持终端进度展示
和 JSONL 分析。架构说明见[运行边界](docs/runtime.md)。

[离线回归测试](tests)通过预设响应的 Provider、Mock 和临时工作区验证这些机制。
[CI](.github/workflows/ci.yml) 在 Python 3.10 和 3.12 下运行测试及 CLI 帮助检查，
无需 API Key 或 Docker。这些测试验证 Runtime 行为，不衡量外部模型的编码能力。
下方的[评测证据](#评测)区分了已纳入仓库的任务输入与尚未公开的 benchmark 结果。

## 推动 Runtime 演进的问题

- **上下文更少，成本就更低吗？** 重写历史可能损失 Prefix Cache 的复用。
  因此，设计逐步转向以追加为主的历史记录，仅在实际上下文压力下进行压缩，
  最终移除了主动检查点机制。相关变更见
  [Cache 相关调整](https://github.com/729194587/tiny_harness/commit/df1fd53dfd5655ff2a94e91f37570c7b630d62f1)
  和[移除检查点](https://github.com/729194587/tiny_harness/commit/9a7635b5ebba3a921ce74f51523f18fad6075ef5)。
- **摘要除了减少 Token，还改变了什么？** 摘要会成为 Agent 后续工作的证据；
  如果将假设写成事实，就可能影响后续决策。摘要约定先强调保留不确定性，
  随后收敛为仅记录事实。压力压缩和恢复流程中的摘要都遵循这一
  [约定](tiny_harness/runtime/context.py)，但不能仅凭摘要就解释某个任务为何失败。
- **测量结果可信吗？** 在模型执行任务前，评测流程先通过基线和参考补丁校验环境。
  事件报告将轮次、Cache 使用、上下文变化和工作区修改关联起来，帮助分析分数背后的
  失败原因。可从[评测流程](evals/swe_bench_lite/README.md)开始了解。
- **什么时候证据足以支持行动？** 反复读取和搜索可能耗尽轮次预算，却没有产生补丁。
  过度探索仍是一个待解决的问题，移除主动压缩并不能解决它。
  [请求归因说明](docs/context-attribution.md)介绍了目前能测量什么，以及这些测量
  无法证明什么。

## 快速开始

在仓库根目录运行，需要 Python 3.10+ 和模型服务的 API Key（PowerShell）：

```powershell
python -m pip install -e .
$env:TINYHARNESS_API_KEY = "..."
$env:TINYHARNESS_MODEL = "deepseek-v4-flash"
$env:TINYHARNESS_BASE_URL = "https://api.deepseek.com"
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

也可以使用 Bash：

```bash
python -m pip install -e .
export TINYHARNESS_API_KEY="..."
export TINYHARNESS_MODEL="deepseek-v4-flash"
export TINYHARNESS_BASE_URL="https://api.deepseek.com"
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

请按所用的 Chat Completions 服务设置模型名称和 Base URL。省略任务参数即可进入
交互会话；使用 `--help` 查看选项。

安装完成后，可运行离线检查（PowerShell 或 Bash）：

```bash
python -m pip install pytest
python -m pytest -q
python -m tiny_harness --help
```

## 评测

SWE-bench Lite Dev 管线先校验基线和参考补丁的测试表现，再让 Agent 执行通过校验的
任务，最后使用官方评测器为生成的补丁评分。
[候选任务池](evals/swe_bench_lite/dev.jsonl)和
[四任务冒烟测试集](evals/swe_bench_lite/selected_tasks.jsonl)均已纳入仓库。
冒烟测试集用于开发验证，其结果不代表完整 benchmark 的分数。

仓库尚未收录完整的历史 15 任务选择文件或已评分的实验产物，因此本 README 不宣称
任何解决率、Cache 改善或排名。
[证据清单与复现指南](evals/swe_bench_lite/README.md#evidence-and-provenance)
说明了现有材料，以及如何一并保存任务选择、源码版本、配置、官方评分结论和
Token、Cache、Context、Tool 指标。现有 `report` 和 `compare` 命令可离线分析
已保存的事件；`finalize` 则为单任务运行补充官方评分与汇总。

## 进一步阅读

[Agent Loop](tiny_harness/agent/loop.py) 是阅读实现的入口。
[Runtime 说明](docs/runtime.md)涵盖执行边界、工具、上下文、Skills、Memory 和 CLI 配置。
[TinyHarness 开发指南](AGENTS.md)介绍贡献规则和针对性测试。
