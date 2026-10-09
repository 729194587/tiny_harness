# TinyHarness

[English](README.md) | [简体中文](README.zh-CN.md)

TinyHarness 是一个基于 Chat Completions API 的同步 Coding Agent Runtime，支持工具调用、权限控制、Hooks、Context 压缩与恢复、Skills、可选的持久化 Memory、Todos 和同步 Subagents。

项目使用编码任务和 SWE-bench Lite Dev 检验 Runtime 的设计。终端输出和 JSONL 日志基于同一套事件记录，方便查看 Agent 的执行过程。架构细节见 [Runtime 说明](docs/runtime.zh-CN.md)。

[离线回归测试](tests)使用预设响应的 Provider、Mock 和临时工作区。[GitHub Actions](.github/workflows/ci.yml) 在 Python 3.10 和 3.12 下运行测试及 CLI 检查。

## 设计与取舍

- **Context 与 Prefix Cache。** 重写历史消息会影响 Prefix Cache 复用，因此 TinyHarness 以追加为主维护消息历史，在上下文达到压力阈值时进行压缩。后续移除了主动检查点机制。相关提交见 [Cache 策略调整](https://github.com/729194587/tiny_harness/commit/df1fd53dfd5655ff2a94e91f37570c7b630d62f1)和[移除检查点](https://github.com/729194587/tiny_harness/commit/9a7635b5ebba3a921ce74f51523f18fad6075ef5)。
- **摘要可靠性。** Agent 会根据压缩后的摘要继续执行任务。为避免未经验证的假设被写成事实，摘要规则逐步调整为只记录事实；压力压缩和错误恢复共用这套[规则](docs/runtime.zh-CN.md#context-与恢复)。
- **评测与运行记录。** SWE-bench Lite Dev 会在 Agent 执行前校验基线和参考补丁的测试结果。事件日志记录执行轮次、Cache 使用、Context 变化和工作区修改，用于分析运行轨迹。详见[评测流程](evals/swe_bench_lite/README.md)。
- **过度探索。** Agent 有时会反复读取文件和搜索，消耗轮次却没有生成补丁。相关行为可以结合[运行轨迹与工作区观测说明](evals/swe_bench_lite/README.md#inspect-existing-runs)分析，这仍是一个待解决的问题。

## 快速开始

需要 Python 3.10+ 和模型服务的 API Key。在仓库根目录运行（PowerShell）：

```powershell
python -m pip install -e .
$env:TINYHARNESS_API_KEY = "..."
$env:TINYHARNESS_MODEL = "deepseek-v4-flash"
$env:TINYHARNESS_BASE_URL = "https://api.deepseek.com"
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

Bash：

```bash
python -m pip install -e .
export TINYHARNESS_API_KEY="..."
export TINYHARNESS_MODEL="deepseek-v4-flash"
export TINYHARNESS_BASE_URL="https://api.deepseek.com"
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

按所用的 Chat Completions 服务设置模型名称和 Base URL。省略任务参数即可进入交互会话，其他选项见 `--help`。

离线测试：

```bash
python -m pip install pytest
python -m pytest -q
python -m tiny_harness --help
```

## 评测

SWE-bench Lite Dev 评测流程先校验基线和参考补丁的测试结果，再让 Agent 执行任务，最后使用官方评测器为生成的补丁评分。

仓库提供候选任务池和四任务冒烟测试集，用于开发和验证评测流程。[证据清单与复现指南](evals/swe_bench_lite/README.md#evidence-and-provenance)提供原始数据入口，并列出实验所需的任务选择、源码版本、配置、官方评分结果及 Token、Cache、Context、Tool 指标。

`report` 和 `compare` 可离线分析已保存的运行事件；`finalize` 可为单任务运行生成官方评分与汇总。

## 进一步阅读

- [Agent Loop](tiny_harness/agent/loop.py)：运行循环的实现入口。
- [Runtime 说明](docs/runtime.zh-CN.md)：执行边界、工具、Context、Skills、Memory 和 CLI 配置。
- [开发约定与测试命令](AGENTS.md)：贡献规则和针对性验证。
