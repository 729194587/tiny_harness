# TinyHarness

[English](README.md) | [简体中文](README.zh-CN.md)

TinyHarness 是一个基于 Chat Completions API 的轻量级同步 Coding Agent Runtime，支持工具执行与权限控制、Context 管理、Skills、可选的持久化 Memory 和 Subagents。

项目通过编码任务和 SWE-bench Lite Dev 检验 Runtime 的设计。终端输出与 JSONL 日志共享同一套事件记录，便于查看 Agent 的执行过程。架构细节见 [Runtime 说明](docs/runtime.zh-CN.md)。

## 设计与取舍

- **Context 与 Prefix Cache。** TinyHarness 以追加为主维护消息历史，仅在上下文达到压力阈值时进行压缩，以减少对 Prefix Cache 复用的影响。参见 [Cache 策略调整](https://github.com/729194587/tiny_harness/commit/df1fd53dfd5655ff2a94e91f37570c7b630d62f1)。
- **摘要可靠性。** Context 压缩可能丢失信息，也可能使未经验证的假设影响后续执行。摘要以事实为主，并保留不确定性。详见 [Context 与恢复](docs/runtime.zh-CN.md#context-与恢复)。
- **评测与运行记录。** SWE-bench Lite Dev 在 Agent 执行前校验基线和参考补丁的测试结果。事件日志记录执行轮次、Cache 使用、Context 变化和工作区修改，便于分析运行轨迹。详见 [评测流程](evals/swe_bench_lite/README.md)。
- **过度探索。** Agent 有时会反复搜索或读取文件，消耗轮次却没有生成补丁。这仍是一个待解决的问题，参见 [运行轨迹分析](evals/swe_bench_lite/README.md#inspect-existing-runs)。

## 快速开始

需要 Python 3.10+ 和模型服务的 API Key。在仓库根目录运行（PowerShell）：

```powershell
python -m pip install -e .
Copy-Item .env.example .env
# 编辑 .env，填写 API Key，并按需调整模型和 Base URL
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

Bash：

```bash
python -m pip install -e .
cp .env.example .env
# 编辑 .env，填写 API Key，并按需调整模型和 Base URL
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

省略任务参数即可进入交互会话，其他选项见 `--help`。

每次运行（包括交互会话中的每次提交）默认在启动目录的 `.tinyharness/events/` 下记录独立 JSONL 日志，
文件名包含 UTC 时间戳和唯一标识。使用 `--no-event-log` 关闭文件日志，或用 `--event-log PATH`
追加写入自定义文件；两者不能同时使用。

## 评测

TinyHarness 使用 SWE-bench Lite Dev 评估 Agent 行为，流程包括环境校准、Agent 执行和官方补丁评分。

仓库提供候选任务池和四任务 Smoke 测试集，用于开发和评测流程验证，不代表公开基准成绩。任务选择、运行指标、复现方式及证据限制见 [评测文档](evals/swe_bench_lite/README.md)。

## 进一步阅读

- [Agent Loop](tiny_harness/agent/loop.py)：运行循环的实现入口。
- [Runtime 说明](docs/runtime.zh-CN.md)：执行边界、工具、Context、Skills、Memory 和 CLI 配置。
- [开发约定与测试命令](AGENTS.md)：贡献规则和针对性验证。
