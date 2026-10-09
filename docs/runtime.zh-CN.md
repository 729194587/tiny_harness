# Runtime 说明

[English](runtime.md) | [简体中文](runtime.zh-CN.md)

本文说明当前 Runtime 的机制。设计动机和实验请从[项目主页](../README.zh-CN.md)开始阅读。

## 执行与故障

CLI 通过 `AgentSession` 提交任务；库调用方可通过 `run_agent()` 组装同一个
[核心循环](../tiny_harness/agent/loop.py)。每轮先准备规范历史，再投影出模型请求，
验证响应，然后执行一批工具调用或返回最终回答。

提交 assistant 消息或执行工具前，会先验证整个响应。调用 ID 必须非空且在批次内唯一，
传输字段和结束原因也必须有效。无效批次会抛出 `ModelProtocolError`；JSON 解码错误和
工具参数错误则作为单次工具结果返回。调用按模型给出的顺序逐个执行，所有 `TOOL_CALLED`
事件均在分发开始前发出。

共享分发流程如下：

```text
lookup / decode → pre hook → permission → started event
                → handler → post hook → result event → return
```

前置 Hook 阻止调用或权限拒绝时，不会执行处理函数。普通工具错误转为反馈；Hook 执行失败和
事件日志失败属于致命错误。权限决策为 `ALLOW`、`DENY` 或 `ASK`；审批请求未获回应时拒绝执行。
Shell 的工作目录不是操作系统沙箱。

非空提交开始后，任何异常或中断都会使 Session 保持失败状态，直到调用 `clear()`。
致命错误可能留下不完整的工具批次，以及真实的文件系统或进程副作用。Runtime 不会补造缺失结果、
重放调用或回滚副作用。`clear()` 只清空历史并重置失败状态；使用前应先检查工作区。
空输入校验不会使 Session 失效。直接调用循环的调用方必须丢弃或重建失败的运行。
参见 [Session](../tiny_harness/agent/session.py) 和
[批次生命周期测试](../tests/test_batch_lifecycle.py)。

Provider 瞬时错误的重试仍属于同一逻辑轮次。最后一个允许的轮次用于收尾：不提供工具，
要求模型给出当前能提供的最佳回答，不再执行工具调用。无效最终回答的处理见
[模型轮次实现](../tiny_harness/agent/turn.py)。

## 工具与运行能力

`tiny_harness/tools/` 下的模块导出 `build_tools(context)`。发现过程按模块名排序扫描；
各工厂返回可用的 `ToolDefinition` 对象，包含名称、描述、参数 Schema、处理函数及可选的
安全追踪元数据。缺少能力（如 Subagent Runner、Skill Catalog 或 Test Runner）时，工厂可以不返回工具。

[Registry](../tiny_harness/tools/registry.py) 负责查找、重复检查和 Schema 投影，
不存在第二份内置 Schema 列表。所有执行均经过共享分发。这是基于目录的发现机制，
不支持热重载或入口点插件。新增工具名称需要明确审查权限策略。
[工具架构方案](superpowers/plans/2026-09-04-tool-architecture-phase-1.md)记录了最初的设计；
以当前代码为准。

`task` 同步运行子 Agent，使用全新的消息，共享工作区和 Provider。子 Agent 继承策略与能力，
但拥有独立的轮次、恢复、Context 和 Todo 状态，不能再创建子 Agent，且跳过 Memory 提取。
父 Agent 将子 Agent 的最终文本作为工具输出接收。`todo_write` 更新本次运行的任务状态，
与持久化 Memory 分离。

Hooks 执行扩展逻辑；事件承载观测信息。
[Hooks 示例](../examples/hooks_demo.py)在前置 Hook 中阻止 `write_file`，在后置 Hook 中观测结果：

```powershell
python examples/hooks_demo.py "Inspect this project" --workspace .
```

库调用方默认使用空事件接收器。CLI 进度写入 stderr，单次运行的最终文本写入 stdout；
JSONL 日志可与终端输出同时启用。事件涵盖模型请求与响应、工具生命周期与耗时、Context 变化及
父子调用关联，不会将工具结果正文写入进度输出。

## Context 与恢复

规范历史是 Runtime 管理的状态，与模型请求不同。请求投影会复制历史并限制过大的 `read_file`
结果，即使关闭压缩也会如此；Runtime 指导信息也只存在于请求中。

硬预算包含消息和工具 Schema。超过配置预算的 80% 时，准备阶段会将工具结果持久化，
以降至 55% 为目标；只有缩减不足时才生成历史摘要。受保护的历史可能要求退回到 80% 的上限。
完整的调用与结果块以及当前请求会保留。不设主动工作检查点。

瞬时错误重试复用已准备的请求。Context 长度错误恢复在每个逻辑请求中最多压缩一次，
目标不超过失败请求的 75%，同时遵守硬预算。自动摘要和响应错误的摘要共用只记录事实的约定，
保留不确定性，不作就绪程度或后续行动的判断。参见
[Context 实现](../tiny_harness/runtime/context.py)和
[Context 不变量](../tests/test_context_invariants.py)。

默认启发式方法根据紧凑 JSON 估算 Token，约每四个字符一个 Token。Python 调用方可以注入
`token_meter`；归因估算与 Provider 计费用量不同。测量限制见
[Context 归因说明](context-attribution.md)。

## Skills 与 Memory

Skills 以 `<name>/SKILL.md` 的形式发现，YAML 中包含 `name` 和 `description`，来源为：

- 内置目录 `tiny_harness/skills/`；
- 用户目录 `~/.tinyharness/skills/`；
- 工作区目录 `.tinyharness/skills/`。

Session 持有目录快照。模型最初只接收有大小限制的元数据，再通过 `load_skill` 获取正文。
正文属于不可信指导，不能覆盖指令、权限、Hooks 或工作区边界。重新发现变更需要启动新的 Session。
背景见 [Skills 设计](superpowers/plans/2026-09-04-formal-skills-phase-2.md)。

持久化 Memory 通过 `--memory` 显式启用，文件位于工作区的 `.tinyharness/memory/`。
运行准备阶段使用运行级标记选择并加载相关条目。根 Agent 的最终回答 Hooks 提取长期记忆，
并可选择整合。普通提取或整合失败不会丢弃回答；`EventLogError` 仍是致命错误。
Memory 存储稳定偏好、反馈、项目事实和引用，不保存当前计划或执行状态。参见
[Memory 实现](../tiny_harness/memory/)。

## CLI 与本地验证

必须设置 `TINYHARNESS_API_KEY`。`TINYHARNESS_MODEL` 和 `TINYHARNESS_BASE_URL`
可覆盖主页示例所示的默认值。省略任务参数会进入复用同一 Session 的 REPL。

运行 `python -m tiny_harness --help` 获取权威选项列表：

| 选项 | 用途 / 默认值 |
| --- | --- |
| `--workspace` | 工作区目录；当前目录 |
| `--max-turns` | 主 Agent 轮次上限；20 |
| `--max-model-retries` | 每个逻辑请求的瞬时错误重试次数；2 |
| `--max-context-tokens` | 估算 Context 预算；125000 |
| `--no-context-compaction` | 关闭预算和压缩，保留请求投影 |
| `--subagent-max-turns` | 子 Agent 轮次上限；10 |
| `--event-log` | 将生命周期事件追加到 JSONL 文件 |
| `--memory` | 启用工作区持久化 Memory |
| `--quiet` / `--verbose` | 减少进度输出 / 增加 Runtime 细节 |

确定性测试使用预设响应的 Provider 和本地测试夹具，不调用真实 API：

```powershell
python -m unittest discover -s tests -v
```

针对性测试命令和扩展不变量见 [AGENTS.md](../AGENTS.md)。
