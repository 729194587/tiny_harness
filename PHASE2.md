# Phase 2 Baseline：Minimal Permission Gate

## 状态

- 状态：Accepted
- 验收日期：2026-08-17
- 设计：完成
- 离线实现：完成
- 自动测试：通过
- 真实 API ALLOW smoke test：通过
- 真实 API DENY smoke test：通过
- 默认策略：文件工具 ALLOW，`bash` ASK

Phase 2 已满足离线和真实 API 验收条件，可以冻结为 baseline 并创建 Git tag。

## 唯一目标

在 Phase 1 工具调用链路中加入最小权限门：

```text
ToolCall
→ 工具查找
→ JSON 参数解析
→ Permission Gate
   ├─ ALLOW → 执行 handler
   ├─ DENY  → 不执行 handler
   └─ ASK   → 由 CLI 用户决定
→ ToolResult
→ Model
```

Permission 的职责仅是决定 handler 是否可以执行。拒绝不是 Agent Loop 异常，而是一个普通 tool result。

## 默认策略

| 工具 | 决定 | 理由 |
|---|---|---|
| `read_file` | ALLOW | 只读且受 workspace 路径边界保护 |
| `list_files` | ALLOW | 只读且受 workspace 路径边界保护 |
| `write_file` | ALLOW | 用户已经显式指定 workspace |
| `edit_file` | ALLOW | 用户已经显式指定 workspace |
| `bash` | ASK | shell 可能访问或修改 workspace 外部资源 |
| 未识别工具 | DENY | 默认关闭；registry 会更早返回 Unknown tool |

Phase 2 不尝试判断 bash 命令是否只读、危险或可并发。每一个 `bash` tool call 都独立 ASK。

## 最小设计

实现位于 `tiny_harness/runtime/permissions.py`。

### `PermissionDecision`

三个值：

- `ALLOW`
- `DENY`
- `ASK`

### `PermissionPolicy`

最小同步 Protocol：

```python
decide(tool_name, arguments) -> PermissionDecision
```

Policy 接收已经成功解析、且顶层为 JSON object 的参数。

### `DefaultPermissionPolicy`

实现已确认的默认策略，不包含：

- bash 命令黑名单
- allowlist / denylist
- 路径规则语言
- 用户或角色
- 持久化选择

### `PermissionPrompt`

一个同步 callable：

```python
prompt(tool_name, arguments) -> bool
```

CLI 是当前唯一的 prompt 实现。返回 `True` 表示允许，`False` 表示拒绝。

### `resolve_permission()`

- ALLOW 和 DENY 不调用 prompt。
- ASK 且存在 prompt 时，由 prompt 的布尔结果解析。
- ASK 但没有 prompt 时 DENY。
- Policy 返回未知值时 DENY。
- Policy 或 prompt 抛出普通异常时 DENY。

该函数采用 fail-closed 行为。`KeyboardInterrupt` 不作为普通权限错误吞掉，仍允许用户中止整个进程。

## 执行顺序

`tools/registry.py` 中的 dispatcher 顺序为：

1. 查找工具。
2. 解析 `arguments_json`。
3. 确认参数顶层是 JSON object。
4. 解析权限决定。
5. DENY 时直接生成拒绝 `ToolResult`。
6. ALLOW 时执行 handler。
7. 将普通异常转换为错误 `ToolResult`。

拒绝结果格式：

```text
Error: Permission denied for tool <tool_name>
```

该结果保留原始 `tool_call_id`。

文件路径的 workspace 校验仍由文件 handler 在实际 I/O 前完成。Phase 2 没有引入独立的工具语义验证管线。

## Agent Loop 行为

Agent Loop 接收可注入的：

- `permission_policy`
- `permission_prompt`

同一次模型响应中的多个 tool calls 保持原始顺序，并分别执行权限判断。例如：

```text
write_file → ALLOW → 执行
bash       → ASK → 用户拒绝 → 不执行
read_file  → ALLOW → 执行
```

一个调用被拒绝不会丢弃其他调用，也不会立即终止 Agent Loop。全部 tool results 按原调用顺序回填模型。

## CLI ASK

CLI 显示：

```text
Permission required:
Tool: bash
Arguments:
{
  "command": "python -m unittest discover -s tests -v"
}
Allow this tool call? [y/N]:
```

输入规则：

- `y` / `yes`：ALLOW
- 空输入：DENY
- 其他输入：DENY
- EOF：DENY
- 不记住以前的选择

## 自动测试

命令：

```powershell
cd D:\learn-claude-code\tinyharness
python -m unittest discover -s tests -v
```

当前结果：

- 执行 50 项测试
- 49 项通过
- 1 项跳过
- 跳过项仍是 Windows 环境无法创建符号链接

权限测试覆盖：

- 默认文件工具 ALLOW
- 默认 bash ASK
- 未知工具默认 DENY
- 显式 ALLOW 不调用 prompt
- 显式 DENY 不调用 handler
- ASK 同意后执行
- ASK 拒绝后不执行
- 无 prompt 时拒绝
- EOF / policy / prompt 异常时 fail-closed
- 拒绝结果保留 tool call ID
- 同批多个工具分别判断并保持结果顺序
- 拒绝结果可以回填模型，Agent Loop 继续

## 明确不做

Phase 2 不实现：

- 危险命令黑名单
- bash 语法或副作用分析
- 权限规则文件
- 命令、路径或参数 pattern 规则
- “本次会话始终允许”
- 权限选择持久化
- 非交互自动批准模式
- 用户身份、角色或多租户权限
- PreToolUse / PostToolUse Hooks
- Event Log
- Context Management
- 并发或流式工具执行
- Session Resume
- ProviderFactory

## 真实 API 验收记录

验收日期：2026-08-17。

### 允许路径

Prompt：

```text
必须调用 bash 工具执行命令 cd，并报告输出。不要使用其他工具。
```

观察结果：

- CLI 显示 `Tool: bash` 和 `{"command": "cd"}`。
- 用户输入 `y`。
- 命令执行成功。
- 工具输出为 `D:\learn-claude-code\tinyharness`。
- 模型收到结果并返回最终自然语言回答。

结论：PASS。

### 拒绝路径

Prompt：

```text
必须调用 bash 工具执行命令 cd；如果工具返回 permission denied，不要重试，直接说明权限被拒绝。
```

观察结果：

- CLI 显示 `Tool: bash` 和 `{"command": "cd"}`。
- 用户输入 `n`。
- bash handler 未执行。
- 模型收到 permission denied tool result。
- 模型没有重试 bash，并返回权限被拒绝的最终回答。

结论：PASS。

两条路径均只使用无副作用的 `cd` 命令，没有删除、覆盖或泄露数据。

## 冻结规则

- Phase 2 后续不覆盖本文档中的历史测试结果。
- 新阶段改变权限语义时，应在新阶段文档记录差异。
- 自动测试继续保护 ALLOW、DENY、ASK 和 fail-closed 不变量。
- 建议在本次提交后创建 `phase-2-baseline` Git tag。

## 参考来源

实现前只查看了 `learn-claude-code-main/s03_permission/code.py` 中与权限控制流直接相关的内容。

TinyHarness 没有复制 s03 的危险命令 deny list、规则匹配或全局 WORKDIR 实现，只借鉴了“工具执行前检查权限、拒绝也回填工具结果”的控制流。没有查看或借鉴 Claude Code 产品源码。
