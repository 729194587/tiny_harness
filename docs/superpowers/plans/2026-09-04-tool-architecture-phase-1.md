# TinyHarness Tool Architecture Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each built-in Tool a self-contained definition discovered at run initialization while preserving the shared permission, hook, and event execution pipeline.

**Architecture:** Add an immutable `ToolDefinition` that owns model metadata and its bound executor, a small convention-based discovery function for `build_tools(context)` exports, and a `ToolRegistry` limited to registration, lookup, listing, duplicate validation, and schema projection. Run composition constructs capabilities first, discovery asks each Tool module to bind only the capabilities it needs, and batch execution sends calls through the shared pipeline to the selected definition.

**Tech Stack:** Python 3.10+, dataclasses, `pkgutil`/`importlib`, unittest/pytest.

**Spec:** The Phase 1 Tool Architecture requirements in the current Codex task dated 2026-09-04.

## Global Constraints

- Only refactor TinyHarness built-in Tools; do not change Skills, Memory, examples, or introduce a plugin framework.
- Discovery occurs during Session/Run initialization with no decorators, global registration side effects, entry points, hot reload, or dependency-injection container.
- Permission policy remains centralized security policy, outside `ToolDefinition`.
- Permission, hooks, and events remain one shared execution pipeline.
- Preserve all unrelated existing worktree changes.

---

### Task 1: Definition and Registry Boundary

**Files:**
- Create: `tiny_harness/tools/definition.py`
- Modify: `tiny_harness/tools/registry.py`
- Test: `tests/test_tool_architecture.py`

**Interfaces:**
- Consumes: `ToolCall` and the existing model function-tool schema format.
- Produces: `ToolDefinition(name, description, parameters, execute)` and `ToolRegistry.register`, `lookup`, `list`, and `model_schemas`.

- [x] **Step 1: Write failing registration, duplicate, and schema-source tests**

```python
definition = ToolDefinition("echo", "Echo text.", parameters, execute)
registry = ToolRegistry()
registry.register(definition)
assert registry.lookup("echo") is definition
assert registry.model_schemas()[0]["function"]["description"] == definition.description
with pytest.raises(ValueError, match="Duplicate tool name: echo"):
    registry.register(definition)
```

- [x] **Step 2: Run the new tests and confirm imports fail because the boundary does not exist**

Run: `python -m pytest tests/test_tool_architecture.py -q`

Expected: collection failure for missing `ToolDefinition` or `ToolRegistry` API.

- [x] **Step 3: Implement the minimal definition and registry**

```python
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: ToolExecutor

class ToolRegistry:
    def register(self, definition: ToolDefinition) -> None: ...
    def lookup(self, name: str) -> ToolDefinition: ...
    def list(self) -> tuple[ToolDefinition, ...]: ...
    def model_schemas(self) -> list[dict[str, Any]]: ...
```

- [x] **Step 4: Run the boundary tests and confirm they pass**

Run: `python -m pytest tests/test_tool_architecture.py -q`

Expected: all Task 1 tests pass.

### Task 2: Convention-Based Discovery

**Files:**
- Create: `tiny_harness/tools/discovery.py`
- Modify: `tests/test_tool_architecture.py`

**Interfaces:**
- Consumes: an importable package name and an opaque run context passed unchanged to Tool factories.
- Produces: `discover_tools(context, package_name=...) -> ToolRegistry`, recognizing only callable module exports named `build_tools`.

- [x] **Step 1: Add failing temporary-package discovery tests**

```python
registry = discover_tools(context, package_name="sample_tools")
assert [item.name for item in registry.list()] == ["first", "second"]
```

The fixture package contains one module exporting two definitions and another ordinary module without the convention.

- [x] **Step 2: Run the discovery tests and confirm the missing implementation failure**

Run: `python -m pytest tests/test_tool_architecture.py -q`

Expected: failure because `discover_tools` is absent.

- [x] **Step 3: Implement deterministic module scanning and factory validation**

```python
for module_info in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
    module = importlib.import_module(f"{package.__name__}.{module_info.name}")
    factory = getattr(module, "build_tools", None)
    if callable(factory):
        for definition in factory(context):
            registry.register(definition)
```

- [x] **Step 4: Run tests and confirm multi-Tool discovery and duplicate failures pass**

Run: `python -m pytest tests/test_tool_architecture.py -q`

Expected: all discovery tests pass.

### Task 3: Self-Contained Built-in Tool Modules

**Files:**
- Modify: `tiny_harness/tools/filesystem.py`
- Modify: `tiny_harness/tools/shell.py`
- Modify: `tiny_harness/tools/task.py`
- Modify: `tiny_harness/tools/skill.py`
- Modify: `tiny_harness/tools/compact.py`
- Modify: `tiny_harness/tools/todo.py`
- Create: `tiny_harness/tools/testing.py`
- Modify: `tests/test_registry.py`
- Modify: `tests/test_test_runner.py`

**Interfaces:**
- Consumes: the existing run context and each module's existing handler/runtime capability.
- Produces: one `build_tools(context)` factory per Tool module, returning zero definitions when its required capability is unavailable.

- [x] **Step 1: Change registry behavior tests to construct a discovered registry and add absent-capability assertions**

```python
context.subagent_runner = None
registry = discover_tools(context)
assert "task" not in {item.name for item in registry.list()}
```

- [x] **Step 2: Run focused tests and confirm they fail because built-ins do not export factories**

Run: `python -m pytest tests/test_tool_architecture.py tests/test_registry.py tests/test_test_runner.py -q`

Expected: built-in Tools are missing from discovery.

- [x] **Step 3: Move every description, JSON schema, executor binding, and availability decision into its owning module**

```python
def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    if context.subagent_runner is None:
        return ()
    return (ToolDefinition("task", "Run a subagent ...", TASK_PARAMETERS, execute),)
```

- [x] **Step 4: Run focused tests and confirm built-in behavior and capability filtering pass**

Run: `python -m pytest tests/test_tool_architecture.py tests/test_registry.py tests/test_test_runner.py -q`

Expected: all focused tests pass.

### Task 4: Run Composition and Shared Execution Pipeline

**Files:**
- Modify: `tiny_harness/agent/context.py`
- Modify: `tiny_harness/agent/tool_batch.py`
- Modify: `tiny_harness/agent/turn.py`
- Modify: `tests/test_hooks.py`
- Modify: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: a run-scoped `ToolRegistry` and the existing permission/hook/event objects.
- Produces: discovery during `create_run_context`, model schemas projected from that registry, and `dispatch(registry, call, ...)` with no subsystem-specific arguments.

- [x] **Step 1: Add a failing Agent Loop test using a discovered custom Tool name**

```python
with patch("tiny_harness.agent.context.discover_tools", return_value=custom_registry):
    answer = run_agent(provider, workspace, messages)
assert answer == "finished"
assert observed == [{"value": "custom"}]
```

- [x] **Step 2: Run Agent Loop and pipeline tests and confirm old composition assumptions fail**

Run: `python -m pytest tests/test_agent_loop.py tests/test_hooks.py tests/test_events.py -q`

Expected: failure until the run context owns and dispatches through the registry.

- [x] **Step 3: Compose capabilities before discovery and route all calls through the generic pipeline**

```python
context.tool_registry = discover_tools(context)
response = context.recovery_executor.complete(
    context.provider, request_messages, context.tool_registry.model_schemas(), ...
)
result = dispatch(context.tool_registry, call, permission_policy=..., tool_hooks=...)
```

- [x] **Step 4: Run pipeline tests and confirm Permission/Hooks/Events behavior is unchanged**

Run: `python -m pytest tests/test_agent_loop.py tests/test_hooks.py tests/test_events.py tests/test_permissions.py -q`

Expected: all tests pass with original event ordering and permission outcomes.

### Task 5: Verification and Scope Audit

**Files:**
- Verify all modified Tool and directly affected runtime files.

**Interfaces:**
- Consumes: the completed Phase 1 diff.
- Produces: fresh test evidence, review findings, diff stat, status, and a Phase 1-only report.

- [x] **Step 1: Run the Tool and directly affected Runtime test set**

Run: `python -m pytest tests/test_tool_architecture.py tests/test_registry.py tests/test_filesystem.py tests/test_permissions.py tests/test_hooks.py tests/test_events.py tests/test_agent_loop.py tests/test_session.py tests/test_context.py tests/test_turn.py tests/test_test_runner.py tests/test_todos.py tests/test_subagent.py tests/test_subagent_executor.py tests/test_skills.py tests/test_skill_runtime.py -q`

Expected: zero failures.

- [x] **Step 2: Run the complete suite if the focused set passes**

Run: `python -m pytest -q`

Expected: zero failures, with environment-related skips reported separately.

- [x] **Step 3: Review the diff against every Phase 1 requirement**

Run: `git diff -- tiny_harness/tools tiny_harness/agent/context.py tiny_harness/agent/tool_batch.py tiny_harness/agent/turn.py tests`

Expected: no Skills/Memory/example architecture changes and no Tool-specific mapping left in the registry.

- [x] **Step 4: Capture final repository evidence**

Run: `git diff --stat` and `git status --short --branch`.

Expected: Phase 1 changes are distinguishable from pre-existing unrelated changes.
