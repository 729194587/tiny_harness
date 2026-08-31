"""Agent Loop 的单次运行上下文与装配入口。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiny_harness.agent.messages import ModelResponse
from tiny_harness.agent.subagent import SubagentExecutor
from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.context import CompactionRequest, ContextCompactor
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, EventLogger
from tiny_harness.runtime.errors import MaxTurnsExceededError
from tiny_harness.runtime.hooks import FinalAnswerHook, ToolHooks
from tiny_harness.runtime.memory import (
    LoadedMemories,
    MemoryCatalog,
    MemoryComplete,
    create_memory_final_answer_hook,
    discover_memories,
    empty_memory_catalog,
    prepare_memory_context,
    upsert_memory_markers,
)
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
    PermissionRejectionTracker,
)
from tiny_harness.runtime.recovery import (
    RecoveryExecutor,
    RecoveryPolicy,
    RecoveryState,
)
from tiny_harness.runtime.skills import (
    SkillCatalog,
    discover_skills,
    upsert_skill_catalog_marker,
)
from tiny_harness.runtime.test_runner import TestRunner
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.registry import tool_schemas
from tiny_harness.tools.task import SubagentRunner

DEFAULT_SUBAGENT_MAX_TURNS = 10


@dataclass
class AgentRunContext:
    """一次 Agent 运行所需的配置、资源和可变运行状态。

    这是依赖容器，不负责实现 Agent Loop，也不通过方法隐藏编排动作。
    """

    provider: ModelProvider
    workspace: Path
    tools: list[dict[str, Any]]
    max_context_chars: int | None
    max_turns: int
    subagent_max_turns: int
    allow_subagent: bool
    permission_policy: PermissionPolicy
    permission_prompt: PermissionPrompt | None
    tool_hooks: ToolHooks | None
    recovery_policy: RecoveryPolicy
    event_logger: EventLogger
    recovery_executor: RecoveryExecutor
    todo_manager: TodoManager
    skill_catalog: SkillCatalog
    memory_enabled: bool
    memory_catalog: MemoryCatalog
    memory_selection_complete: MemoryComplete | None
    compactor: ContextCompactor | None
    compaction_request: CompactionRequest | None
    subagent_runner: SubagentRunner | None
    final_answer_hook: FinalAnswerHook | None
    test_runner: TestRunner | None
    permission_rejections: PermissionRejectionTracker
    current_turn: int = 0
    rounds_since_todo: int = 0


def run_started_data(context: AgentRunContext) -> dict[str, Any]:
    """返回不含提示词和工具结果的运行元数据。"""

    data: dict[str, Any] = {
        "max_turns": context.max_turns,
        "max_model_retries": context.recovery_policy.max_retries,
    }
    if context.allow_subagent:
        data["subagent_max_turns"] = context.subagent_max_turns
    if context.max_context_chars is not None:
        data["max_context_chars"] = context.max_context_chars
    if context.skill_catalog.manifests or context.skill_catalog.issues:
        data["skills_available"] = len(context.skill_catalog.manifests)
        data["skill_discovery_issues"] = len(context.skill_catalog.issues)
    if context.memory_enabled:
        data["memory_enabled"] = True
        data["memories_available"] = len(context.memory_catalog.manifests)
        data["memory_discovery_issues"] = len(context.memory_catalog.issues)
    return data


def initialize_run_state(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
    active_request: str,
) -> None:
    """在第一次模型调用前插入可选的 run-scoped 控制状态。"""

    upsert_skill_catalog_marker(messages, context.skill_catalog)
    if context.memory_enabled:
        if context.memory_selection_complete is None:
            raise RuntimeError("Memory selection requires a completion callback")
        prepare_memory_context(
            messages,
            context.memory_catalog,
            active_request,
            context.memory_selection_complete,
            context.event_logger,
            max_context_chars=context.max_context_chars,
        )
    else:
        upsert_memory_markers(
            messages,
            context.memory_catalog,
            LoadedMemories("", (), ()),
        )


def turn_limit_error(context: AgentRunContext) -> Exception:
    """Return the dedicated error for a run that never ends naturally."""

    return MaxTurnsExceededError(
        f"Maximum model turns reached: {context.max_turns}"
    )


ModelCompletion = Callable[
    [str, list[dict[str, Any]], list[dict[str, Any]]],
    ModelResponse,
]


@dataclass(frozen=True)
class CompactionRuntime:
    compactor: ContextCompactor | None
    request: CompactionRequest | None


def _validate_run_configuration(
    *,
    max_turns: int,
    max_context_chars: int | None,
    subagent_max_turns: int,
) -> None:
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_context_chars is not None and max_context_chars < 1:
        raise ValueError("max_context_chars must be at least 1")
    if subagent_max_turns < 1:
        raise ValueError("subagent_max_turns must be at least 1")


def _memory_catalog(workspace: Path, enabled: bool) -> MemoryCatalog:
    return discover_memories(workspace) if enabled else empty_memory_catalog(workspace)


def _completion_router(
    provider: ModelProvider,
    recovery: RecoveryExecutor,
    current_turn: Callable[[], int],
) -> ModelCompletion:
    def complete_for(
        purpose: str,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return recovery.complete(
            provider,
            request_messages,
            request_tools,
            purpose=purpose,
            turn=current_turn(),
            state=RecoveryState(),
        )

    return complete_for


def _compaction_runtime(
    workspace: Path,
    provider: ModelProvider,
    tools: list[dict[str, Any]],
    max_context_chars: int | None,
    complete_for: ModelCompletion,
    event_logger: EventLogger,
) -> CompactionRuntime:
    if max_context_chars is None:
        return CompactionRuntime(None, None)
    return CompactionRuntime(
        ContextCompactor(
            workspace,
            provider,
            tools,
            max_context_chars,
            event_logger=event_logger,
            summary_complete=lambda messages, schemas: complete_for(
                "summary", messages, schemas
            ),
        ),
        CompactionRequest(),
    )


def _subagent_runner(
    *,
    enabled: bool,
    provider: ModelProvider,
    workspace: Path,
    max_turns: int,
    permission_policy: PermissionPolicy,
    permission_prompt: PermissionPrompt | None,
    event_logger: EventLogger,
    max_context_chars: int | None,
    tool_hooks: ToolHooks | None,
    recovery_policy: RecoveryPolicy,
    memory_enabled: bool,
    test_runner: TestRunner | None,
) -> SubagentRunner | None:
    if not enabled:
        return None

    # Import the composition entry point lazily to avoid context <-> loop
    # initialization recursion. The core Agent Loop remains dependency-free.
    from tiny_harness.agent.loop import run_agent

    return SubagentExecutor(
        run_agent,
        provider,
        workspace,
        max_turns=max_turns,
        permission_policy=permission_policy,
        permission_prompt=permission_prompt,
        event_logger=event_logger,
        max_context_chars=max_context_chars,
        tool_hooks=tool_hooks,
        recovery_policy=recovery_policy,
        memory_enabled=memory_enabled,
        test_runner=test_runner,
    )


def _final_answer_hook(
    *,
    memory_enabled: bool,
    memory_extraction_enabled: bool,
    memory_catalog: MemoryCatalog,
    complete_for: ModelCompletion,
    event_logger: EventLogger,
    max_context_chars: int | None,
) -> FinalAnswerHook | None:
    return (
        create_memory_final_answer_hook(
            memory_catalog,
            lambda messages, schemas: complete_for(
                "memory_extraction", messages, schemas
            ),
            lambda messages, schemas: complete_for(
                "memory_consolidation", messages, schemas
            ),
            event_logger,
            max_context_chars=max_context_chars,
        )
        if memory_enabled and memory_extraction_enabled
        else None
    )


def create_run_context(
    provider: ModelProvider,
    workspace: Path,
    *,
    max_turns: int = 20,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
    event_logger: EventLogger = NULL_EVENT_LOGGER,
    max_context_chars: int | None = None,
    tool_hooks: ToolHooks | None = None,
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
    allow_subagent: bool = True,
    recovery_policy: RecoveryPolicy = RecoveryPolicy(),
    test_runner: TestRunner | None = None,
    memory_enabled: bool = False,
    memory_extraction_enabled: bool = True,
) -> AgentRunContext:
    """Compose one run from top-level policy to concrete runtime state."""

    _validate_run_configuration(
        max_turns=max_turns,
        max_context_chars=max_context_chars,
        subagent_max_turns=subagent_max_turns,
    )

    skill_catalog = discover_skills(workspace)
    memory_catalog = _memory_catalog(workspace, memory_enabled)
    tools = tool_schemas(
        include_task=allow_subagent,
        include_skill=bool(skill_catalog.manifests),
        include_compact=max_context_chars is not None,
        include_run_tests=test_runner is not None,
    )
    todo_manager = TodoManager()

    recovery = RecoveryExecutor(recovery_policy, event_logger=event_logger)
    context: AgentRunContext
    complete_for = _completion_router(
        provider,
        recovery,
        lambda: context.current_turn,
    )

    compaction = _compaction_runtime(
        workspace,
        provider,
        tools,
        max_context_chars,
        complete_for,
        event_logger,
    )
    subagent = _subagent_runner(
        enabled=allow_subagent,
        provider=provider,
        workspace=workspace,
        max_turns=subagent_max_turns,
        permission_policy=permission_policy,
        permission_prompt=permission_prompt,
        event_logger=event_logger,
        max_context_chars=max_context_chars,
        tool_hooks=tool_hooks,
        recovery_policy=recovery_policy,
        memory_enabled=memory_enabled,
        test_runner=test_runner,
    )
    final_answer_hook = _final_answer_hook(
        memory_enabled=memory_enabled,
        memory_extraction_enabled=memory_extraction_enabled,
        memory_catalog=memory_catalog,
        complete_for=complete_for,
        event_logger=event_logger,
        max_context_chars=max_context_chars,
    )

    context = AgentRunContext(
        provider=provider,
        workspace=workspace,
        tools=tools,
        max_context_chars=max_context_chars,
        max_turns=max_turns,
        subagent_max_turns=subagent_max_turns,
        allow_subagent=allow_subagent,
        permission_policy=permission_policy,
        permission_prompt=permission_prompt,
        tool_hooks=tool_hooks,
        recovery_policy=recovery_policy,
        event_logger=event_logger,
        recovery_executor=recovery,
        todo_manager=todo_manager,
        skill_catalog=skill_catalog,
        memory_enabled=memory_enabled,
        memory_catalog=memory_catalog,
        memory_selection_complete=(
            (
                lambda messages, schemas: complete_for(
                    "memory_selection", messages, schemas
                )
            )
            if memory_enabled
            else None
        ),
        compactor=compaction.compactor,
        compaction_request=compaction.request,
        subagent_runner=subagent,
        final_answer_hook=final_answer_hook,
        test_runner=test_runner,
        permission_rejections=PermissionRejectionTracker(),
    )
    return context
