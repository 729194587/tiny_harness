"""Agent Loop 的单次运行上下文与装配入口。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from tiny_harness.agent.messages import ModelResponse
from tiny_harness.agent.subagent import SubagentExecutor
from tiny_harness.context.token_meter import DEFAULT_TOKEN_METER, TokenMeter, CalibratedTokenMeter
from tiny_harness.memory import MemoryRuntime, create_memory_runtime
from tiny_harness.models.base import ModelProvider, ToolChoice
from tiny_harness.runtime.context import CompactionConfig, CompactionRequest, ContextCompactor
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, EventLogger
from tiny_harness.runtime.hooks import FinalAnswerHook, ToolHooks
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
    PermissionRejectionTracker,
)
from tiny_harness.runtime.tool_trace import ToolTraceConfig
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
from tiny_harness.runtime.shell_runner import DEFAULT_SHELL_RUNNER, ShellRunner
from tiny_harness.runtime.test_runner import TestRunner
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.discovery import discover_tools
from tiny_harness.tools.registry import ToolRegistry
from tiny_harness.tools.task import SubagentRunner

DEFAULT_SUBAGENT_MAX_TURNS = 10


@dataclass
class AgentRunContext:
    """一次 Agent 运行所需的配置、资源和可变运行状态。

    这是依赖容器，不负责实现 Agent Loop，也不通过方法隐藏编排动作。
    """

    provider: ModelProvider
    workspace: Path
    tool_registry: ToolRegistry
    max_context_tokens: int | None
    token_meter: TokenMeter
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
    memory: MemoryRuntime
    compactor: ContextCompactor | None
    compaction_request: CompactionRequest | None
    subagent_runner: SubagentRunner | None
    final_answer_hook: FinalAnswerHook | None
    test_runner: TestRunner | None
    shell_runner: ShellRunner
    permission_rejections: PermissionRejectionTracker
    current_turn: int = 0
    last_finish_reason: str | None = None
    tool_trace: ToolTraceConfig = ToolTraceConfig()
    compaction_config: CompactionConfig = CompactionConfig()

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Project the run's discovered Tools for model requests."""

        return self.tool_registry.model_schemas()


def run_started_data(context: AgentRunContext) -> dict[str, Any]:
    """返回不含提示词和工具结果的运行元数据。"""

    data: dict[str, Any] = {
        "max_turns": context.max_turns,
        "max_model_retries": context.recovery_policy.max_retries,
        "working_context_trigger_tokens": context.compaction_config.working_context_trigger_tokens,
        "working_context_target_tokens": context.compaction_config.working_context_target_tokens,
    }
    if context.allow_subagent:
        data["subagent_max_turns"] = context.subagent_max_turns
    if context.max_context_tokens is not None:
        data["max_context_tokens"] = context.max_context_tokens
    if context.skill_catalog.manifests or context.skill_catalog.issues:
        data["skills_available"] = len(context.skill_catalog.manifests)
        data["skill_discovery_issues"] = len(context.skill_catalog.issues)
    data.update(context.memory.run_metadata())
    return data


def initialize_run_state(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
    active_request: str,
) -> None:
    """在第一次模型调用前插入可选的 run-scoped 控制状态。"""

    upsert_skill_catalog_marker(messages, context.skill_catalog)
    context.memory.initialize(messages, active_request)


class ModelCompletion(Protocol):
    def __call__(
        self, purpose: str, request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]], *,
        tool_choice: ToolChoice | None = None,
        state: RecoveryState | None = None,
    ) -> ModelResponse: ...


def _validate_run_configuration(
    *,
    max_turns: int,
    max_context_tokens: int | None,
    subagent_max_turns: int,
) -> None:
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_context_tokens is not None and max_context_tokens < 1:
        raise ValueError("max_context_tokens must be at least 1")
    if subagent_max_turns < 1:
        raise ValueError("subagent_max_turns must be at least 1")


def _completion_router(
    provider: ModelProvider,
    recovery: RecoveryExecutor,
    current_turn: Callable[[], int],
) -> ModelCompletion:
    def complete_for(
        purpose: str,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
        *,
        tool_choice: ToolChoice | None = None,
        state: RecoveryState | None = None,
    ) -> ModelResponse:
        return recovery.complete(
            provider,
            request_messages,
            request_tools,
            purpose=purpose,
            turn=current_turn(),
            state=state if state is not None else RecoveryState(),
            tool_choice=tool_choice,
        )

    return complete_for


def _compactor(
    workspace: Path,
    provider: ModelProvider,
    tools: list[dict[str, Any]],
    max_context_tokens: int | None,
    token_meter: TokenMeter,
    complete_for: ModelCompletion,
    event_logger: EventLogger,
    config: CompactionConfig,
) -> ContextCompactor | None:
    if max_context_tokens is None:
        return None
    return ContextCompactor(
        workspace,
        provider,
        tools,
        max_context_tokens,
        token_meter=token_meter,
        event_logger=event_logger,
        config=config,
        summary_complete=partial(complete_for, "summary"),
    )


@dataclass(frozen=True)
class RunConfig:
    """Inheritable dependencies, separate from per-run execution state."""

    provider: ModelProvider
    workspace: Path
    max_turns: int = 20
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY
    permission_prompt: PermissionPrompt | None = None
    event_logger: EventLogger = NULL_EVENT_LOGGER
    max_context_tokens: int | None = None
    working_context_trigger_tokens: int = CompactionConfig.working_context_trigger_tokens
    working_context_target_tokens: int = CompactionConfig.working_context_target_tokens
    token_meter: TokenMeter = DEFAULT_TOKEN_METER
    tool_hooks: ToolHooks | None = None
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS
    allow_subagent: bool = True
    recovery_policy: RecoveryPolicy = RecoveryPolicy()
    test_runner: TestRunner | None = None
    shell_runner: ShellRunner = DEFAULT_SHELL_RUNNER
    skill_catalog: SkillCatalog | None = None
    memory_enabled: bool = False
    memory_extraction_enabled: bool = True
    tool_trace: ToolTraceConfig = ToolTraceConfig()


def create_run_context(
    provider: ModelProvider,
    workspace: Path,
    *,
    max_turns: int = 20,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
    event_logger: EventLogger = NULL_EVENT_LOGGER,
    max_context_tokens: int | None = None,
    working_context_trigger_tokens: int = CompactionConfig.working_context_trigger_tokens,
    working_context_target_tokens: int = CompactionConfig.working_context_target_tokens,
    token_meter: TokenMeter = DEFAULT_TOKEN_METER,
    tool_hooks: ToolHooks | None = None,
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
    allow_subagent: bool = True,
    recovery_policy: RecoveryPolicy = RecoveryPolicy(),
    test_runner: TestRunner | None = None,
    shell_runner: ShellRunner = DEFAULT_SHELL_RUNNER,
    skill_catalog: SkillCatalog | None = None,
    memory_enabled: bool = False,
    memory_extraction_enabled: bool = True,
    tool_trace: ToolTraceConfig = ToolTraceConfig(),
) -> AgentRunContext:
    """Compose one run from top-level policy to concrete runtime state."""

    return build_run_context(RunConfig(
        provider=provider,
        workspace=workspace,
        max_turns=max_turns,
        permission_policy=permission_policy,
        permission_prompt=permission_prompt,
        event_logger=event_logger,
        max_context_tokens=max_context_tokens,
        working_context_trigger_tokens=working_context_trigger_tokens,
        working_context_target_tokens=working_context_target_tokens,
        token_meter=token_meter,
        tool_hooks=tool_hooks,
        subagent_max_turns=subagent_max_turns,
        allow_subagent=allow_subagent,
        recovery_policy=recovery_policy,
        test_runner=test_runner,
        shell_runner=shell_runner,
        skill_catalog=skill_catalog,
        memory_enabled=memory_enabled,
        memory_extraction_enabled=memory_extraction_enabled,
        tool_trace=tool_trace,
    ))


def build_run_context(config: RunConfig) -> AgentRunContext:
    """Build fresh runtime state from one explicit configuration source."""

    _validate_run_configuration(
        max_turns=config.max_turns,
        max_context_tokens=config.max_context_tokens,
        subagent_max_turns=config.subagent_max_turns,
    )
    compaction_config = CompactionConfig(
        working_context_trigger_tokens=config.working_context_trigger_tokens,
        working_context_target_tokens=config.working_context_target_tokens,
    )
    if config.max_context_tokens is not None and config.working_context_trigger_tokens >= config.max_context_tokens:
        raise ValueError("working context requires target < trigger < max_context_tokens")

    if (
        config.skill_catalog is not None
        and config.skill_catalog.workspace != config.workspace.resolve()
    ):
        raise ValueError("Skill catalog workspace does not match run workspace")
    active_skill_catalog = (
        discover_skills(config.workspace) if config.skill_catalog is None else config.skill_catalog
    )
    todo_manager = TodoManager()
    compaction_request = (
        CompactionRequest() if config.max_context_tokens is not None else None
    )

    token_meter = (config.token_meter if isinstance(config.token_meter, CalibratedTokenMeter)
                   else CalibratedTokenMeter(config.token_meter))
    recovery = RecoveryExecutor(config.recovery_policy, event_logger=config.event_logger)
    context: AgentRunContext
    complete_for = _completion_router(
        config.provider,
        recovery,
        lambda: context.current_turn,
    )
    memory = create_memory_runtime(
        config.workspace,
        enabled=config.memory_enabled,
        extraction_enabled=config.memory_extraction_enabled,
        complete_for=complete_for,
        event_logger=config.event_logger,
        max_context_tokens=config.max_context_tokens,
        token_meter=token_meter,
    )

    # Children inherit the resolved skill catalog snapshot.
    config = replace(config, skill_catalog=active_skill_catalog)
    subagent = SubagentExecutor(config) if config.allow_subagent else None
    context = AgentRunContext(
        compaction_config=compaction_config,
        provider=config.provider,
        workspace=config.workspace,
        tool_registry=ToolRegistry(),
        max_context_tokens=config.max_context_tokens,
        token_meter=token_meter,
        max_turns=config.max_turns,
        subagent_max_turns=config.subagent_max_turns,
        allow_subagent=config.allow_subagent,
        permission_policy=config.permission_policy,
        permission_prompt=config.permission_prompt,
        tool_hooks=config.tool_hooks,
        recovery_policy=config.recovery_policy,
        event_logger=config.event_logger,
        recovery_executor=recovery,
        todo_manager=todo_manager,
        skill_catalog=active_skill_catalog,
        memory=memory,
        compactor=None,
        compaction_request=compaction_request,
        subagent_runner=subagent,
        final_answer_hook=memory.final_answer_hook,
        test_runner=config.test_runner,
        shell_runner=config.shell_runner,
        permission_rejections=PermissionRejectionTracker(),
        tool_trace=config.tool_trace,
    )
    context.tool_registry = discover_tools(context)
    context.compactor = _compactor(
        config.workspace,
        config.provider,
        context.tools,
        config.max_context_tokens,
        token_meter,
        complete_for,
        config.event_logger,
        compaction_config,
    )
    return context
