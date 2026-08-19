"""Agent Loop 的单次运行上下文与装配入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiny_harness.agent.messages import ModelResponse
from tiny_harness.agent.subagent import SubagentExecutor
from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.context import CompactionRequest, ContextCompactor
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, EventLogger
from tiny_harness.runtime.goal import (
    DEFAULT_MAX_GOAL_RETRIES,
    GoalEvaluator,
    GoalNotAchievedError,
    GoalState,
    PromptGoalEvaluator,
    create_goal_state,
    create_goal_stop_hook,
    upsert_goal_marker,
)
from tiny_harness.runtime.hooks import StopHook, ToolHooks
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
)
from tiny_harness.runtime.recovery import (
    RecoveryExecutor,
    RecoveryPolicy,
    RecoveryState,
)
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
    compactor: ContextCompactor | None
    compaction_request: CompactionRequest | None
    subagent_runner: SubagentRunner | None
    goal_state: GoalState | None
    stop_hook: StopHook | None
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
    if context.goal_state is not None:
        data["goal_enabled"] = True
        data["max_goal_retries"] = context.goal_state.max_retries
    return data


def initialize_run_state(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
) -> None:
    """在第一次模型调用前插入可选的 run-scoped 控制状态。"""

    if context.goal_state is not None:
        upsert_goal_marker(messages, context.goal_state)


def run_finished_data(context: AgentRunContext) -> dict[str, Any]:
    """返回最终生命周期事件使用的非敏感控制元数据。"""

    if context.goal_state is None:
        return {}
    return {"goal_evaluations": context.goal_state.evaluations}


def turn_limit_error(context: AgentRunContext) -> Exception:
    """根据是否启用 Goal 返回明确的轮次上限错误。"""

    if context.goal_state is not None:
        return GoalNotAchievedError(
            "Maximum model turns reached before goal verification: "
            f"{context.max_turns}"
        )
    return RuntimeError(f"Maximum model turns reached: {context.max_turns}")


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
    goal_condition: str | None = None,
    max_goal_retries: int = DEFAULT_MAX_GOAL_RETRIES,
    goal_evaluator: GoalEvaluator | None = None,
) -> AgentRunContext:
    """校验配置并装配一次运行所需的依赖。"""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_context_chars is not None and max_context_chars < 1:
        raise ValueError("max_context_chars must be at least 1")
    if subagent_max_turns < 1:
        raise ValueError("subagent_max_turns must be at least 1")
    if max_goal_retries < 0:
        raise ValueError("max_goal_retries must be at least 0")
    if goal_evaluator is not None and goal_condition is None:
        raise ValueError("goal_evaluator requires goal_condition")

    tools = tool_schemas(
        include_task=allow_subagent,
        include_compact=max_context_chars is not None,
    )
    todo_manager = TodoManager()
    recovery_executor = RecoveryExecutor(
        recovery_policy,
        event_logger=event_logger,
    )

    # 闭包需要读取当前物理请求所属的逻辑 turn；context 在下方装配完成。
    context_ref: list[AgentRunContext] = []

    def complete_for(
        purpose: str,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return recovery_executor.complete(
            provider,
            request_messages,
            request_tools,
            purpose=purpose,
            turn=context_ref[0].current_turn,
            state=RecoveryState(),
        )

    goal_state: GoalState | None = None
    stop_hook: StopHook | None = None
    if goal_condition is not None:
        evaluator = goal_evaluator or PromptGoalEvaluator(
            lambda messages, schemas: complete_for(
                "goal_evaluation", messages, schemas
            ),
            max_context_chars=max_context_chars,
        )
        goal_state = create_goal_state(
            goal_condition,
            max_retries=max_goal_retries,
        )
        stop_hook = create_goal_stop_hook(
            goal_state,
            evaluator,
            event_logger,
        )

    compaction_request = (
        CompactionRequest() if max_context_chars is not None else None
    )
    compactor = (
        ContextCompactor(
            workspace,
            provider,
            tools,
            max_context_chars,
            event_logger=event_logger,
            summary_complete=lambda messages, schemas: complete_for(
                "summary", messages, schemas
            ),
        )
        if max_context_chars is not None
        else None
    )
    subagent_runner = None
    if allow_subagent:
        # 局部导入避免 composition 层与核心循环形成模块初始化环。
        from tiny_harness.agent.loop import run_agent

        subagent_runner = SubagentExecutor(
            run_agent,
            provider,
            workspace,
            max_turns=subagent_max_turns,
            permission_policy=permission_policy,
            permission_prompt=permission_prompt,
            event_logger=event_logger,
            max_context_chars=max_context_chars,
            tool_hooks=tool_hooks,
            recovery_policy=recovery_policy,
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
        recovery_executor=recovery_executor,
        todo_manager=todo_manager,
        compactor=compactor,
        compaction_request=compaction_request,
        subagent_runner=subagent_runner,
        goal_state=goal_state,
        stop_hook=stop_hook,
    )
    context_ref.append(context)
    return context
