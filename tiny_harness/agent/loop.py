"""TinyHarness 的稳定 Agent Loop 与配置装配入口。"""

from pathlib import Path
from typing import Any

from tiny_harness.agent.context import (
    DEFAULT_SUBAGENT_MAX_TURNS,
    AgentRunContext,
    create_run_context,
    initialize_run_state,
    run_finished_data,
    run_started_data,
    turn_limit_error,
)
from tiny_harness.agent.messages import assistant_message_from_response
from tiny_harness.agent.tool_batch import execute_tool_batch
from tiny_harness.agent.turn import call_model
from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.context import prepare_context
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
)
from tiny_harness.runtime.goal import (
    DEFAULT_MAX_GOAL_RETRIES,
    GoalEvaluator,
)
from tiny_harness.runtime.hooks import ToolHooks, run_stop_hook
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
)
from tiny_harness.runtime.recovery import RecoveryPolicy


def agent_loop(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
    active_request: str,
) -> str:
    """执行循环，直到返回最终回答。"""

    context.event_logger.emit(
        EventType.RUN_STARTED,
        run_started_data(context),
    )
    initialize_run_state(messages, context, active_request)

    try:
        for turn in range(1, context.max_turns + 1):
            context.current_turn = turn
            prepared = prepare_context(
                messages,
                context.compactor,
                context.todo_manager.render(),
                active_request,
            )
            if prepared is not None:
                # 只有四层处理和最终验证全部成功后，才提交canonical history。
                messages[:] = prepared.messages

            response = call_model(messages, context)
            assistant_message = assistant_message_from_response(response)

            if not response.tool_calls:
                answer = response.content or ""
                context.event_logger.emit(
                    EventType.STOP_PROPOSED,
                    {
                        "turn": turn,
                        "answer_length": len(answer),
                        "has_next_turn": turn < context.max_turns,
                    },
                )
                stop_decision = run_stop_hook(
                    context.stop_hook,
                    messages,
                    answer,
                    turn=turn,
                    has_next_turn=turn < context.max_turns,
                )
                context.event_logger.emit(
                    EventType.STOP_DECIDED,
                    {
                        "turn": turn,
                        "action": stop_decision.action,
                    },
                )
                if stop_decision.action == "block":
                    continue

                messages.append(assistant_message)
                finish_data = {
                    "turns": turn,
                    "answer_length": len(answer),
                }
                finish_data.update(run_finished_data(context))
                context.event_logger.emit(
                    EventType.RUN_FINISHED,
                    finish_data,
                )
                return answer

            messages.append(assistant_message)
            execute_tool_batch(
                messages,
                response.tool_calls,
                context,
            )

        raise turn_limit_error(context)
    except EventLogError:
        # Event Logger 自身失败时直接上抛，避免再次记录同一个故障。
        raise
    except Exception as error:
        context.event_logger.emit(
            EventType.RUN_FAILED,
            {
                "turn": context.current_turn,
                "error_type": type(error).__name__,
            },
        )
        raise


def run_agent(
    provider: ModelProvider,
    workspace: Path,
    messages: list[dict[str, Any]],
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
    inject_goal_context: bool = True,
    memory_enabled: bool = False,
    memory_extraction_enabled: bool = True,
) -> str:
    """兼容配置入口：装配运行上下文后进入三参数核心循环。"""

    context = create_run_context(
        provider,
        workspace,
        max_turns=max_turns,
        permission_policy=permission_policy,
        permission_prompt=permission_prompt,
        event_logger=event_logger,
        max_context_chars=max_context_chars,
        tool_hooks=tool_hooks,
        subagent_max_turns=subagent_max_turns,
        allow_subagent=allow_subagent,
        recovery_policy=recovery_policy,
        goal_condition=goal_condition,
        max_goal_retries=max_goal_retries,
        goal_evaluator=goal_evaluator,
        inject_goal_context=inject_goal_context,
        memory_enabled=memory_enabled,
        memory_extraction_enabled=memory_extraction_enabled,
    )
    active_request = next(
        (
            str(message.get("content") or "")
            for message in reversed(messages)
            if message.get("role") == "user" and not message.get("name")
        ),
        "",
    )
    return agent_loop(messages, context, active_request)
