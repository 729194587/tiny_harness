"""TinyHarness 的稳定 Agent Loop 与配置装配入口。"""

from pathlib import Path
from typing import Any

from tiny_harness.agent.context import (
    DEFAULT_SUBAGENT_MAX_TURNS,
    AgentRunContext,
    create_run_context,
    initialize_run_state,
    run_started_data,
)
from tiny_harness.agent.messages import ModelResponse, assistant_message_from_response
from tiny_harness.agent.environment import EnvironmentAdapter
from tiny_harness.agent.tool_batch import execute_tool_batch
from tiny_harness.agent.turn import call_model, prepare_model_request_inputs
from tiny_harness.context.token_meter import DEFAULT_TOKEN_METER, TokenMeter
from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.context import CompactionConfig, prepare_context
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
)
from tiny_harness.runtime.hooks import ToolHooks, run_final_answer_hook
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
)
from tiny_harness.runtime.tool_trace import ToolTraceConfig
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.skills import SkillCatalog
from tiny_harness.runtime.shell_runner import DEFAULT_SHELL_RUNNER, ShellRunner
from tiny_harness.runtime.test_runner import TestRunner


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
            finalization = turn == context.max_turns
            context.token_meter.reconcile(messages, context.tools)
            prepared = prepare_context(
                messages,
                context.compactor,
                context.todo_manager.render(),
                active_request,
            )
            if prepared is not None:
                # 只有四层处理和最终验证全部成功后，才提交canonical history。
                messages[:] = prepared.messages

            request_messages, request_tools = prepare_model_request_inputs(
                messages,
                context,
                finalization=finalization,
            )
            context_tokens = context.token_meter.estimate_request(
                messages, request_messages, request_tools
            )
            hard_limit = context.max_context_tokens
            soft_limit = context.compactor.soft_limit if context.compactor else None
            context.event_logger.emit(
                EventType.CONTEXT_PREPARED,
                {
                    "turn": turn,
                    "context_tokens": context_tokens,
                    "soft_limit": soft_limit,
                    "hard_limit": hard_limit,
                    **({"finalization": True} if finalization else {}),
                    "pressure": (
                        context_tokens / hard_limit if hard_limit is not None else None
                    ),
                },
            )

            response = call_model(
                messages,
                context,
                finalization=finalization,
                prepared_request=(request_messages, request_tools),
            )
            if finalization and response.tool_calls:
                response = ModelResponse(
                    content=response.content,
                    reasoning_content=response.reasoning_content,
                    tool_calls=[],
                    finish_reason="stop",
                )
            assistant_message = assistant_message_from_response(response)

            if not response.tool_calls:
                answer = response.content or ""
                run_final_answer_hook(
                    context.final_answer_hook,
                    messages,
                    answer,
                    turn=turn,
                )

                messages.append(assistant_message)
                finish_data = {
                    "turns": turn,
                    "answer_length": len(answer),
                }
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

        raise RuntimeError("Agent loop ended without a final answer")
    except EventLogError:
        # Event Logger 自身失败时直接上抛，避免再次记录同一个故障。
        raise
    except Exception as error:
        context.event_logger.emit(
            EventType.RUN_FAILED,
            {
                "turn": context.current_turn,
                "error_type": type(error).__name__,
                "last_finish_reason": context.last_finish_reason,
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
    max_context_tokens: int | None = None,
    working_context_trigger_tokens: int = CompactionConfig.working_context_trigger_tokens,
    working_context_target_tokens: int = CompactionConfig.working_context_target_tokens,
    keep_recent_tool_batches: int = CompactionConfig.keep_recent_tool_batches,
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
    is_main_agent: bool = True,
    environment_adapter: EnvironmentAdapter | None = None,
    tool_trace: ToolTraceConfig = ToolTraceConfig(),
) -> str:
    """兼容配置入口：装配运行上下文后进入三参数核心循环。"""

    context = create_run_context(
        provider,
        workspace,
        max_turns=max_turns,
        permission_policy=permission_policy,
        permission_prompt=permission_prompt,
        event_logger=event_logger,
        max_context_tokens=max_context_tokens,
        working_context_trigger_tokens=working_context_trigger_tokens,
        working_context_target_tokens=working_context_target_tokens,
        keep_recent_tool_batches=keep_recent_tool_batches,
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
        is_main_agent=is_main_agent,
        environment_adapter=environment_adapter,
        tool_trace=tool_trace,
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
