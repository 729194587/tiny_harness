"""准备并执行一次可被 Agent Loop 接受的模型调用。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tiny_harness.agent.messages import ModelResponse, validate_model_response
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import model_context_messages
from tiny_harness.runtime.events import EventType
from tiny_harness.runtime.recovery import RecoveryState

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


FINALIZATION_INSTRUCTION = (
    "The execution turn budget is exhausted.\n"
    "No further tool use is available.\n"
    "Using the evidence already gathered, provide the best possible final answer "
    "to the user's request now.\n"
    "Briefly state any important limitation if the investigation is incomplete."
)


def _runtime_state(context: AgentRunContext, *, finalization: bool) -> str:
    remaining_turns = context.max_turns - context.current_turn
    return (
        "TinyHarness runtime state:\n"
        f"- current main-agent turn: {context.current_turn} / {context.max_turns}\n"
        f"- remaining main-agent turns: {remaining_turns}\n"
        f"- finalization: {str(finalization).lower()}\n"
        f"- tools available: {str(not finalization and bool(context.tools)).lower()}"
    )


def model_request_inputs(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
    *,
    finalization: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one request without committing runtime-only instructions to history."""

    request_messages = model_context_messages(messages)
    insert_at = 0
    while (
        insert_at < len(request_messages)
        and request_messages[insert_at].get("role") == "system"
    ):
        insert_at += 1
    if context.is_main_agent:
        request_messages.insert(
            insert_at,
            {
                "role": "system",
                "content": _runtime_state(context, finalization=finalization),
            },
        )
    if context.progress_tracker is not None:
        request_messages.insert(
            insert_at,
            {"role": "system", "content": context.progress_tracker.render(
                turn=context.current_turn, max_turns=context.max_turns,
            )},
        )
        insert_at += 1
    if finalization:
        request_messages.insert(
            insert_at + int(context.is_main_agent),
            {"role": "system", "content": FINALIZATION_INSTRUCTION},
        )
    if context.working_memory is not None:
        request_messages.append(context.working_memory.projection())
    return request_messages, ([] if finalization else context.tools)


def call_model(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
    *,
    finalization: bool = False,
) -> ModelResponse:
    """准备上下文、执行有界恢复并返回协议合法的模型响应。

    返回值尚未写入 canonical messages。调用方只有在该函数成功返回后，
    才能提交 assistant message 或执行工具。
    """

    request_messages, request_tools = model_request_inputs(
        messages,
        context,
        finalization=finalization,
    )

    recovery_state = RecoveryState()
    while True:
        try:
            response = context.recovery_executor.complete(
                context.provider,
                request_messages,
                request_tools,
                purpose="main",
                turn=context.current_turn,
                state=recovery_state,
                context_recovery_available=(
                    context.compactor is not None
                    and not recovery_state.reactive_compact_used
                ),
                request_metadata={
                    "max_turns": context.max_turns,
                    "remaining_turns": context.max_turns - context.current_turn,
                    "context_tokens": context.token_meter.estimate_request(
                        messages, request_messages, request_tools,
                    ),
                },
                finalization=finalization,
                tool_choice=(
                    "none" if finalization else ("auto" if request_tools else None)
                ),
            )
            break
        except ModelProviderError as error:
            if (
                error.kind is not ModelErrorKind.CONTEXT_LENGTH
                or context.compactor is None
                or recovery_state.reactive_compact_used
            ):
                raise

            # 先占用本次逻辑请求唯一的 reactive recovery 机会，避免摘要失败
            # 后递归触发第二次 reactive compact。
            recovery_state.reactive_compact_used = True
            failed_request_tokens = context.token_meter.estimate_request(
                messages, request_messages, request_tools,
            )
            prepared = context.compactor.reactive_compact(
                messages,
                context.todo_manager.render(),
                failed_request_tokens=failed_request_tokens,
            )
            messages[:] = prepared.messages
            request_messages, request_tools = model_request_inputs(
                messages,
                context,
                finalization=finalization,
            )
            context.event_logger.emit(
                EventType.MODEL_RETRY_SCHEDULED,
                {
                    "purpose": "main",
                    "turn": context.current_turn,
                    "attempt": recovery_state.attempt,
                    "delay_ms": 0,
                    "error_kind": error.kind.value,
                    "recovery": "reactive_compact",
                },
            )

    context.last_finish_reason = response.finish_reason
    validate_model_response(response)
    context.token_meter.observe(
        messages, request_messages, request_tools, response.prompt_tokens
    )
    return response
