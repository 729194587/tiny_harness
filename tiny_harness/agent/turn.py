"""准备并执行一次可被 Agent Loop 接受的模型调用。"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from tiny_harness.agent.messages import ModelProtocolError, ModelResponse, validate_model_response
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import model_context_messages
from tiny_harness.runtime.events import EventLogError, EventType
from tiny_harness.runtime.recovery import RecoveryState

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


TOOL_USE_EFFICIENCY_GUIDANCE = (
    "Tool-use efficiency: When multiple read-only investigation actions are "
    "independent, their arguments are already known, and no action needs another "
    "call's result to determine its arguments, prefer issuing multiple tool_calls "
    "in the same assistant response. Analyze the results together after all calls "
    "return. If a later action depends on an earlier result, use separate turns. "
    "Do not expand the investigation scope or add low-value calls just to form a "
    "batch. Choose the tools and number of calls according to the task."
)


NEAR_BUDGET_NORMAL_TURNS = 3
WORKING_CHECKPOINT_MIN_REMAINING_TURNS = 3
NEAR_BUDGET_MARKER = "tinyharness_near_budget"
NEAR_BUDGET_INSTRUCTION = (
    "The execution budget is nearly exhausted.\n"
    "Prioritize completing the user's request with the evidence already gathered.\n"
    "Before using another tool, consider whether the missing information would "
    "materially change the answer. Do not broaden the investigation unless necessary."
)


FINALIZATION_INSTRUCTION = (
    "The execution turn budget is exhausted.\n"
    "The tool-use phase has ended.\n"
    "Answer the user's original request now using only the evidence already gathered.\n"
    "Do not request, invoke, or describe additional tool calls.\n"
    "Do not emit tool-call syntax or protocol markup.\n"
    "If some detail remains unverified, state that limitation explicitly instead "
    "of continuing investigation."
)

FINALIZATION_FAILURE = (
    "工具使用回合预算已耗尽，模型未能生成有效的最终回答。"
    "本次任务已停止，尚未确认的结果无法验证。"
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
    if not finalization and context.tools:
        request_messages.insert(
            insert_at,
            {"role": "system", "content": TOOL_USE_EFFICIENCY_GUIDANCE},
        )
        insert_at += 1
    if finalization:
        request_messages.append(
            {"role": "system", "content": FINALIZATION_INSTRUCTION},
        )
    return request_messages, ([] if finalization else context.tools)


def prepare_model_request_inputs(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
    *,
    finalization: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply working pressure once, after all request-only projections."""

    request_messages, request_tools = model_request_inputs(
        messages, context, finalization=finalization,
    )
    if context.compactor is not None:
        before_tokens = context.token_meter.estimate_request(messages, request_messages, request_tools)
        if before_tokens >= context.compactor.config.working_context_trigger_tokens:
            remaining_turns = context.max_turns - context.current_turn
            if remaining_turns <= WORKING_CHECKPOINT_MIN_REMAINING_TURNS:
                context.event_logger.emit(
                    EventType.CONTEXT_COMPACTION_SKIPPED,
                    {
                        "reason": "working",
                        "skip_reason": "insufficient_remaining_execution_horizon",
                        "turn": context.current_turn,
                        "remaining_turns": remaining_turns,
                        "context_tokens": before_tokens,
                    },
                )
                return request_messages, request_tools

            def measure_compacted(candidate: list[dict[str, Any]]) -> int:
                projected, tools = model_request_inputs(
                    candidate, context, finalization=finalization,
                )
                return context.token_meter.estimate(projected, tools)

            try:
                prepared = context.compactor.compact_history(
                    messages, context.todo_manager.render(), reason="working",
                    max_tokens=context.compactor.config.working_context_target_tokens,
                    recent_tail_budget=max(
                        1, context.compactor.config.working_context_target_tokens // 3,
                    ),
                    summary_source_messages=copy.deepcopy(messages),
                    summary_request_messages=request_messages,
                    summary_request_tools=request_tools,
                    turn=context.current_turn,
                    before_tokens=before_tokens,
                    measure_compacted=measure_compacted,
                )
            except EventLogError:
                raise
            except Exception:
                pass
            else:
                messages[:] = prepared.messages
                request_messages, request_tools = model_request_inputs(
                    messages, context, finalization=finalization,
                )
                context.token_meter.estimate_request(messages, request_messages, request_tools)
    return request_messages, request_tools


def call_model(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
    *,
    finalization: bool = False,
    prepared_request: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> ModelResponse:
    """准备上下文、执行有界恢复并返回协议合法的模型响应。

    返回值尚未写入 canonical messages。调用方只有在该函数成功返回后，
    才能提交 assistant message 或执行工具。
    """

    request_messages, request_tools = (
        prepared_request if prepared_request is not None
        else prepare_model_request_inputs(messages, context, finalization=finalization)
    )

    recovery_state = RecoveryState()
    finalization_retries = 0
    while True:
        try:
            response = context.recovery_executor.complete(
                context.provider,
                request_messages,
                request_tools,
                purpose="main",
                token_meter=context.token_meter,
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
            continue

        context.last_finish_reason = response.finish_reason
        validate_model_response(response)
        context.token_meter.observe(
            messages, request_messages, request_tools, response.prompt_tokens
        )
        if finalization and (
            response.tool_calls or response.contains_tool_protocol
            or not (response.content or "").strip()
        ):
            if finalization_retries == 1:
                return ModelResponse(FINALIZATION_FAILURE, None, [], "stop")
            finalization_retries += 1
            context.event_logger.emit(
                EventType.MODEL_RETRY_SCHEDULED,
                {
                    "purpose": "main", "turn": context.current_turn,
                    "attempt": recovery_state.attempt, "delay_ms": 0,
                    "error_kind": "invalid_final_answer",
                    "recovery": "finalization",
                },
            )
            continue
        if response.contains_tool_protocol:
            raise ModelProtocolError("Model response contains tool protocol markup")
        return response
