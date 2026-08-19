"""准备并执行一次可被 Agent Loop 接受的模型调用。"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from tiny_harness.agent.messages import ModelResponse, validate_model_response
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import context_char_count
from tiny_harness.runtime.events import EventType
from tiny_harness.runtime.recovery import RecoveryState

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def call_model(
    messages: list[dict[str, Any]],
    context: AgentRunContext,
) -> ModelResponse:
    """准备上下文、执行有界恢复并返回协议合法的模型响应。

    返回值尚未写入 canonical messages。调用方只有在该函数成功返回后，
    才能提交 assistant message 或执行工具。
    """

    request_messages = copy.deepcopy(messages)

    recovery_state = RecoveryState()
    while True:
        try:
            response = context.recovery_executor.complete(
                context.provider,
                request_messages,
                context.tools,
                purpose="main",
                turn=context.current_turn,
                state=recovery_state,
                context_recovery_available=(
                    context.compactor is not None
                    and not recovery_state.reactive_compact_used
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
            failed_request_chars = context_char_count(
                request_messages,
                context.tools,
            )
            prepared = context.compactor.reactive_compact(
                messages,
                context.todo_manager.render(),
                failed_request_chars=failed_request_chars,
            )
            messages[:] = prepared.messages
            request_messages = copy.deepcopy(messages)
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

    validate_model_response(response)
    return response
