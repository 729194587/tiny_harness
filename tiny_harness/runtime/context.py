"""Deterministic context preparation for bounded model requests."""

import copy
import json
from dataclasses import dataclass
from typing import Any


class ContextError(RuntimeError):
    """Base error raised while preparing a model context."""


class ContextProtocolError(ContextError):
    """Raised when assistant tool calls and tool results are not paired."""


class ContextLimitError(ContextError):
    """Raised when required context cannot fit within the configured budget."""


@dataclass(frozen=True)
class PreparedContext:
    """A protocol-safe model context and its deterministic size metadata."""

    messages: list[dict[str, Any]]
    before_chars: int
    after_chars: int
    dropped_blocks: int
    dropped_messages: int


def context_char_count(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> int:
    """Count characters in the compact JSON request context."""

    try:
        serialized = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ContextProtocolError("Context must be JSON serializable") from error
    return len(serialized)


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ContextProtocolError(
            "Assistant tool_calls must be a non-empty list"
        )

    call_ids: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ContextProtocolError("Each assistant tool call must be an object")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ContextProtocolError("Each assistant tool call must have an ID")
        call_ids.append(call_id)

    if len(call_ids) != len(set(call_ids)):
        raise ContextProtocolError("Assistant tool call IDs must be unique")
    return call_ids


def _split_context(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Split the pinned task prefix from complete assistant/tool blocks."""

    prefix: list[dict[str, Any]] = []
    blocks: list[list[dict[str, Any]]] = []
    index = 0

    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            raise ContextProtocolError("Each context message must be an object")
        if message.get("role") == "tool":
            raise ContextProtocolError("Tool result has no preceding tool call")
        if message.get("role") == "assistant" and message.get("tool_calls"):
            break
        prefix.append(message)
        index += 1

    while index < len(messages):
        assistant = messages[index]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise ContextProtocolError(
                "Expected an assistant tool-call message after tool history"
            )
        call_ids = _tool_call_ids(assistant)
        block = [assistant]
        index += 1

        result_ids: list[str] = []
        while index < len(messages):
            result = messages[index]
            if not isinstance(result, dict):
                raise ContextProtocolError("Each context message must be an object")
            if result.get("role") != "tool":
                break
            result_id = result.get("tool_call_id")
            if not isinstance(result_id, str) or not result_id:
                raise ContextProtocolError("Each tool result must have a tool call ID")
            result_ids.append(result_id)
            block.append(result)
            index += 1

        if result_ids != call_ids:
            raise ContextProtocolError(
                "Assistant tool calls and tool results must match in order"
            )
        blocks.append(block)

    return prefix, blocks


def prepare_context(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_chars: int,
) -> PreparedContext:
    """Build a bounded context without splitting assistant/tool blocks."""

    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    prefix, blocks = _split_context(messages)
    before_chars = context_char_count(messages, tools)
    kept_blocks = list(blocks)
    dropped_messages = 0

    while len(kept_blocks) > 1:
        candidate = prefix + [message for block in kept_blocks for message in block]
        if context_char_count(candidate, tools) <= max_chars:
            break
        removed = kept_blocks.pop(0)
        dropped_messages += len(removed)

    prepared_messages = prefix + [
        message for block in kept_blocks for message in block
    ]
    after_chars = context_char_count(prepared_messages, tools)
    if after_chars > max_chars:
        raise ContextLimitError(
            "Required context exceeds configured character budget: "
            f"{after_chars} > {max_chars}"
        )

    return PreparedContext(
        messages=copy.deepcopy(prepared_messages),
        before_chars=before_chars,
        after_chars=after_chars,
        dropped_blocks=len(blocks) - len(kept_blocks),
        dropped_messages=dropped_messages,
    )
