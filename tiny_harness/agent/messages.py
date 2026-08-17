"""Minimal messages exchanged by the model, agent loop, and tools."""

from dataclasses import dataclass


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments_json: str


@dataclass
class ToolResult:
    """Text returned for one model-requested tool invocation."""

    tool_call_id: str
    content: str


@dataclass
class ModelResponse:
    """The subset of a model response needed by the agent loop."""

    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str
