"""A self-contained model-facing and executable Tool definition."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tiny_harness.agent.messages import ToolCall


ToolExecutor = Callable[[ToolCall, dict[str, Any]], str]
TraceMetadata = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, init=False)
class ToolDefinition:
    """Own one Tool's model metadata and its run-bound executor."""

    name: str
    description: str
    _parameters_json: str = field(repr=False)
    execute: ToolExecutor
    trace_metadata: TraceMetadata = field(compare=False, repr=False)

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execute: ToolExecutor,
        trace_metadata: TraceMetadata | None = None,
    ) -> None:
        encoded_parameters = json.dumps(
            parameters,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        decoded_parameters = json.loads(encoded_parameters)
        if not isinstance(decoded_parameters, dict):
            raise TypeError("Tool parameters must be a JSON object")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "_parameters_json", encoded_parameters)
        object.__setattr__(self, "execute", execute)
        object.__setattr__(self, "trace_metadata", trace_metadata or (lambda arguments: {}))

    @property
    def parameters(self) -> dict[str, Any]:
        """Return an isolated mutable projection of the immutable schema."""

        return json.loads(self._parameters_json)

    def model_schema(self) -> dict[str, Any]:
        """Project this definition into Chat Completions Tool format."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
