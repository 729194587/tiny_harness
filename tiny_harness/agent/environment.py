"""Optional environment contributions to runtime composition."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


ENVIRONMENT_CONTEXT_MARKER = "tinyharness_environment_context"


class EnvironmentAdapter(Protocol):
    """Provide additive tools and factual context, without executing a run.

    Factories bind tools to the supplied run's capabilities. They must not
    mutate runtime state or register tools themselves. Return an empty iterable
    or empty string when there is no contribution. Instances may be reused by
    successive runs and child agents; do not retain mutable run state.
    """

    def build_tools(self, context: AgentRunContext) -> Iterable[ToolDefinition]:
        """Return additional definitions; existing names cannot be replaced."""
        ...

    def initial_context(self, context: AgentRunContext) -> str:
        """Return environment information, not higher-priority instructions."""
        ...
