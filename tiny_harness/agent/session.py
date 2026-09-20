"""In-process conversation state for repeated agent submissions."""

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tiny_harness.agent.context import (
    DEFAULT_SUBAGENT_MAX_TURNS,
    create_run_context,
)
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.turn import NEAR_BUDGET_MARKER
from tiny_harness.context.token_meter import DEFAULT_TOKEN_METER, TokenMeter, CalibratedTokenMeter
from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, EventLogger
from tiny_harness.runtime.permissions import PermissionPrompt
from tiny_harness.runtime.tool_trace import ToolTraceConfig
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.skills import discover_skills


RUN_SCOPED_MARKERS = frozenset(
    {
        NEAR_BUDGET_MARKER,
        "tinyharness_memory_catalog",
        "tinyharness_relevant_memory",
        "tinyharness_skill_catalog",
        "tinyharness_todo_state",
    }
)


def _null_event_logger() -> EventLogger:
    return NULL_EVENT_LOGGER


class SessionFailedError(RuntimeError):
    """The previous submission did not finish; explicit clear is required."""


class AgentSession:
    """Keep canonical messages across successful in-process submissions."""

    def __init__(
        self,
        provider: ModelProvider,
        workspace: Path,
        system_prompt: str,
        *,
        max_turns: int = 20,
        permission_prompt: PermissionPrompt | None = None,
        max_context_tokens: int | None = None,
        token_meter: TokenMeter = DEFAULT_TOKEN_METER,
        subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
        recovery_policy: RecoveryPolicy = RecoveryPolicy(),
        memory_enabled: bool = False,
        event_logger_factory: Callable[[], EventLogger] = _null_event_logger,
        tool_trace: ToolTraceConfig = ToolTraceConfig(),
    ) -> None:
        self.provider = provider
        self.tool_trace = tool_trace
        self.workspace = workspace.resolve()
        self.skill_catalog = discover_skills(self.workspace)
        self.system_message: dict[str, Any] = {
            "role": "system",
            "content": system_prompt,
        }
        self.messages: list[dict[str, Any]] = [
            copy.deepcopy(self.system_message)
        ]
        self.max_turns = max_turns
        self.permission_prompt = permission_prompt
        self.max_context_tokens = max_context_tokens
        self.token_meter = CalibratedTokenMeter(token_meter)
        self.subagent_max_turns = subagent_max_turns
        self.recovery_policy = recovery_policy
        self.memory_enabled = memory_enabled
        self.event_logger_factory = event_logger_factory
        self._failed = False

    @property
    def failed(self) -> bool:
        """Whether a submission did not finish and clear() is required."""

        return self._failed

    def submit(self, task: str) -> str:
        """Append one user turn and run the agent against shared history."""

        if self._failed:
            raise SessionFailedError(
                "Session is failed after an interrupted submission. Tool side "
                "effects may remain and history may be incomplete. Inspect the "
                "workspace and call clear() before submitting again; clear() "
                "does not undo tool side effects."
            )
        if not task.strip():
            raise ValueError("task cannot be empty")
        # Pessimistic until success: exceptions and interrupts propagate unchanged,
        # retaining history without replaying or pretending to undo side effects.
        self._failed = True
        self._remove_run_scoped_markers()
        self.messages.append({"role": "user", "content": task})
        context = create_run_context(
            self.provider,
            self.workspace,
            max_turns=self.max_turns,
            permission_prompt=self.permission_prompt,
            event_logger=self.event_logger_factory(),
            max_context_tokens=self.max_context_tokens,
            token_meter=self.token_meter,
            subagent_max_turns=self.subagent_max_turns,
            recovery_policy=self.recovery_policy,
            skill_catalog=self.skill_catalog,
            memory_enabled=self.memory_enabled,
            tool_trace=self.tool_trace,
        )
        answer = agent_loop(self.messages, context, task)
        self._remove_run_scoped_markers()
        self._failed = False
        return answer

    def clear(self) -> None:
        """Forget conversation messages without deleting workspace artifacts."""

        self.token_meter.reset()
        self.messages[:] = [copy.deepcopy(self.system_message)]
        self._failed = False

    def _remove_run_scoped_markers(self) -> None:
        before = len(self.messages)
        self.messages[:] = [
            message
            for message in self.messages
            if message.get("name") not in RUN_SCOPED_MARKERS
        ]
        if len(self.messages) != before:
            self.token_meter.reset()
