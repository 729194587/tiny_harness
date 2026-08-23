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
from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, EventLogger
from tiny_harness.runtime.goal import DEFAULT_MAX_GOAL_RETRIES
from tiny_harness.runtime.permissions import PermissionPrompt
from tiny_harness.runtime.recovery import RecoveryPolicy


RUN_SCOPED_MARKERS = frozenset(
    {
        "tinyharness_goal_state",
        "tinyharness_memory_catalog",
        "tinyharness_relevant_memory",
        "tinyharness_skill_catalog",
        "tinyharness_todo_state",
    }
)


def _null_event_logger() -> EventLogger:
    return NULL_EVENT_LOGGER


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
        max_context_chars: int | None = None,
        subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
        recovery_policy: RecoveryPolicy = RecoveryPolicy(),
        max_goal_retries: int = DEFAULT_MAX_GOAL_RETRIES,
        memory_enabled: bool = False,
        event_logger_factory: Callable[[], EventLogger] = _null_event_logger,
    ) -> None:
        self.provider = provider
        self.workspace = workspace.resolve()
        self.system_message: dict[str, Any] = {
            "role": "system",
            "content": system_prompt,
        }
        self.messages: list[dict[str, Any]] = [
            copy.deepcopy(self.system_message)
        ]
        self.max_turns = max_turns
        self.permission_prompt = permission_prompt
        self.max_context_chars = max_context_chars
        self.subagent_max_turns = subagent_max_turns
        self.recovery_policy = recovery_policy
        self.max_goal_retries = max_goal_retries
        self.memory_enabled = memory_enabled
        self.event_logger_factory = event_logger_factory

    def submit(
        self,
        task: str,
        *,
        goal_condition: str | None = None,
    ) -> str:
        """Append one user turn and run the agent against shared history."""

        if not task.strip():
            raise ValueError("task cannot be empty")
        if goal_condition is not None and any(
            message.get("role") == "user" for message in self.messages
        ):
            raise ValueError(
                "goal_condition is only supported on the first submit "
                "of a fresh AgentSession"
            )
        self._remove_run_scoped_markers()
        self.messages.append({"role": "user", "content": task})
        context = create_run_context(
            self.provider,
            self.workspace,
            max_turns=self.max_turns,
            permission_prompt=self.permission_prompt,
            event_logger=self.event_logger_factory(),
            max_context_chars=self.max_context_chars,
            subagent_max_turns=self.subagent_max_turns,
            recovery_policy=self.recovery_policy,
            goal_condition=goal_condition,
            max_goal_retries=self.max_goal_retries,
            memory_enabled=self.memory_enabled,
        )
        answer = agent_loop(self.messages, context, task)
        self._remove_run_scoped_markers()
        return answer

    def clear(self) -> None:
        """Forget conversation messages without deleting workspace artifacts."""

        self.messages[:] = [copy.deepcopy(self.system_message)]

    def _remove_run_scoped_markers(self) -> None:
        self.messages[:] = [
            message
            for message in self.messages
            if message.get("name") not in RUN_SCOPED_MARKERS
        ]
