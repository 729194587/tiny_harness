"""Run-local facts observed from tool lifecycle events, without judgments.

Started counts describe handler entry, not subprocess launches. A returned
handler is not evidence of a passing test or a successful command exit code.
Workspace changes are unknown: existing events provide no filesystem baseline.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tiny_harness.runtime.events import EventType


@dataclass
class ExecutionState:
    tool_calls: int = 0
    tool_succeeded: int = 0
    tool_failed: int = 0
    tool_denied: int = 0
    commands_started: int = 0
    tests_started: int = 0


class ProgressTracker:
    """Observe only this run's event stream; rendering has no side effects."""

    def __init__(self) -> None:
        self.state = ExecutionState()

    def emit(
        self, event_type: EventType, data: Mapping[str, Any] | None = None,
    ) -> None:
        data = data or {}
        state = self.state
        if event_type == EventType.TOOL_CALLED:
            state.tool_calls += 1
        elif event_type == EventType.TOOL_STARTED:
            if data.get("tool_name") in {"bash", "git_status", "git_diff"}:
                state.commands_started += 1
            elif data.get("tool_name") == "run_tests":
                state.tests_started += 1
        elif event_type == EventType.TOOL_RESULT:
            outcome = data.get("outcome")
            if outcome == "returned":
                state.tool_succeeded += 1
            elif outcome == "error":
                state.tool_failed += 1
            elif outcome in {"permission_denied", "hook_blocked"}:
                state.tool_denied += 1

    def render(self, *, turn: int, max_turns: int) -> str:
        state = self.state
        return (
            "Execution state (runtime observation):\n"
            f"- Turn: {turn}/{max_turns}\n"
            f"- Tool calls: {state.tool_calls}\n"
            f"- Tool outcomes: {state.tool_succeeded} returned, "
            f"{state.tool_failed} errors, {state.tool_denied} denied/blocked\n"
            f"- Command tools started: {state.commands_started}\n"
            f"- Explicit test tools started: {state.tests_started}\n"
            "- Workspace changed: unknown\n"
            "Counts reflect tool lifecycle events; returned does not imply "
            "command or test success."
        )
