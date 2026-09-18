"""Fresh-context execution for the synchronous task subagent."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from tiny_harness.context.token_meter import CalibratedTokenMeter
from tiny_harness.runtime.events import ScopedEventLogger, EventType, EventLogError

if TYPE_CHECKING:
    from tiny_harness.agent.context import RunConfig


class SubagentExecutor:
    """Run one delegated prompt through an isolated child Agent Loop."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def __call__(self, prompt: str, parent_tool_call_id: str) -> str:
        child_messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding subagent sharing the same workspace as "
                    "the parent agent. "
                    "Complete only the delegated task and return a concise "
                    "final answer. Use todo_write for multi-step work."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        child_logger = ScopedEventLogger(
            self.config.event_logger,
            {
                "agent_scope": "subagent",
                "parent_tool_call_id": parent_tool_call_id,
            },
        )
        child_logger.emit(EventType.SUBAGENT_STARTED)
        try:
            # Import lazily to avoid the context/loop composition cycle.
            from tiny_harness.agent.context import build_run_context
            from tiny_harness.agent.loop import agent_loop

            meter = self.config.token_meter
            child_config = replace(
                self.config,
                max_turns=self.config.subagent_max_turns,
                allow_subagent=False,
                memory_extraction_enabled=False,
                event_logger=child_logger,
                # Each child calibrates independently from the shared heuristic.
                token_meter=meter.heuristic if isinstance(meter, CalibratedTokenMeter) else meter,
            )
            answer = agent_loop(
                child_messages, build_run_context(child_config), prompt,
            )
        except EventLogError:
            raise
        except Exception as error:
            child_logger.emit(EventType.SUBAGENT_FAILED, {"error_type": type(error).__name__})
            raise
        child_logger.emit(EventType.SUBAGENT_FINISHED)
        return answer or "(no summary)"
