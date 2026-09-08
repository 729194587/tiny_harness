"""Planning tool backed by a run-scoped TodoManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_harness.runtime.events import NULL_EVENT_LOGGER, EventLogger, EventType
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def todo_write(
    manager: TodoManager, todos: list[object] | str,
    *, event_logger: EventLogger = NULL_EVENT_LOGGER,
) -> str:
    """Replace todos and report counts without exposing task contents."""

    output = manager.update(todos)
    event_logger.emit(EventType.TODO_UPDATED, {
        "total": len(manager.items),
        "completed": sum(item.status == "completed" for item in manager.items),
        "in_progress": sum(item.status == "in_progress" for item in manager.items),
    })
    return output


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    """Bind todo_write to this run's TodoManager."""

    manager = context.todo_manager
    if manager is None:
        return ()
    return (
        ToolDefinition(
            name="todo_write",
            description="Create and manage a task list for the current coding run.",
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "minLength": 1},
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                    ],
                                },
                            },
                            "required": ["content", "status"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["todos"],
                "additionalProperties": False,
            },
            execute=lambda call, arguments: todo_write(manager, event_logger=context.event_logger, **arguments),
        ),
    )
