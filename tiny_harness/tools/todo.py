"""Planning tool backed by a run-scoped TodoManager."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def todo_write(manager: TodoManager, todos: list[object] | str) -> str:
    """Replace and display the current run's todo list."""

    output = manager.update(todos)
    display = (
        "暂无任务。"
        if output == "No todos."
        else re.sub(
            r"\((\d+)/(\d+) completed\)$",
            r"（已完成 \1/\2）",
            output,
        )
    )
    print(f"\n## 当前任务\n{display}")
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
            execute=lambda call, arguments: todo_write(manager, **arguments),
        ),
    )
