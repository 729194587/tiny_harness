"""Run-scoped todo state for the coding agent."""

import ast
import json
from dataclasses import dataclass
from typing import Literal, cast

TodoStatus = Literal["pending", "in_progress", "completed"]


@dataclass(frozen=True)
class TodoItem:
    """One validated planning item."""

    content: str
    status: TodoStatus


class TodoManager:
    """Validate, replace, and render the current run's todo list."""

    def __init__(self) -> None:
        self.items: list[TodoItem] = []

    def update(self, todos: list[object] | str) -> str:
        """Atomically replace the todo list and return its rendered state."""

        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                try:
                    todos = ast.literal_eval(todos)
                except (SyntaxError, ValueError) as error:
                    raise ValueError(
                        "todos must be a list or JSON array string"
                    ) from error

        if not isinstance(todos, list):
            raise ValueError("todos must be a list")
        if len(todos) > 20:
            raise ValueError("Max 20 todos allowed")

        validated: list[TodoItem] = []
        in_progress_count = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] must be an object")

            raw_content = todo.get("content", "")
            if not isinstance(raw_content, str):
                raise ValueError(f"todos[{index}].content must be a string")
            content = raw_content.strip()
            status = str(todo.get("status", "pending")).lower()
            if not content:
                raise ValueError(f"todos[{index}] requires content")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(
                    f"todos[{index}] has invalid status '{status}'"
                )
            if status == "in_progress":
                in_progress_count += 1
            validated.append(
                TodoItem(content=content, status=cast(TodoStatus, status))
            )

        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """Render the current list in the compact s05-style format."""

        if not self.items:
            return "No todos."

        markers = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }
        lines = [
            f"{markers[item.status]} {item.content}"
            for item in self.items
        ]
        completed = sum(item.status == "completed" for item in self.items)
        lines.append(f"\n({completed}/{len(self.items)} completed)")
        return "\n".join(lines)
