"""Planning tool backed by a run-scoped TodoManager."""

import re

from tiny_harness.runtime.todos import TodoManager


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
