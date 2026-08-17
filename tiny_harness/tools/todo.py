"""Planning tool backed by a run-scoped TodoManager."""

from tiny_harness.runtime.todos import TodoManager


def todo_write(manager: TodoManager, todos: list[object] | str) -> str:
    """Replace and display the current run's todo list."""

    output = manager.update(todos)
    print(f"\n## Current Tasks\n{output}")
    return output
