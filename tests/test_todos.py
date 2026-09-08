import contextlib
import io
import unittest

from tiny_harness.runtime.todos import TodoItem, TodoManager
from tiny_harness.tools.todo import todo_write


class TodoManagerTest(unittest.TestCase):
    def test_updates_and_renders_structured_todos(self) -> None:
        manager = TodoManager()

        output = manager.update(
            [
                {"content": "Inspect files", "status": "completed"},
                {"content": "Implement change", "status": "in_progress"},
                {"content": "Run tests", "status": "pending"},
            ]
        )

        self.assertEqual(
            manager.items,
            [
                TodoItem("Inspect files", "completed"),
                TodoItem("Implement change", "in_progress"),
                TodoItem("Run tests", "pending"),
            ],
        )
        self.assertEqual(manager.revision, 1)
        self.assertEqual(
            output,
            "[x] Inspect files\n"
            "[>] Implement change\n"
            "[ ] Run tests\n\n"
            "(1/3 completed)",
        )

    def test_accepts_json_and_python_list_strings_without_eval(self) -> None:
        manager = TodoManager()

        json_output = manager.update(
            '[{"content":"JSON task","status":"pending"}]'
        )
        literal_output = manager.update(
            "[{'content': 'Literal task', 'status': 'completed'}]"
        )

        self.assertIn("[ ] JSON task", json_output)
        self.assertIn("[x] Literal task", literal_output)
        self.assertEqual(manager.revision, 2)

    def test_empty_list_clears_todos(self) -> None:
        manager = TodoManager()
        manager.update([{"content": "Task", "status": "pending"}])

        output = manager.update([])

        self.assertEqual(output, "No todos.")
        self.assertEqual(manager.items, [])
        self.assertEqual(manager.revision, 2)

    def test_rejects_invalid_todo_lists(self) -> None:
        invalid_values = [
            ({"content": "not a list"}, "todos must be a list"),
            (["not an object"], r"todos\[0\] must be an object"),
            (
                [{"content": None, "status": "pending"}],
                r"todos\[0\]\.content must be a string",
            ),
            ([{"content": "", "status": "pending"}], "requires content"),
            ([{"content": "Task", "status": "unknown"}], "invalid status"),
            (
                [
                    {"content": "First", "status": "in_progress"},
                    {"content": "Second", "status": "in_progress"},
                ],
                "Only one todo",
            ),
            (
                [
                    {"content": f"Task {index}", "status": "pending"}
                    for index in range(21)
                ],
                "Max 20",
            ),
            ("not valid data", "todos must be a list or JSON array string"),
        ]

        for value, message in invalid_values:
            with self.subTest(value=value):
                manager = TodoManager()
                with self.assertRaisesRegex(ValueError, message):
                    manager.update(value)
                self.assertEqual(manager.items, [])
                self.assertEqual(manager.revision, 0)

    def test_invalid_update_preserves_previous_state(self) -> None:
        manager = TodoManager()
        manager.update([{"content": "Keep me", "status": "pending"}])

        with self.assertRaises(ValueError):
            manager.update([{"content": "", "status": "pending"}])

        self.assertEqual(manager.items, [TodoItem("Keep me", "pending")])
        self.assertEqual(manager.revision, 1)

    def test_tool_returns_current_tasks_without_printing(self) -> None:
        manager = TodoManager()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            output = todo_write(
                manager,
                [{"content": "Visible task", "status": "in_progress"}],
            )

        self.assertEqual(output, "[>] Visible task\n\n(0/1 completed)")
        self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
