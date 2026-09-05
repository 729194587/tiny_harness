from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tiny_harness.tools.search import glob_files, grep_text


class SearchToolTest(unittest.TestCase):
    def test_glob_finds_matching_files_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)

            (workspace / "b.py").write_text("b", encoding="utf-8")
            (workspace / "a.py").write_text("a", encoding="utf-8")
            (workspace / "notes.txt").write_text("notes", encoding="utf-8")

            result = glob_files(workspace, "*.py")

        self.assertEqual(result.splitlines(), ["a.py", "b.py"])

    def test_glob_rejects_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)

            with self.assertRaises(ValueError):
                glob_files(workspace, "*.py", path="..")

    def test_grep_returns_file_line_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)

            (workspace / "example.py").write_text(
                "first line\nMemoryRuntime here\nlast line\n",
                encoding="utf-8",
            )

            result = grep_text(
                workspace,
                "MemoryRuntime",
                include="**/*.py",
            )

        self.assertEqual(
            result,
            "example.py:2:MemoryRuntime here",
        )

    def test_grep_supports_case_insensitive_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)

            (workspace / "example.py").write_text(
                "MemoryRuntime\n",
                encoding="utf-8",
            )

            result = grep_text(
                workspace,
                "memoryruntime",
                include="**/*.py",
                case_sensitive=False,
            )

        self.assertEqual(
            result,
            "example.py:1:MemoryRuntime",
        )

    def test_grep_exact_result_limit_is_not_marked_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)

            (workspace / "example.py").write_text(
                "needle\n",
                encoding="utf-8",
            )

            result = grep_text(
                workspace,
                "needle",
                max_results=1,
            )

        self.assertEqual(
            result,
            "example.py:1:needle",
        )

    def test_grep_marks_results_truncated_only_when_more_matches_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)

            (workspace / "example.py").write_text(
                "needle\nneedle\n",
                encoding="utf-8",
            )

            result = grep_text(
                workspace,
                "needle",
                max_results=1,
            )

        self.assertEqual(
            result.splitlines(),
            [
                "example.py:1:needle",
                "... truncated after 1 results",
            ],
        )


if __name__ == "__main__":
    unittest.main()