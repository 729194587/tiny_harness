import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiny_harness.agent.messages import ToolCall
from tiny_harness.runtime.hooks import HookBlock, ToolHooks
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.discovery import discover_tools
from tiny_harness.tools.filesystem import read_file, write_file, edit_file, list_files
from tiny_harness.tools.search import (
    search_code, grep_text, glob_files, MAX_SEARCH_OUTPUT_CHARS,
    MAX_SEARCH_FILE_BYTES,
)
from tiny_harness.tools.registry import dispatch


class ToolErgonomicsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.file = self.workspace / "a.py"
        self.file.write_bytes(b"first\r\nneedle.*\r\nlast")

    def test_ranged_read(self):
        for start, end, expected in (
            (None, None, "first\nneedle.*\nlast"),
            (2, 2, "needle.*\n"), (2, None, "needle.*\nlast"),
            (None, 2, "first\nneedle.*\n"), (2, 100, "needle.*\nlast"),
            (10, None, ""), (10, 20, ""), (3, 3, "last"),
            (10**30, None, ""), (3, 10**30, "last"),
        ):
            with self.subTest(start=start, end=end):
                self.assertEqual(read_file(self.workspace, "a.py", start, end), expected)
        self.file.write_text("")
        self.assertEqual(read_file(self.workspace, "a.py", 1, 3), "")

    def test_invalid_ranges(self):
        for start, end in ((0, None), (-1, 2), (3, 2), (None, 0),
                           (True, None), (None, False), (1.5, 3), (1, "2")):
            with self.subTest(start=start, end=end):
                with self.assertRaises((ValueError, TypeError)):
                    read_file(self.workspace, "a.py", start, end)

    def test_absolute_paths_for_file_tools(self):
        path = str(self.file)
        self.assertIn("needle", read_file(self.workspace, path))
        write_file(self.workspace, path, "old")
        edit_file(self.workspace, path, "old", "new")
        self.assertEqual(read_file(self.workspace, path), "new")
        self.assertEqual(list_files(self.workspace, str(self.workspace)), "a.py")

    def test_search_literal_context_and_scope(self):
        sub = self.workspace / "src"
        sub.mkdir()
        (sub / "b.py").write_text("needle.*\n", encoding="utf-8")
        self.assertEqual(search_code(self.workspace, "needle.*", "a.py"),
                         "a.py-1-first\na.py:2:needle.*\na.py-3-last")
        self.assertEqual(search_code(self.workspace, "needle.*", str(sub), context_lines=0),
                         "src/b.py:1:needle.*")
        self.assertEqual(search_code(self.workspace, "needle.*", str(self.file), context_lines=0),
                         "a.py:2:needle.*")
        self.assertEqual(search_code(self.workspace, "NEEDLE"), "(no matches)")
        self.assertEqual(search_code(self.workspace, "needle.+"), "(no matches)")
        self.assertIn("src/b.py:1:", search_code(self.workspace, "needle.*"))

    def test_search_bounds(self):
        self.file.write_text("needle\n" * 25)
        result = search_code(self.workspace, "needle", max_results=2, context_lines=0)
        self.assertEqual(result, "a.py:1:needle\n--\na.py:2:needle\n... truncated after 2 results")
        self.assertEqual(search_code(self.workspace, "needle", max_results=25, context_lines=0).count(":needle"), 25)
        self.assertNotIn("truncated", search_code(self.workspace, "needle", max_results=25))
        self.assertEqual(search_code(self.workspace, "needle", context_lines=0).count(":needle"), 20)
        self.file.write_text(("needle" + "x" * 1000 + "\n") * 500)
        result = search_code(self.workspace, "needle", max_results=500, context_lines=10)
        self.assertLessEqual(len(result), MAX_SEARCH_OUTPUT_CHARS)
        self.assertIn("output character limit", result)
        self.assertNotIn("x" * 501, result)

    def test_search_skips_binary_git_large_and_unreadable(self):
        for name, data in (("nul", b"needle\x00"), ("invalid", b"needle\xff"),
                           ("large", b"needle" + b"x" * MAX_SEARCH_FILE_BYTES)):
            (self.workspace / name).write_bytes(data)
        git = self.workspace / ".git"
        git.mkdir()
        (git / "config").write_text("needle")
        self.assertEqual(search_code(self.workspace, "needle", context_lines=0), "a.py:2:needle.*")
        self.assertEqual(search_code(self.workspace, "needle", ".git/config"), "(no matches)")
        self.assertEqual(search_code(self.workspace, "needle", ".git"), "(no matches)")
        with patch.object(Path, "open", side_effect=PermissionError("unreadable")):
            self.assertEqual(search_code(self.workspace, "needle"), "(no matches)")

    def test_invalid_search_arguments(self):
        for arguments in ({"query": ""}, {"query": "a\nb"}, {"query": 3},
                          {"max_results": 0}, {"max_results": 501}, {"max_results": True},
                          {"context_lines": -1}, {"context_lines": 11},
                          {"context_lines": False}, {"context_lines": 1.5}, {"path": "missing"}):
            with self.subTest(arguments=arguments):
                with self.assertRaises((ValueError, TypeError)):
                    search_code(self.workspace, **({"query": "needle"} | arguments))

    def test_escape_paths(self):
        outside = self.root / "outside.txt"
        outside.write_text("needle")
        for path in (str(outside), "../outside.txt", str(self.workspace / ".." / "outside.txt")):
            for operation in (
                lambda: read_file(self.workspace, path, 1, 1),
                lambda: write_file(self.workspace, path, "changed"),
                lambda: edit_file(self.workspace, path, "needle", "changed"),
                lambda: list_files(self.workspace, path),
                lambda: search_code(self.workspace, "needle", path),
                lambda: grep_text(self.workspace, "needle", path),
                lambda: glob_files(self.workspace, "*", path),
            ):
                with self.subTest(path=path, operation=operation):
                    with self.assertRaisesRegex(ValueError, "escapes workspace"):
                        operation()
        self.assertEqual(outside.read_text(), "needle")

    def test_symlink_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "secret").write_text("needle")
        link = self.workspace / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "escapes workspace"):
            search_code(self.workspace, "needle", "link")
        with self.assertRaisesRegex(ValueError, "escapes workspace"):
            read_file(self.workspace, "link/secret", 1)
        self.assertNotIn("secret", search_code(self.workspace, "needle"))

    def test_discovery_schemas_dispatch_and_boundaries(self):
        registry = discover_tools(SimpleNamespace(
            workspace=self.workspace, todo_manager=TodoManager(), subagent_runner=None,
            skill_catalog=None, compaction_request=None, test_runner=None,
        ))
        schemas = {s["function"]["name"]: s["function"]["parameters"]
                   for s in registry.model_schemas()}
        self.assertEqual(set(schemas["read_file"]["properties"]), {"path", "start_line", "end_line"})
        self.assertEqual(set(schemas["search_code"]["properties"]),
                         {"query", "path", "max_results", "context_lines"})
        call = ToolCall("search", "search_code", json.dumps({"query": "needle"}))
        self.assertIn("a.py:2:needle", dispatch(registry, call).content)
        read = ToolCall("read", "read_file", json.dumps({"path": str(self.file), "start_line": 2, "end_line": 2}))
        self.assertEqual(dispatch(registry, read).content, "needle.*\n")
        bad = ToolCall("bad", "search_code", '{"query":"needle","max_results":0}')
        self.assertTrue(dispatch(registry, bad).content.startswith("Error:"))
        hooks = ToolHooks()
        hooks.register_pre(lambda context: HookBlock("blocked"))
        deny = SimpleNamespace(decide=lambda *args: PermissionDecision.DENY)
        with patch("tiny_harness.tools.search.search_code", side_effect=AssertionError("executed")):
            self.assertIn("blocked", dispatch(registry, call, tool_hooks=hooks).content)
            self.assertIn("Permission denied", dispatch(registry, call, permission_policy=deny).content)
        self.assertNotIn("query", registry.lookup("search_code").trace_metadata({"query": "secret"}))
