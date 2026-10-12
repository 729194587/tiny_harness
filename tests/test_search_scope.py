import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.tools.search import (
    grep_text, search_code, glob_files, IGNORED_SEARCH_DIRECTORIES,
)


class SearchScopeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        for path in ("root.py", "root.txt", "src/main.py", "src/deep/extra.py", "src/main.txt"):
            self.write(path)

    def write(self, path):
        file = self.workspace / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("needle\n", encoding="utf-8")

    def search(self, tool, **kwargs):
        if tool is search_code:
            kwargs["context_lines"] = 0
        return tool(self.workspace, "needle", **kwargs)

    def test_globs_are_relative_and_double_star_includes_zero_directories(self):
        for tool in (grep_text, search_code):
            for pattern, expected in (
                ("*.py", {"root.py"}),
                ("**/*.py", {"root.py", "src/main.py", "src/deep/extra.py"}),
                ("src/**/*.py", {"src/main.py", "src/deep/extra.py"}),
                ("src/m?in.[pt]y", {"src/main.py"}),
            ):
                with self.subTest(tool=tool.__name__, pattern=pattern):
                    result = self.search(tool, include=pattern)
                    self.assertEqual({line.split(":")[0] for line in result.splitlines() if ":" in line}, expected)
            self.assertEqual(self.search(tool, path="src", include="*.py"), "src/main.py:1:needle")
            self.assertEqual(self.search(tool, path="root.py", include="*.txt"), "root.py:1:needle")
            self.assertEqual(self.search(tool, include="**/*.missing"), "(no matches)")
            for invalid in ("", "../*.py", str(self.workspace / "*.py"), None, 1):
                with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                    self.search(tool, include=invalid)

    def test_ignored_directories_pruned_and_explicit_paths_override(self):
        for name in IGNORED_SEARCH_DIRECTORIES:
            self.write(f"{name}/hidden.py")
            self.write(f"src/{name}/hidden.py")
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertFalse(set(path.relative_to(self.workspace).parts) & IGNORED_SEARCH_DIRECTORIES)
            return original_open(path, *args, **kwargs)

        for tool in (grep_text, search_code):
            with patch.object(Path, "open", guarded_open):
                self.assertNotIn("hidden", self.search(tool))
            for name in IGNORED_SEARCH_DIRECTORIES:
                for path in (name, f"{name}/hidden.py", f"src/{name}"):
                    with self.subTest(tool=tool.__name__, path=path):
                        self.assertIn("hidden.py:1:needle", self.search(tool, path=path))
                self.assertEqual(self.search(tool, include=f"{name}/**/*.py"), "(no matches)")
        # Discovery is intentionally unaffected by content-search exclusions.
        self.assertIn(".venv/hidden.py", glob_files(self.workspace, "**/*.py"))

    def test_symlinks_do_not_bypass_boundary_or_default_exclusions(self):
        self.write(".venv/hidden.py")
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret.py"
            secret.write_text("needle", encoding="utf-8")
            try:
                (self.workspace / "alias.py").symlink_to(self.workspace / ".venv/hidden.py")
                (self.workspace / "escape.py").symlink_to(secret)
                (self.workspace / "dir_alias").symlink_to(self.workspace / "src", target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Symlinks unavailable: {error}")
            for tool in (grep_text, search_code):
                result = self.search(tool)
                self.assertNotIn("alias", result)
                self.assertNotIn("escape", result)
                self.assertIn("needle", self.search(tool, path="alias.py"))
                with self.assertRaisesRegex(ValueError, "escapes workspace"):
                    self.search(tool, path="escape.py")
