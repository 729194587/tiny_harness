import tempfile
import unittest
from pathlib import Path

from tiny_harness.tools.filesystem import edit_file, list_files, read_file, write_file


class FilesystemToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_write_read_and_edit_file(self) -> None:
        result = write_file(self.workspace, "src/example.txt", "hello 世界")

        self.assertEqual(result, "Wrote 12 bytes to src/example.txt")
        self.assertEqual(read_file(self.workspace, "src/example.txt"), "hello 世界")
        self.assertEqual(
            edit_file(self.workspace, "src/example.txt", "hello", "goodbye"),
            "Edited src/example.txt",
        )
        self.assertEqual(read_file(self.workspace, "src/example.txt"), "goodbye 世界")

    def test_edit_rejects_non_unique_old_text(self) -> None:
        write_file(self.workspace, "example.txt", "same same")

        with self.assertRaisesRegex(
            ValueError,
            "Text is not unique in example.txt: found 2 occurrences",
        ):
            edit_file(self.workspace, "example.txt", "same", "new")

        self.assertEqual(read_file(self.workspace, "example.txt"), "same same")

    def test_edit_rejects_empty_or_missing_old_text(self) -> None:
        write_file(self.workspace, "example.txt", "content")

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            edit_file(self.workspace, "example.txt", "", "new")
        with self.assertRaisesRegex(ValueError, "Text not found"):
            edit_file(self.workspace, "example.txt", "missing", "new")

    def test_edit_preserves_lf_without_whole_file_newline_changes(self) -> None:
        path = self.workspace / "lf.txt"
        path.write_bytes(b"first\nTARGET\nthird\n")

        edit_file(self.workspace, "lf.txt", "TARGET", "changed")

        self.assertEqual(path.read_bytes(), b"first\nchanged\nthird\n")
        self.assertNotIn(b"\r\n", path.read_bytes())

    def test_edit_preserves_crlf_and_normalizes_replacement_to_it(self) -> None:
        path = self.workspace / "crlf.txt"
        path.write_bytes(b"first\r\nTARGET\r\nthird\r\n")

        edit_file(
            self.workspace,
            "crlf.txt",
            "TARGET\nthird",
            "changed\nreplacement",
        )

        self.assertEqual(
            path.read_bytes(),
            b"first\r\nchanged\r\nreplacement\r\n",
        )

    def test_write_file_does_not_translate_newlines(self) -> None:
        write_file(self.workspace, "exact.txt", "one\ntwo\r\nthree\n")

        self.assertEqual(
            (self.workspace / "exact.txt").read_bytes(),
            b"one\ntwo\r\nthree\n",
        )

    def test_list_files_returns_sorted_direct_children(self) -> None:
        (self.workspace / "zeta.txt").write_text("z", encoding="utf-8")
        (self.workspace / "Alpha.txt").write_text("a", encoding="utf-8")
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "nested.txt").write_text("n", encoding="utf-8")

        self.assertEqual(list_files(self.workspace), "Alpha.txt\nsrc/\nzeta.txt")
        self.assertEqual(list_files(self.workspace, "src"), "src/nested.txt")

    def test_list_files_reports_empty_directory(self) -> None:
        (self.workspace / "empty").mkdir()

        self.assertEqual(list_files(self.workspace, "empty"), "(no files)")

    def test_parent_traversal_is_rejected_by_every_file_tool(self) -> None:
        outside_file = self.root / "outside.txt"
        outside_file.write_text("outside", encoding="utf-8")

        operations = (
            lambda: read_file(self.workspace, "../outside.txt"),
            lambda: write_file(self.workspace, "../outside.txt", "changed"),
            lambda: edit_file(self.workspace, "../outside.txt", "outside", "changed"),
            lambda: list_files(self.workspace, ".."),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "Path escapes workspace"):
                    operation()

        self.assertEqual(outside_file.read_text(encoding="utf-8"), "outside")

    def test_absolute_path_outside_workspace_is_rejected(self) -> None:
        outside_file = self.root / "outside.txt"
        outside_file.write_text("outside", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Path escapes workspace"):
            read_file(self.workspace, str(outside_file))

    def test_existing_symlink_escape_is_rejected(self) -> None:
        outside_directory = self.root / "outside"
        outside_directory.mkdir()
        (outside_directory / "secret.txt").write_text("secret", encoding="utf-8")
        link = self.workspace / "link"
        try:
            link.symlink_to(outside_directory, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Creating symlinks is unavailable: {error}")

        with self.assertRaisesRegex(ValueError, "Path escapes workspace"):
            read_file(self.workspace, "link/secret.txt")
        with self.assertRaisesRegex(ValueError, "Path escapes workspace"):
            write_file(self.workspace, "link/new.txt", "new")


if __name__ == "__main__":
    unittest.main()
