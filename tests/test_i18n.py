import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import i18n


class I18nTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for pair in i18n.REQUIRED:
            self.make_pair(pair)

    def make_pair(self, pair):
        en, zh, _ = i18n.pair_paths(self.root, pair)
        en.parent.mkdir(parents=True, exist_ok=True)
        en.write_text("# English\n", encoding="utf-8")
        zh.write_text("# 中文\n", encoding="utf-8")
        i18n.record(self.root, pair)

    def snapshot(self):
        return {p.relative_to(self.root): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob("*") if p.is_file()}

    def test_valid_check_is_read_only(self):
        before = self.snapshot()
        with patch.object(i18n, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(i18n.main(["check"]), 0)
        self.assertEqual(before, self.snapshot())

    def test_missing_files_including_entire_required_pair(self):
        en, zh, record = i18n.pair_paths(self.root, "README")
        for path in (en, zh, record):
            with self.subTest(path=path.name):
                content = path.read_bytes()
                path.unlink()
                self.assertTrue(i18n.check(self.root))
                path.write_bytes(content)
        for path in (en, zh, record):
            path.unlink()
        self.assertTrue(any("README" in error for error in i18n.check(self.root)))

    def test_changed_languages_are_reported_without_writes(self):
        en, zh, _ = i18n.pair_paths(self.root, "README")
        for paths, labels in [([en], ["English"]), ([zh], ["简体中文"]),
                              ([en, zh], ["English", "简体中文"])]:
            with self.subTest(labels=labels):
                i18n.record(self.root, "README")
                for path in paths:
                    path.write_bytes(path.read_bytes() + b"changed\n")
                before = self.snapshot()
                errors = i18n.check(self.root)
                self.assertEqual(len(errors), 1)
                for label in labels:
                    self.assertIn(label, errors[0])
                for label in {"English", "简体中文"} - set(labels):
                    self.assertNotIn(label, errors[0])
                self.assertIn("共同审阅", errors[0])
                self.assertEqual(before, self.snapshot())

    def test_invalid_records(self):
        record = self.root / "README.i18n.yaml"
        valid = record.read_text(encoding="utf-8")
        for data in ["", "[", "[]", "null", "en_sha256: short\nzh_CN_sha256: bad\n",
                     "en_sha256: 1\nzh_CN_sha256: 2\n", valid + "extra: value\n",
                     valid + "en_sha256: '" + "a" * 64 + "'\n"]:
            with self.subTest(data=data):
                record.write_text(data, encoding="utf-8")
                before = self.snapshot()
                self.assertTrue(any("无效记录" in e for e in i18n.check(self.root)))
                self.assertEqual(before, self.snapshot())

    def test_new_document_and_orphan_record_are_discovered(self):
        for pair in ("new-guide", "docs/nested/new-guide", "evals/swe_bench_lite/new-guide"):
            en, zh, record = i18n.pair_paths(self.root, pair)
            en.parent.mkdir(parents=True, exist_ok=True)
            en.write_text("New\n", encoding="utf-8")
            errors = i18n.check(self.root)
            self.assertTrue(any(pair in e and "缺少语言" in e for e in errors))
            self.make_pair(pair)
        orphan = self.root / "docs/old.i18n.yaml"
        orphan.write_text((self.root / "README.i18n.yaml").read_text(), encoding="utf-8")
        self.assertTrue(any("无法识别记录" in e for e in i18n.check(self.root)))

    def test_explicit_record_updates_only_requested_pair(self):
        en, zh, record = i18n.pair_paths(self.root, "README")
        en.write_text("Changed English\n", encoding="utf-8")
        zh.write_text("更新中文\n", encoding="utf-8")
        record.write_text("invalid: [", encoding="utf-8")
        before = self.snapshot()
        with patch.object(i18n, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(i18n.main(["record", "README.md"]), 0)
        after = self.snapshot()
        self.assertEqual([p for p in before if before[p] != after[p]],
                         [Path("README.i18n.yaml")])
        self.assertEqual(i18n.read_record(record), {
            "en_sha256": hashlib.sha256(en.read_bytes()).hexdigest(),
            "zh_CN_sha256": hashlib.sha256(zh.read_bytes()).hexdigest(),
        })
        self.assertEqual(i18n.check(self.root), [])

    def test_whole_file_bytes_include_line_endings(self):
        en = self.root / "README.md"
        en.write_bytes(b"# English\r\n")
        i18n.record(self.root, "README")
        en.write_bytes(b"# English\n")
        self.assertIn("English", i18n.check(self.root)[0])

    def test_record_rejects_incomplete_or_out_of_scope_pair(self):
        (self.root / "README.zh-CN.md").unlink()
        for pair in ("README", "../outside", "tiny_harness/new", "README.zh-CN.md"):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                i18n.record(self.root, pair)

    def test_scope_exclusions(self):
        for name in ("AGENTS.md", "docs/SKILL.md", "docs/AGENTS.zh-CN.md",
                     "docs/superpowers/plans/history.md", "tiny_harness/skills/x/SKILL.md",
                     "evals/results/run/summary.md"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Excluded\n", encoding="utf-8")
        self.assertEqual(i18n.check(self.root), [])

    def test_check_cli_failure_does_not_record(self):
        (self.root / "README.i18n.yaml").unlink()
        before = self.snapshot()
        with patch.object(i18n, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(i18n.main(["check"]), 1)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
