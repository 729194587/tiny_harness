import hashlib
import inspect
import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE / "src"))

import contacts.export as export_module
from contacts.export import export_contacts
from contacts.formatting import format_contact

EXPECTED_TEST_HASHES = {
    "test_export.py": "ec10de73f87569aaf296294141e72bba9f4b5a54cc0b281a95170cd8e79d84a0",
    "test_formatting.py": "39927f8bc8ae8cd7daab8c524ccc4994a97c565855f3116057936bdce6354f44",
}


class ContactNameOrderHiddenTest(unittest.TestCase):
    def test_formatter_modes_and_default(self):
        self.assertEqual(format_contact("Ada", "Lovelace"), "Ada Lovelace")
        self.assertEqual(
            format_contact("Ada", "Lovelace", name_order="first_last"),
            "Ada Lovelace",
        )
        self.assertEqual(
            format_contact("Ada", "Lovelace", name_order="last_first"),
            "Lovelace, Ada",
        )

    def test_export_preserves_order_and_forwards_mode(self):
        contacts = [
            {"first_name": "Ada", "last_name": "Lovelace"},
            {"first_name": "Grace", "last_name": "Hopper"},
        ]
        self.assertEqual(
            export_contacts(contacts, name_order="last_first"),
            ["Lovelace, Ada", "Hopper, Grace"],
        )

        calls = []
        original = export_module.format_contact

        def recording_formatter(first_name, last_name, *, name_order="first_last"):
            calls.append((first_name, last_name, name_order))
            return f"{last_name}, {first_name}"

        try:
            export_module.format_contact = recording_formatter
            export_module.export_contacts(contacts, name_order="last_first")
        finally:
            export_module.format_contact = original
        self.assertEqual(
            calls,
            [
                ("Ada", "Lovelace", "last_first"),
                ("Grace", "Hopper", "last_first"),
            ],
        )

    def test_invalid_modes_raise_even_for_empty_export(self):
        with self.assertRaises(ValueError):
            format_contact("Ada", "Lovelace", name_order="family_first")
        with self.assertRaises(ValueError):
            export_contacts(
                [{"first_name": "Ada", "last_name": "Lovelace"}],
                name_order="family_first",
            )
        with self.assertRaises(ValueError):
            export_contacts([], name_order="family_first")

    def test_name_order_is_keyword_only(self):
        for function in (format_contact, export_contacts):
            parameter = inspect.signature(function).parameters["name_order"]
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(parameter.default, "first_last")

    def test_visible_tests_are_unchanged(self):
        tests = WORKSPACE / "tests"
        self.assertEqual(
            {path.name for path in tests.glob("test_*.py")},
            set(EXPECTED_TEST_HASHES),
        )
        for name, expected in EXPECTED_TEST_HASHES.items():
            content = (tests / name).read_text(encoding="utf-8")
            actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
