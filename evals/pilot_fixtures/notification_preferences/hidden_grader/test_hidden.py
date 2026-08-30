import hashlib
import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE / "src"))

from preferences.loader import load_preferences
from preferences.model import NotificationPreferences
from preferences.serialization import preferences_to_dict

EXPECTED_TEST_HASHES = {
    "test_loader.py": "4ceaf096d7611aad921d7f03a5e6928d27b4a6eaf9175dce3b88d3c8a8313502",
    "test_model.py": "f73806428bb7d0c8c746a3717ed64bb88a59d333ba74cb47c04fc6da435d49c1",
    "test_serialization.py": "0ea8d02276a2d371f4c9e6f83e22023b158accb02cfad00a8b365ad087418923",
}


class NotificationPreferenceHiddenTest(unittest.TestCase):
    def test_direct_and_legacy_defaults(self):
        self.assertIs(NotificationPreferences().push_enabled, True)
        self.assertIs(load_preferences({}).push_enabled, True)

    def test_explicit_false_survives_load_and_serialization(self):
        preferences = load_preferences(
            {"email_enabled": False, "push_enabled": False}
        )
        self.assertIs(preferences.push_enabled, False)
        payload = preferences_to_dict(preferences)
        self.assertIs(payload["push_enabled"], False)
        self.assertIs(type(payload["push_enabled"]), bool)

    def test_non_bool_push_values_are_rejected_at_both_boundaries(self):
        invalid_values = ("false", "", 0, 1, None, [], [1], {}, {"x": 1})
        for value in invalid_values:
            with self.subTest(value=repr(value), boundary="model"):
                with self.assertRaises(ValueError):
                    NotificationPreferences(push_enabled=value)
            with self.subTest(value=repr(value), boundary="loader"):
                with self.assertRaises(ValueError):
                    load_preferences({"push_enabled": value})

    def test_serialization_always_includes_a_bool(self):
        for push_enabled in (True, False):
            payload = preferences_to_dict(
                NotificationPreferences(push_enabled=push_enabled)
            )
            self.assertIn("push_enabled", payload)
            self.assertIs(type(payload["push_enabled"]), bool)

    def test_email_behavior_is_unchanged(self):
        self.assertIs(NotificationPreferences().email_enabled, True)
        preferences = load_preferences({"email_enabled": False})
        self.assertIs(preferences.email_enabled, False)
        self.assertIs(preferences_to_dict(preferences)["email_enabled"], False)

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
