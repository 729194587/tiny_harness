import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from preferences.model import NotificationPreferences


class PreferenceModelTest(unittest.TestCase):
    def test_push_defaults_to_true(self):
        preferences = NotificationPreferences()
        self.assertIs(preferences.push_enabled, True)

    def test_email_false_is_unchanged(self):
        preferences = NotificationPreferences(email_enabled=False)
        self.assertIs(preferences.email_enabled, False)


if __name__ == "__main__":
    unittest.main()
