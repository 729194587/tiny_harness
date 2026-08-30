import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from preferences.model import NotificationPreferences
from preferences.serialization import preferences_to_dict


class PreferenceSerializationTest(unittest.TestCase):
    def test_includes_push_value(self):
        payload = preferences_to_dict(NotificationPreferences())
        self.assertIs(payload["push_enabled"], True)


if __name__ == "__main__":
    unittest.main()
