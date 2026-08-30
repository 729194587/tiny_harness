import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from preferences.loader import load_preferences


class PreferenceLoaderTest(unittest.TestCase):
    def test_explicit_push_false_is_preserved(self):
        preferences = load_preferences({"push_enabled": False})
        self.assertIs(preferences.push_enabled, False)


if __name__ == "__main__":
    unittest.main()
