import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from weather.loader import load_settings


class SettingsLoaderTest(unittest.TestCase):
    def test_runtime_default(self):
        self.assertEqual(load_settings({}).timeout_seconds, 30)

    def test_environment_override(self):
        self.assertEqual(
            load_settings({"WEATHER_TIMEOUT": "12"}).timeout_seconds,
            12,
        )


if __name__ == "__main__":
    unittest.main()
