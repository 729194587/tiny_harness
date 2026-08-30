import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from weather.client import WeatherClient


class WeatherClientTest(unittest.TestCase):
    def test_uses_runtime_default(self):
        self.assertEqual(WeatherClient(environment={}).timeout_seconds, 30)

    def test_explicit_timeout_wins(self):
        client = WeatherClient(
            timeout_seconds=9,
            environment={"WEATHER_TIMEOUT": "12"},
        )
        self.assertEqual(client.timeout_seconds, 9)


if __name__ == "__main__":
    unittest.main()
