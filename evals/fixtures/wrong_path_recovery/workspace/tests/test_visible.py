import unittest

from src.settings import TIMEOUT_SECONDS


class SettingsVisibleTest(unittest.TestCase):
    def test_timeout(self):
        self.assertEqual(TIMEOUT_SECONDS, 30)


if __name__ == "__main__":
    unittest.main()
