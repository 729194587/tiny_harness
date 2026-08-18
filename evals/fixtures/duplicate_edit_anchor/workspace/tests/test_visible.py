import unittest

from status import is_closed, is_ready


class StatusVisibleTest(unittest.TestCase):
    def test_ready(self):
        self.assertTrue(is_ready("ready"))

    def test_unrelated_function_is_unchanged(self):
        self.assertFalse(is_closed("closed"))


if __name__ == "__main__":
    unittest.main()
