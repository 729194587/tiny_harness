import unittest

from palindrome import is_palindrome


class PalindromeVisibleTest(unittest.TestCase):
    def test_ignores_case(self):
        self.assertTrue(is_palindrome("RaceCar"))

    def test_ignores_punctuation(self):
        self.assertTrue(is_palindrome("Never odd, or even!"))

    def test_negative(self):
        self.assertFalse(is_palindrome("TinyHarness"))


if __name__ == "__main__":
    unittest.main()
