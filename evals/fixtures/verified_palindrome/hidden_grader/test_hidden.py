import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE))

from palindrome import is_palindrome


class PalindromeHiddenTest(unittest.TestCase):
    def test_phrase(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama"))

    def test_digits_and_empty_normalized_value(self):
        self.assertTrue(is_palindrome("1-2-1"))
        self.assertTrue(is_palindrome("!!!"))

    def test_negative(self):
        self.assertFalse(is_palindrome("OpenAI"))
