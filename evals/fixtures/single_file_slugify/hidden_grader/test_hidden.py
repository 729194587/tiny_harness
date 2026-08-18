import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE))

from text_utils import slugify


class SlugifyHiddenTest(unittest.TestCase):
    def test_collapses_mixed_whitespace(self):
        self.assertEqual(slugify("  Alpha\t \nBeta   Gamma "), "alpha-beta-gamma")

    def test_empty_input(self):
        self.assertEqual(slugify(" \t\n "), "")
