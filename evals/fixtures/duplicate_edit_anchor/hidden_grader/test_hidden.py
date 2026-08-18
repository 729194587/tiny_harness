import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE))

from status import is_closed, is_ready


class StatusHiddenTest(unittest.TestCase):
    def test_ready_only_for_ready(self):
        self.assertTrue(is_ready("ready"))
        self.assertFalse(is_ready("pending"))
        self.assertFalse(is_ready("closed"))

    def test_closed_behavior_remains_false(self):
        self.assertFalse(is_closed("closed"))
        self.assertFalse(is_closed("ready"))
