import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE))

from src.settings import TIMEOUT_SECONDS


class SettingsHiddenTest(unittest.TestCase):
    def test_real_setting_changed(self):
        self.assertEqual(TIMEOUT_SECONDS, 30)

    def test_wrong_duplicate_path_not_created(self):
        self.assertFalse((WORKSPACE / "app" / "settings.py").exists())
