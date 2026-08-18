import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE))

import config
import formatter


class CurrencyHiddenTest(unittest.TestCase):
    def test_uses_live_config_value(self):
        original = config.CURRENCY
        try:
            config.CURRENCY = "JPY"
            self.assertEqual(formatter.format_price(3), "JPY 3.00")
        finally:
            config.CURRENCY = original

    def test_negative_amount(self):
        self.assertEqual(formatter.format_price(-1.2), "EUR -1.20")
