import unittest

import config
from formatter import format_price


class CurrencyVisibleTest(unittest.TestCase):
    def test_default_is_euro(self):
        self.assertEqual(config.CURRENCY, "EUR")

    def test_formats_price(self):
        self.assertEqual(format_price(12.5), "EUR 12.50")


if __name__ == "__main__":
    unittest.main()
