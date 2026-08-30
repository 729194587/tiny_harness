import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from shop.messages import free_shipping_message


class MessageTest(unittest.TestCase):
    def test_default_message(self):
        self.assertEqual(
            free_shipping_message(),
            "Free shipping on orders of $75.00 or more.",
        )


if __name__ == "__main__":
    unittest.main()
