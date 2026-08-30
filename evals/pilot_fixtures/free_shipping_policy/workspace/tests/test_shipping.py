import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from shop import policy
from shop.shipping import shipping_cost


class ShippingTest(unittest.TestCase):
    def test_default_threshold(self):
        self.assertEqual(policy.FREE_SHIPPING_THRESHOLD, 75.0)

    def test_cost_above_threshold(self):
        self.assertEqual(shipping_cost(80.0), 0.0)

    def test_cost_below_threshold(self):
        self.assertEqual(shipping_cost(40.0), 5.99)


if __name__ == "__main__":
    unittest.main()
