import hashlib
import inspect
import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE / "src"))

from shop import policy
from shop.messages import free_shipping_message
from shop.shipping import shipping_cost

EXPECTED_TEST_HASHES = {
    "test_messages.py": "70f137ff421a857c34842a6c1e5571d65370aefe7c2479fda64e074d29faa467",
    "test_shipping.py": "a7059a508907128a7d814ce47e1de3bbd79a12b1e3854a4454120f0aaa767950",
}


class FreeShippingHiddenTest(unittest.TestCase):
    def test_default_policy_and_boundaries(self):
        self.assertEqual(policy.FREE_SHIPPING_THRESHOLD, 75.0)
        self.assertEqual(shipping_cost(74.99), 5.99)
        self.assertEqual(shipping_cost(75.0), 0.0)
        self.assertEqual(shipping_cost(75.01), 0.0)
        self.assertEqual(
            free_shipping_message(),
            "Free shipping on orders of $75.00 or more.",
        )

    def test_both_consumers_read_the_current_policy_each_time(self):
        original = policy.FREE_SHIPPING_THRESHOLD
        try:
            policy.FREE_SHIPPING_THRESHOLD = 90.0
            self.assertEqual(shipping_cost(80.0), 5.99)
            self.assertEqual(shipping_cost(90.0), 0.0)
            self.assertEqual(
                free_shipping_message(),
                "Free shipping on orders of $90.00 or more.",
            )
        finally:
            policy.FREE_SHIPPING_THRESHOLD = original

    def test_public_signatures_are_unchanged(self):
        shipping_parameters = list(inspect.signature(shipping_cost).parameters.values())
        self.assertEqual(len(shipping_parameters), 1)
        self.assertEqual(shipping_parameters[0].name, "order_total")
        self.assertEqual(
            shipping_parameters[0].kind,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        self.assertEqual(len(inspect.signature(free_shipping_message).parameters), 0)

    def test_visible_tests_are_unchanged(self):
        tests = WORKSPACE / "tests"
        self.assertEqual(
            {path.name for path in tests.glob("test_*.py")},
            set(EXPECTED_TEST_HASHES),
        )
        for name, expected in EXPECTED_TEST_HASHES.items():
            content = (tests / name).read_text(encoding="utf-8")
            actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
