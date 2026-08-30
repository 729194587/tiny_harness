import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from importer.records import clean_customer_record


class CustomerRecordTest(unittest.TestCase):
    def test_cleans_normal_record_without_mutating_input(self):
        source = {"email": " ada@example.com ", "age": " 36 ", "tier": "pro"}

        cleaned = clean_customer_record(source)

        self.assertEqual(
            cleaned,
            {"email": "ada@example.com", "age": 36, "tier": "pro"},
        )
        self.assertEqual(
            source,
            {"email": " ada@example.com ", "age": " 36 ", "tier": "pro"},
        )
        self.assertIsNot(cleaned, source)

    def test_blank_email_becomes_none(self):
        self.assertIs(clean_customer_record({"email": "  "})["email"], None)


if __name__ == "__main__":
    unittest.main()
