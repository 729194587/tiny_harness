import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from contacts.formatting import format_contact


class ContactFormattingTest(unittest.TestCase):
    def test_default_keeps_existing_output(self):
        self.assertEqual(format_contact("Ada", "Lovelace"), "Ada Lovelace")

    def test_last_name_first(self):
        self.assertEqual(
            format_contact("Ada", "Lovelace", name_order="last_first"),
            "Lovelace, Ada",
        )


if __name__ == "__main__":
    unittest.main()
