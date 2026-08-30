import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from contacts.export import export_contacts


class ContactExportTest(unittest.TestCase):
    def test_default_batch_keeps_existing_output(self):
        contacts = [
            {"first_name": "Ada", "last_name": "Lovelace"},
            {"first_name": "Grace", "last_name": "Hopper"},
        ]
        self.assertEqual(
            export_contacts(contacts),
            ["Ada Lovelace", "Grace Hopper"],
        )

    def test_non_empty_last_name_first_batch(self):
        contacts = [{"first_name": "Ada", "last_name": "Lovelace"}]
        self.assertEqual(
            export_contacts(contacts, name_order="last_first"),
            ["Lovelace, Ada"],
        )


if __name__ == "__main__":
    unittest.main()
