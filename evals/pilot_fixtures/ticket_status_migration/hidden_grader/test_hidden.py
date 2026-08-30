import hashlib
import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE / "src"))

from tickets.codec import ticket_from_dict, ticket_to_dict
from tickets.model import Ticket

EXPECTED_TEST_HASHES = {
    "test_codec.py": "07bc8443bf244e2d9d3e439a19a81ebbba47b4a1751d71a8a7f61cd2608f451b",
    "test_model.py": "aecc5d76b6414b461a5f92eba57b7c77ef95ea45ac1ee47a48d0e61398fcd54d",
}


class TicketStatusHiddenTest(unittest.TestCase):
    def test_new_model_accepts_only_canonical_statuses(self):
        for status in ("open", "active", "closed"):
            with self.subTest(status=status):
                self.assertEqual(Ticket(status).status, status)
        for status in ("in_progress", "pending", ""):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    Ticket(status)

    def test_legacy_payload_is_normalized_across_round_trip(self):
        ticket = ticket_from_dict({"status": "in_progress"})
        self.assertEqual(ticket.status, "active")
        self.assertEqual(ticket_to_dict(ticket), {"status": "active"})

    def test_open_and_closed_round_trip_unchanged(self):
        for status in ("open", "closed"):
            with self.subTest(status=status):
                ticket = ticket_from_dict({"status": status})
                self.assertEqual(ticket.status, status)
                self.assertEqual(ticket_to_dict(ticket), {"status": status})

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
