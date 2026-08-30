import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tickets.codec import ticket_from_dict, ticket_to_dict
from tickets.model import Ticket


class TicketCodecTest(unittest.TestCase):
    def test_loads_legacy_status_as_active(self):
        ticket = ticket_from_dict({"status": "in_progress"})
        self.assertEqual(ticket.status, "active")

    def test_serializes_active(self):
        self.assertEqual(ticket_to_dict(Ticket("active")), {"status": "active"})


if __name__ == "__main__":
    unittest.main()
