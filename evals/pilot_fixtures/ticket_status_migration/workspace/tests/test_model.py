import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tickets.model import Ticket


class TicketModelTest(unittest.TestCase):
    def test_accepts_active(self):
        self.assertEqual(Ticket("active").status, "active")

    def test_open_is_unchanged(self):
        self.assertEqual(Ticket("open").status, "open")


if __name__ == "__main__":
    unittest.main()
