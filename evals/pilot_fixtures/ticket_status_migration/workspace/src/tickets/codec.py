"""Ticket dictionary serialization."""

from collections.abc import Mapping
from typing import Any

from tickets.model import Ticket


def ticket_from_dict(payload: Mapping[str, Any]) -> Ticket:
    return Ticket(status=payload["status"])


def ticket_to_dict(ticket: Ticket) -> dict[str, str]:
    return {"status": ticket.status}
