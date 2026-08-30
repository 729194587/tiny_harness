"""Export contact names."""

from collections.abc import Iterable, Mapping

from contacts.formatting import format_contact


def export_contacts(
    contacts: Iterable[Mapping[str, str]],
) -> list[str]:
    return [
        format_contact(contact["first_name"], contact["last_name"])
        for contact in contacts
    ]
