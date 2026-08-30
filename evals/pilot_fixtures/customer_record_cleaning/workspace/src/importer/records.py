"""Individual customer record cleanup."""

from typing import Any


def clean_customer_record(record: dict[str, Any]) -> dict[str, Any]:
    email = record.get("email")
    if isinstance(email, str):
        record["email"] = email.strip() or None
    age = record.get("age")
    if isinstance(age, str):
        record["age"] = int(age.strip())
    return record
