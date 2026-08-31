import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evals.core import load_cases
from evals.graders import prepare_case, run_hidden_grader

ROOT = Path(__file__).parents[1]
PILOT_CASES = ROOT / "evals" / "pilot_cases.json"
PILOT_FIXTURES = ROOT / "evals" / "pilot_fixtures"

EXPECTED_FILES = {
    "free_shipping_policy": {
        "pyproject.toml",
        "src/shop/__init__.py",
        "src/shop/messages.py",
        "src/shop/policy.py",
        "src/shop/shipping.py",
        "tests/test_messages.py",
        "tests/test_shipping.py",
    },
    "ticket_status_migration": {
        "pyproject.toml",
        "src/tickets/__init__.py",
        "src/tickets/codec.py",
        "src/tickets/model.py",
        "src/tickets/statuses.py",
        "tests/test_codec.py",
        "tests/test_model.py",
    },
    "notification_preferences": {
        "pyproject.toml",
        "src/preferences/__init__.py",
        "src/preferences/loader.py",
        "src/preferences/model.py",
        "src/preferences/serialization.py",
        "tests/test_loader.py",
        "tests/test_model.py",
        "tests/test_serialization.py",
    },
    "customer_record_cleaning": {
        "pyproject.toml",
        "src/importer/__init__.py",
        "src/importer/pipeline.py",
        "src/importer/records.py",
        "tests/test_pipeline.py",
        "tests/test_records.py",
    },
    "weather_timeout_source": {
        "config/settings.example.json",
        "pyproject.toml",
        "src/weather/__init__.py",
        "src/weather/client.py",
        "src/weather/defaults.py",
        "src/weather/loader.py",
        "tests/test_client.py",
        "tests/test_loader.py",
    },
    "contact_name_order": {
        "pyproject.toml",
        "src/contacts/__init__.py",
        "src/contacts/export.py",
        "src/contacts/formatting.py",
        "tests/test_export.py",
        "tests/test_formatting.py",
    },
}

REFERENCE_CHANGES = {
    "free_shipping_policy": {
        "src/shop/policy.py": (
            '"""Shared shop policies."""\n\n'
            "FREE_SHIPPING_THRESHOLD = 75.0\n"
        ),
        "src/shop/shipping.py": (
            '"""Shipping cost calculation."""\n\n'
            "from shop import policy\n\n\n"
            "def shipping_cost(order_total: float) -> float:\n"
            "    if order_total >= policy.FREE_SHIPPING_THRESHOLD:\n"
            "        return 0.0\n"
            "    return 5.99\n"
        ),
        "src/shop/messages.py": (
            '"""Customer-facing policy messages."""\n\n'
            "from shop import policy\n\n\n"
            "def free_shipping_message() -> str:\n"
            "    return (\n"
            '        f"Free shipping on orders of '
            '${policy.FREE_SHIPPING_THRESHOLD:.2f} or more."\n'
            "    )\n"
        ),
    },
    "ticket_status_migration": {
        "src/tickets/statuses.py": (
            '"""Ticket status definitions."""\n\n'
            'VALID_STATUSES = frozenset({"open", "active", "closed"})\n'
        ),
        "src/tickets/codec.py": (
            '"""Ticket dictionary serialization."""\n\n'
            "from collections.abc import Mapping\n"
            "from typing import Any\n\n"
            "from tickets.model import Ticket\n\n\n"
            "def ticket_from_dict(payload: Mapping[str, Any]) -> Ticket:\n"
            '    status = "active" if payload["status"] == "in_progress" '
            'else payload["status"]\n'
            "    return Ticket(status=status)\n\n\n"
            "def ticket_to_dict(ticket: Ticket) -> dict[str, str]:\n"
            '    return {"status": ticket.status}\n'
        ),
    },
    "notification_preferences": {
        "src/preferences/model.py": (
            '"""Preference model."""\n\n'
            "from dataclasses import dataclass\n\n\n"
            "@dataclass(frozen=True)\n"
            "class NotificationPreferences:\n"
            "    email_enabled: bool = True\n"
            "    push_enabled: bool = True\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if type(self.email_enabled) is not bool:\n"
            '            raise ValueError("email_enabled must be a bool")\n'
            "        if type(self.push_enabled) is not bool:\n"
            '            raise ValueError("push_enabled must be a bool")\n'
        ),
        "src/preferences/loader.py": (
            '"""Load preferences from stored payloads."""\n\n'
            "from collections.abc import Mapping\n"
            "from typing import Any\n\n"
            "from preferences.model import NotificationPreferences\n\n\n"
            "def load_preferences(payload: Mapping[str, Any]) "
            "-> NotificationPreferences:\n"
            "    return NotificationPreferences(\n"
            '        email_enabled=payload.get("email_enabled", True),\n'
            '        push_enabled=payload.get("push_enabled", True),\n'
            "    )\n"
        ),
        "src/preferences/serialization.py": (
            '"""Serialize notification preferences."""\n\n'
            "from preferences.model import NotificationPreferences\n\n\n"
            "def preferences_to_dict(\n"
            "    preferences: NotificationPreferences,\n"
            ") -> dict[str, bool]:\n"
            "    return {\n"
            '        "email_enabled": preferences.email_enabled,\n'
            '        "push_enabled": preferences.push_enabled,\n'
            "    }\n"
        ),
    },
    "customer_record_cleaning": {
        "src/importer/records.py": (
            '"""Individual customer record cleanup."""\n\n'
            "import re\n"
            "from typing import Any\n\n"
            '_INTEGER = re.compile(r"[+-]?\\d+")\n\n\n'
            "def clean_customer_record(record: dict[str, Any]) "
            "-> dict[str, Any]:\n"
            "    cleaned = dict(record)\n"
            '    email = cleaned.get("email")\n'
            "    if isinstance(email, str):\n"
            '        cleaned["email"] = email.strip() or None\n'
            '    age = cleaned.get("age")\n'
            "    if isinstance(age, str):\n"
            "        normalized = age.strip()\n"
            "        if _INTEGER.fullmatch(normalized) is None:\n"
            '            raise ValueError("age must be a decimal integer")\n'
            '        cleaned["age"] = int(normalized)\n'
            "    return cleaned\n"
        ),
        "src/importer/pipeline.py": (
            '"""Customer import pipeline."""\n\n'
            "from collections.abc import Iterable\n"
            "from typing import Any\n\n"
            "from importer.records import clean_customer_record\n\n\n"
            "def import_customers(\n"
            "    records: Iterable[dict[str, Any]],\n"
            ") -> list[dict[str, Any]]:\n"
            "    return [clean_customer_record(record) for record in records]\n"
        ),
    },
    "weather_timeout_source": {
        "src/weather/defaults.py": (
            '"""Runtime defaults for the weather package."""\n\n'
            "DEFAULT_TIMEOUT_SECONDS = 30\n"
        ),
    },
    "contact_name_order": {
        "src/contacts/formatting.py": (
            '"""Format a contact name."""\n\n\n'
            "def format_contact(\n"
            "    first_name: str,\n"
            "    last_name: str,\n"
            "    *,\n"
            '    name_order: str = "first_last",\n'
            ") -> str:\n"
            '    if name_order == "first_last":\n'
            '        return f"{first_name} {last_name}"\n'
            '    if name_order == "last_first":\n'
            '        return f"{last_name}, {first_name}"\n'
            '    raise ValueError(f"Unsupported name order: {name_order}")\n'
        ),
        "src/contacts/export.py": (
            '"""Export contact names."""\n\n'
            "from collections.abc import Iterable, Mapping\n\n"
            "from contacts.formatting import format_contact\n\n\n"
            "def export_contacts(\n"
            "    contacts: Iterable[Mapping[str, str]],\n"
            "    *,\n"
            '    name_order: str = "first_last",\n'
            ") -> list[str]:\n"
            '    if name_order not in {"first_last", "last_first"}:\n'
            '        raise ValueError(f"Unsupported name order: {name_order}")\n'
            "    return [\n"
            "        format_contact(\n"
            '            contact["first_name"],\n'
            '            contact["last_name"],\n'
            "            name_order=name_order,\n"
            "        )\n"
            "        for contact in contacts\n"
            "    ]\n"
        ),
    },
}


def run_visible_tests(workspace: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
        ],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


class PilotCaseContractTest(unittest.TestCase):
    def test_six_cases_have_complete_user_tasks(self):
        cases = load_cases(PILOT_CASES)

        self.assertEqual(len(cases), 6)
        self.assertEqual({case.id for case in cases}, set(EXPECTED_FILES))
        self.assertTrue(all(case.task for case in cases))
        self.assertTrue(all(case.max_turns == 16 for case in cases))
        self.assertTrue(
            all(
                case.allowed_bash
                == ("python -m unittest discover -s tests -v",)
                for case in cases
            )
        )
        self.assertTrue(all("必须运行" not in case.task for case in cases))

    def test_task_four_explicitly_allows_optional_sign(self):
        case = next(
            case
            for case in load_cases(PILOT_CASES)
            if case.id == "customer_record_cleaning"
        )

        self.assertIn("可选正号或负号", case.task)

    def test_seed_repositories_have_exact_small_file_sets_and_valid_python(self):
        for case_id, expected in EXPECTED_FILES.items():
            with self.subTest(case=case_id):
                workspace = PILOT_FIXTURES / case_id / "workspace"
                actual = {
                    path.relative_to(workspace).as_posix()
                    for path in workspace.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(actual, expected)
                grader = PILOT_FIXTURES / case_id / "hidden_grader" / "test_hidden.py"
                self.assertTrue(grader.is_file())
                for path in [*workspace.rglob("*.py"), grader]:
                    compile(path.read_text(encoding="utf-8"), str(path), "exec")


class PilotGraderTest(unittest.TestCase):
    def test_every_seed_runs_and_fails_visible_and_hidden_checks_deterministically(self):
        cases = load_cases(PILOT_CASES)
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            for case in cases:
                with self.subTest(case=case.id):
                    prepared = prepare_case(
                        case,
                        fixtures_root=PILOT_FIXTURES,
                        results_root=results,
                        profile="seed",
                        repetition=1,
                    )
                    visible_first = run_visible_tests(prepared.workspace)
                    visible_second = run_visible_tests(prepared.workspace)
                    self.assertNotEqual(visible_first.returncode, 0)
                    self.assertEqual(
                        visible_first.returncode,
                        visible_second.returncode,
                    )

                    hidden_first = run_hidden_grader(prepared)
                    hidden_second = run_hidden_grader(prepared)
                    self.assertTrue(hidden_first.valid)
                    self.assertFalse(hidden_first.passed)
                    self.assertEqual(hidden_first.passed, hidden_second.passed)
                    self.assertEqual(
                        hidden_first.exit_code,
                        hidden_second.exit_code,
                    )
                    self.assertEqual(
                        hidden_first.snapshot_digest,
                        hidden_second.snapshot_digest,
                    )

    def test_every_reference_change_passes_visible_and_hidden_checks(self):
        cases = load_cases(PILOT_CASES)
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            for case in cases:
                with self.subTest(case=case.id):
                    prepared = prepare_case(
                        case,
                        fixtures_root=PILOT_FIXTURES,
                        results_root=results,
                        profile="reference",
                        repetition=1,
                    )
                    for relative, content in REFERENCE_CHANGES[case.id].items():
                        (prepared.workspace / relative).write_text(
                            content,
                            encoding="utf-8",
                        )

                    visible = run_visible_tests(prepared.workspace)
                    self.assertEqual(
                        visible.returncode,
                        0,
                        msg=visible.stdout + visible.stderr,
                    )
                    hidden = run_hidden_grader(prepared)
                    self.assertTrue(hidden.valid)
                    self.assertTrue(hidden.passed)
                    self.assertEqual(hidden.exit_code, 0)

    def test_each_grader_rejects_visible_test_modification(self):
        cases = load_cases(PILOT_CASES)
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            for case in cases:
                with self.subTest(case=case.id):
                    prepared = prepare_case(
                        case,
                        fixtures_root=PILOT_FIXTURES,
                        results_root=results,
                        profile="test-mutation",
                        repetition=1,
                    )
                    for relative, content in REFERENCE_CHANGES[case.id].items():
                        (prepared.workspace / relative).write_text(
                            content,
                            encoding="utf-8",
                        )
                    visible_test = next(
                        (prepared.workspace / "tests").glob("test_*.py")
                    )
                    visible_test.write_text(
                        visible_test.read_text(encoding="utf-8") + "\n# changed\n",
                        encoding="utf-8",
                    )

                    hidden = run_hidden_grader(prepared)

                    self.assertTrue(hidden.valid)
                    self.assertFalse(hidden.passed)


if __name__ == "__main__":
    unittest.main()
