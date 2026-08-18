import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.core import (
    EvalResult,
    RunMetrics,
    collect_metrics,
    load_cases,
)
from evals.graders import (
    GradeResult,
    hidden_grader_changed,
    prepare_case,
    run_hidden_grader,
)
from evals.run import (
    DEFAULT_CASES,
    DEFAULT_FIXTURES,
    EvalPermissionPolicy,
    main,
    render_markdown,
    run_real_case,
)
from evals.scenarios import run_offline_scenarios
from tiny_harness.runtime.permissions import PermissionDecision


class EvalContractTest(unittest.TestCase):
    def test_loads_five_unique_real_cases(self):
        cases = load_cases(DEFAULT_CASES)

        self.assertEqual(len(cases), 5)
        self.assertEqual(len({case.id for case in cases}), 5)
        self.assertTrue(all(case.goal for case in cases))
        self.assertTrue(all(case.allowed_bash for case in cases))
        self.assertTrue(all(case.max_turns == 12 for case in cases))

    def test_collects_purpose_specific_attempts_and_control_counts(self):
        events = [
            {
                "event_type": "model_requested",
                "data": {"purpose": "main", "turn": 1},
            },
            {
                "event_type": "model_requested",
                "data": {"purpose": "main", "turn": 2},
            },
            {
                "event_type": "model_requested",
                "data": {"purpose": "goal_evaluation", "turn": 2},
            },
            {
                "event_type": "model_requested",
                "data": {"purpose": "summary", "turn": 2},
            },
            {
                "event_type": "goal_evaluated",
                "data": {"retries_used": 1},
            },
            {"event_type": "model_retry_scheduled", "data": {}},
            {"event_type": "tool_started", "data": {}},
        ]

        metrics = collect_metrics(events)

        self.assertEqual(metrics.main_model_attempts, 2)
        self.assertEqual(metrics.goal_model_attempts, 1)
        self.assertEqual(metrics.summary_model_attempts, 1)
        self.assertEqual(metrics.total_model_attempts, 4)
        self.assertEqual(metrics.turns, 2)
        self.assertEqual(metrics.retries, 1)
        self.assertEqual(metrics.continuations, 1)
        self.assertEqual(metrics.tool_calls, 1)

    def test_eval_permission_policy_uses_exact_bash_allowlist(self):
        policy = EvalPermissionPolicy(["python -m unittest"])

        self.assertIs(
            policy.decide("read_file", {"path": "x"}),
            PermissionDecision.ALLOW,
        )
        self.assertIs(
            policy.decide("bash", {"command": "python -m unittest"}),
            PermissionDecision.ALLOW,
        )
        self.assertIs(
            policy.decide("bash", {"command": "python -m unittest -v"}),
            PermissionDecision.DENY,
        )


class ExternalGraderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.results_root = Path(self.temporary_directory.name)
        self.case = next(
            case
            for case in load_cases(DEFAULT_CASES)
            if case.id == "single_file_slugify"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def prepare(self):
        return prepare_case(
            self.case,
            fixtures_root=DEFAULT_FIXTURES,
            results_root=self.results_root,
            profile="test",
            repetition=1,
        )

    def test_hidden_grader_is_a_sibling_and_detects_mutation(self):
        prepared = self.prepare()

        self.assertEqual(prepared.workspace.parent, prepared.hidden_grader.parent)
        self.assertFalse(prepared.hidden_grader.is_relative_to(prepared.workspace))
        self.assertFalse(hidden_grader_changed(prepared))

        hidden_test = prepared.hidden_grader / "test_hidden.py"
        hidden_test.write_text(
            hidden_test.read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )
        self.assertTrue(hidden_grader_changed(prepared))

    def test_external_grader_fails_seed_and_passes_correct_implementation(self):
        prepared = self.prepare()

        self.assertFalse(run_hidden_grader(prepared).passed)
        (prepared.workspace / "text_utils.py").write_text(
            "import re\n\n"
            "def slugify(value: str) -> str:\n"
            "    return re.sub(r'\\s+', '-', value.strip().lower())\n",
            encoding="utf-8",
        )

        self.assertTrue(run_hidden_grader(prepared).passed)

    def test_every_hidden_grader_accepts_its_reference_change(self):
        changes = {
            "single_file_slugify": {
                "text_utils.py": (
                    "import re\n\n"
                    "def slugify(value: str) -> str:\n"
                    "    return re.sub(r'\\s+', '-', value.strip().lower())\n"
                )
            },
            "multi_file_currency": {
                "config.py": 'CURRENCY = "EUR"\n',
                "formatter.py": (
                    "import config\n\n"
                    "def format_price(amount: float) -> str:\n"
                    "    return f\"{config.CURRENCY} {amount:.2f}\"\n"
                ),
            },
            "duplicate_edit_anchor": {
                "status.py": (
                    "def is_ready(state: str) -> bool:\n"
                    "    if state == \"ready\":\n"
                    "        return True\n"
                    "    return False\n\n"
                    "def is_closed(state: str) -> bool:\n"
                    "    if state == \"closed\":\n"
                    "        return False\n"
                    "    return False\n"
                )
            },
            "wrong_path_recovery": {
                "src/settings.py": "TIMEOUT_SECONDS = 30\n"
            },
            "verified_palindrome": {
                "palindrome.py": (
                    "def is_palindrome(value: str) -> bool:\n"
                    "    normalized = ''.join(\n"
                    "        char.lower() for char in value if char.isalnum()\n"
                    "    )\n"
                    "    return normalized == normalized[::-1]\n"
                )
            },
        }
        cases = load_cases(DEFAULT_CASES)
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            for case in cases:
                with self.subTest(case=case.id):
                    prepared = prepare_case(
                        case,
                        fixtures_root=DEFAULT_FIXTURES,
                        results_root=results_root,
                        profile="reference",
                        repetition=1,
                    )
                    self.assertFalse(run_hidden_grader(prepared).passed)
                    for relative, content in changes[case.id].items():
                        target = prepared.workspace / relative
                        target.write_text(content, encoding="utf-8")
                    self.assertTrue(run_hidden_grader(prepared).passed)


class OfflineScenarioTest(unittest.TestCase):
    def test_controlled_faults_and_safety_invariants_are_deterministic(self):
        results = run_offline_scenarios()

        self.assertEqual(len(results), 8)
        self.assertTrue(all(result.fault_triggered for result in results))
        controlled = [
            result
            for result in results
            if result.category == "controlled_failure_recovery"
        ]
        reliable = [
            result for result in controlled if result.profile == "reliable"
        ]
        basic = [
            result
            for result in controlled
            if result.profile == "basic_ablation"
        ]
        self.assertEqual(len(reliable), 3)
        self.assertTrue(all(result.recovery_success for result in reliable))
        self.assertEqual(len(basic), 3)
        self.assertTrue(all(not result.recovery_success for result in basic))
        invariants = [
            result
            for result in results
            if result.category == "safety_invariant"
        ]
        self.assertTrue(all(result.invariant_passed for result in invariants))
        self.assertTrue(
            all(not result.side_effect_violation for result in invariants)
        )

    def test_report_has_three_separate_sections_and_attempt_breakdown(self):
        sample = EvalResult(
            category="real_coding",
            case_id="case",
            profile="reliable",
            verified_success=True,
            agent_returned=True,
            metrics=RunMetrics(
                main_model_attempts=2,
                goal_model_attempts=1,
                total_model_attempts=3,
                turns=2,
            ),
        )

        report = render_markdown([sample, *run_offline_scenarios()])

        self.assertIn("Real Coding: Basic vs Reliable", report)
        self.assertIn("Real Coding Per-run Results", report)
        self.assertIn("Controlled Failure Recovery", report)
        self.assertIn("Safety Invariants", report)
        self.assertIn("Avg main", report)
        self.assertIn("Avg goal", report)
        self.assertIn("premature_goal", report)

    def test_offline_cli_writes_json_and_markdown_without_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"

            exit_code = main(
                ["--suite", "offline", "--results-dir", str(output)]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "report.md").is_file())
            payload = json.loads(
                (output / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["results"]), 8)


class ProfileComparisonTest(unittest.TestCase):
    def test_profiles_keep_tools_and_permissions_equal(self):
        case = load_cases(DEFAULT_CASES)[0]
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            with patch("evals.run.agent_loop", return_value="done") as loop:
                with patch(
                    "evals.run.run_hidden_grader",
                    return_value=GradeResult(True, 0),
                ):
                    with patch(
                        "evals.run.hidden_grader_changed",
                        return_value=False,
                    ):
                        run_real_case(
                            case,
                            profile="basic_ablation",
                            repetition=1,
                            provider=object(),
                            fixtures_root=DEFAULT_FIXTURES,
                            results_root=results_root,
                        )
                        basic = loop.call_args
                        run_real_case(
                            case,
                            profile="reliable",
                            repetition=1,
                            provider=object(),
                            fixtures_root=DEFAULT_FIXTURES,
                            results_root=results_root,
                        )
                        reliable = loop.call_args

        self.assertTrue(basic.kwargs["allow_subagent"])
        self.assertTrue(reliable.kwargs["allow_subagent"])
        self.assertIsNone(basic.kwargs["max_context_chars"])
        self.assertIsNone(reliable.kwargs["max_context_chars"])
        self.assertEqual(
            basic.kwargs["permission_policy"].allowed_bash,
            reliable.kwargs["permission_policy"].allowed_bash,
        )
        self.assertEqual(basic.kwargs["recovery_policy"].max_retries, 0)
        self.assertEqual(reliable.kwargs["recovery_policy"].max_retries, 2)
        self.assertIsNone(basic.kwargs["goal_condition"])
        self.assertEqual(reliable.kwargs["goal_condition"], case.goal)


if __name__ == "__main__":
    unittest.main()
