import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.core import (
    EvalResult,
    RecordingEventLogger,
    RunMetrics,
    collect_metrics,
    load_cases,
)
from evals.graders import (
    GradeResult,
    PreparedCase,
    ProposalGradingEventLogger,
    directory_digest,
    hidden_grader_changed,
    prepare_case,
    run_hidden_grader,
)
from evals.run import (
    DEFAULT_CASES,
    DEFAULT_FIXTURES,
    EvalPermissionPolicy,
    balanced_profile_order,
    main,
    render_markdown,
    run_real_case,
)
from evals.scenarios import run_offline_scenarios
from tiny_harness.agent.loop import run_agent as agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.goal import GoalEvaluation
from tiny_harness.runtime.permissions import (
    PermissionDecision,
    permission_denial_feedback,
)


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        if not self.responses:
            raise AssertionError("ScriptedProvider has no response left")
        return self.responses.pop(0)


class RecordingGoalEvaluator:
    def __init__(self, outcomes, order=None):
        self.outcomes = list(outcomes)
        self.calls = []
        self.order = order

    def evaluate(self, condition, messages, candidate_answer):
        if self.order is not None:
            self.order.append("goal")
        self.calls.append(
            {
                "condition": condition,
                "messages": copy.deepcopy(messages),
                "candidate": candidate_answer,
            }
        )
        return self.outcomes.pop(0)


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

    def test_eval_permission_policy_allows_safe_unittest_variants(self):
        policy = EvalPermissionPolicy(
            ["python -m unittest discover -s tests -v"]
        )

        self.assertIs(
            policy.decide("read_file", {"path": "x"}),
            PermissionDecision.ALLOW,
        )
        commands = (
            "python -m unittest",
            "python -m unittest -v",
            "python -m unittest tests",
            "python -m unittest tests.test_models -q",
            "python -m unittest tests/test_models.py -v",
            r"python -m unittest tests\test_models.py -v",
            r'python -m unittest "tests\test_models.py" -v',
            "python -m unittest discover -s ./tests -p 'test_*.py' -v",
            r"python -m unittest discover -s tests\unit -v",
            "python3 -m unittest discover --start-directory=tests -v",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIs(
                    policy.decide("bash", {"command": command}),
                    PermissionDecision.ALLOW,
                )

    def test_eval_permission_policy_still_denies_unsafe_or_non_test_bash(self):
        policy = EvalPermissionPolicy(
            ["python -m unittest discover -s tests -v"]
        )
        commands = (
            "python -c 'print(1)'",
            "pytest -q",
            "python -m unittest discover -s ..",
            r"python -m unittest discover -s tests\..\secret -v",
            r"python -m unittest C:\tests\test_models.py -v",
            "python -m unittest os",
            "python -m unittest -v discover -s tests",
            "python -m unittest discover -s tests && del important.txt",
            "python -m unittest discover -s tests > results.txt",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIs(
                    policy.decide("bash", {"command": command}),
                    PermissionDecision.DENY,
                )

        guidance = policy.denial_guidance("bash", {"command": "pytest -q"})
        self.assertIn("Allowed verification path", guidance)
        self.assertNotIn("grader", guidance.casefold())

        feedback = permission_denial_feedback(
            policy,
            "bash",
            {"command": "pytest -q"},
        )
        self.assertIn("current shell command is not allowed", feedback)
        self.assertIn("python -m unittest", feedback)
        self.assertIn("provide the final answer", feedback)


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

    def test_grader_runs_on_physical_snapshot_and_discards_private_output(self):
        run_root = self.results_root / "snapshot-case"
        workspace = run_root / "workspace"
        hidden_grader = run_root / "hidden_grader"
        workspace.mkdir(parents=True)
        hidden_grader.mkdir()
        (workspace / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
        (hidden_grader / "test_hidden.py").write_text(
            "import os\n"
            "import unittest\n"
            "from pathlib import Path\n\n"
            "WORKSPACE = Path(os.environ['TINYHARNESS_EVAL_WORKSPACE'])\n\n"
            "class HiddenTest(unittest.TestCase):\n"
            "    def test_snapshot(self):\n"
            "        (WORKSPACE / 'grader-marker.txt').write_text('SECRET')\n"
            "        print('PRIVATE_GRADER_SENTINEL')\n"
            "        self.fail('PRIVATE_FAILURE_REASON')\n",
            encoding="utf-8",
        )
        prepared = PreparedCase(
            run_root=run_root,
            workspace=workspace,
            hidden_grader=hidden_grader,
            event_log=run_root / "events.jsonl",
            grader_digest_before=directory_digest(hidden_grader),
        )

        grade = run_hidden_grader(prepared)

        self.assertTrue(grade.valid)
        self.assertFalse(grade.passed)
        self.assertIsNotNone(grade.snapshot_digest)
        self.assertFalse((workspace / "grader-marker.txt").exists())
        self.assertNotIn("PRIVATE", json.dumps(grade.__dict__))

    def test_proposal_grading_precedes_gate_and_never_leaks(self):
        prepared = self.prepare()
        order = []

        def grade(_prepared):
            order.append("grade")
            return GradeResult(
                False,
                1,
                snapshot_digest="PRIVATE_GRADER_SENTINEL",
            )

        base_logger = RecordingEventLogger()
        logger = ProposalGradingEventLogger(
            base_logger,
            prepared,
            grader=grade,
        )
        provider = ScriptedProvider(
            [
                ModelResponse("premature", None, [], "stop"),
                ModelResponse("verified", None, [], "stop"),
            ]
        )
        evaluator = RecordingGoalEvaluator(
            [
                GoalEvaluation(False, "collect evidence"),
                GoalEvaluation(True, "verified"),
            ],
            order=order,
        )
        messages = [{"role": "user", "content": "task"}]

        answer = agent_loop(
            provider,
            prepared.workspace,
            messages,
            max_turns=2,
            event_logger=logger,
            goal_condition="tests pass",
            goal_evaluator=evaluator,
            inject_goal_context=False,
        )

        self.assertEqual(answer, "verified")
        self.assertEqual(order, ["grade", "goal", "grade", "goal"])
        self.assertEqual(len(logger.records), 2)
        self.assertEqual(
            [record.gate_action for record in logger.records],
            ["block", "allow"],
        )
        agent_input = json.dumps(
            [call["messages"] for call in provider.calls]
        )
        evaluator_input = json.dumps(evaluator.calls)
        self.assertNotIn("PRIVATE_GRADER_SENTINEL", agent_input)
        self.assertNotIn("PRIVATE_GRADER_SENTINEL", evaluator_input)
        self.assertNotIn("PRIVATE_GRADER_SENTINEL", json.dumps(messages))
        self.assertNotIn(
            "PRIVATE_GRADER_SENTINEL",
            json.dumps(base_logger.events),
        )

    def test_subagent_stop_is_not_task_level_proposal_grade(self):
        prepared = self.prepare()
        grade_calls = []

        def grade(_prepared):
            grade_calls.append("root")
            return GradeResult(True, 0)

        logger = ProposalGradingEventLogger(
            RecordingEventLogger(),
            prepared,
            grader=grade,
        )
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"child"}')],
                    "tool_calls",
                ),
                ModelResponse("child final", None, [], "stop"),
                ModelResponse("parent final", None, [], "stop"),
            ]
        )

        answer = agent_loop(
            provider,
            prepared.workspace,
            [{"role": "user", "content": "delegate"}],
            max_turns=2,
            event_logger=logger,
        )

        self.assertEqual(answer, "parent final")
        self.assertEqual(grade_calls, ["root"])
        self.assertEqual(len(logger.records), 1)
        self.assertEqual(logger.records[0].turn, 2)

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
            profile="goal_gated",
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

        self.assertIn("Real Coding: Baseline vs Goal Gated", report)
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
    def test_profiles_differ_only_by_goal_gate_configuration(self):
        case = load_cases(DEFAULT_CASES)[0]
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            with patch("evals.run.agent_loop", return_value="done") as loop:
                run_real_case(
                    case,
                    profile="baseline",
                    repetition=1,
                    provider=object(),
                    fixtures_root=DEFAULT_FIXTURES,
                    results_root=results_root,
                )
                baseline = loop.call_args
                run_real_case(
                    case,
                    profile="goal_gated",
                    repetition=1,
                    provider=object(),
                    fixtures_root=DEFAULT_FIXTURES,
                    results_root=results_root,
                )
                goal_gated = loop.call_args

        self.assertEqual(baseline.args[2], goal_gated.args[2])
        self.assertTrue(baseline.kwargs["allow_subagent"])
        self.assertTrue(goal_gated.kwargs["allow_subagent"])
        self.assertIsNone(baseline.kwargs["max_context_chars"])
        self.assertIsNone(goal_gated.kwargs["max_context_chars"])
        self.assertEqual(
            baseline.kwargs["permission_policy"].allowed_bash,
            goal_gated.kwargs["permission_policy"].allowed_bash,
        )
        baseline_config = dict(baseline.kwargs)
        goal_config = dict(goal_gated.kwargs)
        baseline_logger = baseline_config.pop("event_logger")
        goal_logger = goal_config.pop("event_logger")
        del baseline_logger, goal_logger
        baseline_permission = baseline_config.pop("permission_policy")
        goal_permission = goal_config.pop("permission_policy")
        self.assertEqual(
            baseline_permission.allowed_bash,
            goal_permission.allowed_bash,
        )
        self.assertIsNone(baseline_config.pop("goal_condition"))
        self.assertEqual(goal_config.pop("goal_condition"), case.goal)
        self.assertEqual(baseline_config, goal_config)
        self.assertEqual(
            baseline.kwargs["recovery_policy"].max_retries,
            goal_gated.kwargs["recovery_policy"].max_retries,
        )
        self.assertFalse(baseline.kwargs["inject_goal_context"])
        self.assertFalse(goal_gated.kwargs["inject_goal_context"])

    def test_first_main_provider_request_is_identical_across_profiles(self):
        case = load_cases(DEFAULT_CASES)[0]
        baseline_provider = ScriptedProvider(
            [ModelResponse("done", None, [], "stop")]
        )
        gated_provider = ScriptedProvider(
            [
                ModelResponse("done", None, [], "stop"),
                ModelResponse(
                    '{"ok":true,"reason":"verified","impossible":false}',
                    None,
                    [],
                    "stop",
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            with patch(
                "evals.graders.run_hidden_grader",
                return_value=GradeResult(False, 1),
            ):
                run_real_case(
                    case,
                    profile="baseline",
                    repetition=1,
                    provider=baseline_provider,
                    fixtures_root=DEFAULT_FIXTURES,
                    results_root=results_root,
                )
                run_real_case(
                    case,
                    profile="goal_gated",
                    repetition=1,
                    provider=gated_provider,
                    fixtures_root=DEFAULT_FIXTURES,
                    results_root=results_root,
                )

        self.assertEqual(
            baseline_provider.calls[0]["messages"],
            gated_provider.calls[0]["messages"],
        )
        self.assertEqual(
            baseline_provider.calls[0]["tools"],
            gated_provider.calls[0]["tools"],
        )
        system_prompt = gated_provider.calls[0]["messages"][0]["content"]
        self.assertIn("Verify your changes when appropriate.", system_prompt)
        self.assertNotIn("declared verification command", system_prompt)
        self.assertNotIn("direct tool results", system_prompt)
        self.assertNotIn("unsupported completion claims", system_prompt)
        serialized = json.dumps(gated_provider.calls[0]["messages"])
        self.assertNotIn("tinyharness_goal_state", serialized)
        self.assertNotIn(case.goal, serialized)

    def test_profile_order_is_cross_balanced(self):
        profiles = ("baseline", "goal_gated")

        self.assertNotEqual(
            balanced_profile_order(profiles, case_index=0, repetition=1),
            balanced_profile_order(profiles, case_index=0, repetition=2),
        )
        self.assertNotEqual(
            balanced_profile_order(profiles, case_index=0, repetition=1),
            balanced_profile_order(profiles, case_index=1, repetition=1),
        )

    def test_grading_error_marks_invalid_run_instead_of_failure(self):
        case = load_cases(DEFAULT_CASES)[0]
        provider = ScriptedProvider(
            [ModelResponse("done", None, [], "stop")]
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "evals.graders.run_hidden_grader",
                side_effect=RuntimeError("PRIVATE_GRADING_ERROR"),
            ):
                result = run_real_case(
                    case,
                    profile="baseline",
                    repetition=1,
                    provider=provider,
                    fixtures_root=DEFAULT_FIXTURES,
                    results_root=Path(directory),
                )

        self.assertTrue(result.invalid_run)
        self.assertFalse(result.verified_success)
        self.assertFalse(result.false_success)
        self.assertTrue(result.agent_returned)


if __name__ == "__main__":
    unittest.main()
