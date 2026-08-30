import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.core import EvalResult, RunMetrics, load_cases
from evals.graders import GradeResult, POST_RUN_GRADE_FILENAME
from evals.pilot import (
    PILOT_CASES,
    PILOT_FIXTURES,
    main,
    run_pilot,
    select_cases,
)
from evals.pilot_metrics import (
    aggregate_runs,
    count_tool_denied,
    enrich_proposals,
    write_run_ledger,
)
from evals.run import DEFAULT_CASES, DEFAULT_FIXTURES, run_real_case
from tiny_harness.agent.messages import ModelResponse
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.errors import MaxTurnsExceededError


def proposal(
    passed,
    action,
    *,
    continued=False,
    valid=True,
    number=1,
):
    return {
        "proposal": number,
        "turn": number,
        "valid": valid,
        "passed": passed if valid else None,
        "exit_code": 0 if passed else (1 if valid else None),
        "snapshot_digest": f"digest-{number}" if valid else None,
        "gate_action": action,
        "agent_continued": continued,
        "external_grade": (
            "PASS" if passed is True else "FAIL" if passed is False else "INVALID"
        ),
    }


def ledger(
    case_id,
    profile,
    outcome,
    proposals,
    *,
    useful=False,
    attempts=1,
    goal_calls=0,
    turns=1,
    tools=0,
    wall=100,
):
    return {
        "schema_version": 1,
        "case_id": case_id,
        "profile": profile,
        "repetition": 1,
        "outcome": outcome,
        "verified_terminal_completion": outcome
        == "verified_terminal_completion",
        "premature_terminal_completion": outcome
        == "premature_terminal_completion",
        "explicit_failure": outcome == "explicit_failure",
        "invalid_run": outcome == "invalid_run",
        "invalid_reason": "GradingTimeout" if outcome == "invalid_run" else None,
        "error_type": None,
        "agent_returned": outcome
        in {"verified_terminal_completion", "premature_terminal_completion"},
        "main_model_attempts": attempts,
        "goal_evaluator_calls": goal_calls,
        "logical_turns": turns,
        "tool_calls": tools,
        "wall_time_ms": wall,
        "useful_intervention": useful,
        "final_grade": {},
        "proposals": proposals,
    }


class SyntheticPilotMetricsTest(unittest.TestCase):
    def setUp(self):
        self.runs = [
            ledger(
                "baseline-pass",
                "baseline",
                "verified_terminal_completion",
                [proposal(True, "allow")],
            ),
            ledger(
                "baseline-fail",
                "baseline",
                "premature_terminal_completion",
                [proposal(False, "allow")],
            ),
            ledger(
                "useful",
                "goal_gated",
                "verified_terminal_completion",
                [
                    proposal(False, "block", continued=True, number=1),
                    proposal(True, "allow", number=2),
                ],
                # Aggregation must derive this from primitive observations.
                useful=False,
                attempts=3,
                goal_calls=2,
                turns=2,
                tools=1,
                wall=300,
            ),
            ledger(
                "false-accept",
                "goal_gated",
                "premature_terminal_completion",
                [proposal(False, "allow")],
                goal_calls=1,
            ),
            ledger(
                "unnecessary-block",
                "goal_gated",
                "verified_terminal_completion",
                [
                    proposal(True, "block", continued=True, number=1),
                    proposal(True, "allow", number=2),
                ],
                goal_calls=2,
            ),
            ledger(
                "blocked-no-recovery",
                "goal_gated",
                "explicit_failure",
                [proposal(False, "block", continued=True)],
                goal_calls=1,
            ),
            ledger(
                "invalid-grader",
                "goal_gated",
                "invalid_run",
                [proposal(False, "block", continued=True)],
                useful=True,
                attempts=9,
                goal_calls=9,
                turns=9,
                tools=9,
                wall=900,
            ),
        ]

    def test_run_rates_and_invalid_exclusion(self):
        summary = aggregate_runs(self.runs)
        overall = summary["overall"]

        self.assertEqual(overall["runs"]["total"], 7)
        self.assertEqual(overall["runs"]["valid"], 6)
        self.assertEqual(overall["runs"]["invalid"], 1)
        self.assertEqual(overall["runs"]["verified_terminal_completion"], 3)
        self.assertEqual(overall["runs"]["premature_terminal_completion"], 2)
        self.assertEqual(overall["runs"]["explicit_failure"], 1)
        self.assertEqual(overall["run_rates"]["verified_completion_rate"], 0.5)
        self.assertAlmostEqual(
            overall["run_rates"]["premature_terminal_completion_rate"],
            2 / 6,
        )
        self.assertEqual(len(summary["invalid_runs"]), 1)

    def test_stop_metrics_confusion_matrix_and_useful_intervention(self):
        summary = aggregate_runs(self.runs)["overall"]

        # Baseline FAIL is included here, but the invalid run is excluded.
        self.assertEqual(summary["stop_proposals"]["premature"], 4)
        gate = summary["goal_gate"]
        self.assertEqual(gate["correct_block"], 2)
        self.assertEqual(gate["false_accept"], 1)
        self.assertEqual(gate["unnecessary_block"], 1)
        self.assertEqual(gate["correct_allow"], 2)
        self.assertAlmostEqual(gate["correct_block_rate"], 2 / 3)
        self.assertAlmostEqual(gate["false_accept_rate"], 1 / 3)
        self.assertAlmostEqual(gate["unnecessary_block_rate"], 1 / 3)

        useful = summary["useful_intervention"]
        self.assertEqual(useful["runs"], 1)
        self.assertEqual(useful["runs_with_correct_block"], 2)
        self.assertEqual(useful["recovery_after_correct_block_rate"], 0.5)

    def test_invalid_cost_is_reported_but_not_used_as_an_outcome(self):
        summary = aggregate_runs(self.runs)["overall"]

        # Resource cost includes invalid runs so infrastructure cost is visible.
        self.assertEqual(summary["resource_usage"]["total_main_model_attempts"], 17)
        self.assertEqual(summary["resource_usage"]["total_goal_evaluator_calls"], 15)
        self.assertEqual(summary["goal_gate"]["correct_block"], 2)
        self.assertEqual(summary["useful_intervention"]["runs"], 1)

    def test_useful_intervention_requires_all_three_conditions(self):
        variants = [
            ledger(
                "not-blocked",
                "goal_gated",
                "verified_terminal_completion",
                [proposal(False, "allow", continued=True)],
                useful=False,
            ),
            ledger(
                "not-continued",
                "goal_gated",
                "verified_terminal_completion",
                [
                    proposal(False, "block", continued=False),
                    proposal(True, "allow", number=2),
                ],
                useful=False,
            ),
            ledger(
                "not-recovered",
                "goal_gated",
                "premature_terminal_completion",
                [
                    proposal(False, "block", continued=True),
                    proposal(False, "allow", number=2),
                ],
                useful=False,
            ),
        ]

        summary = aggregate_runs(variants)["overall"]

        self.assertEqual(summary["useful_intervention"]["runs"], 0)

    def test_useful_intervention_does_not_trust_cached_ledger_flag(self):
        qualifying = ledger(
            "qualifying",
            "goal_gated",
            "verified_terminal_completion",
            [
                proposal(False, "block", continued=True, number=1),
                proposal(True, "allow", number=2),
            ],
            useful=False,
        )

        summary = aggregate_runs([qualifying])["overall"]

        self.assertEqual(summary["useful_intervention"]["runs"], 1)

    def test_diagnostics_do_not_change_run_outcomes(self):
        verified_with_post_fail = ledger(
            "verified",
            "baseline",
            "verified_terminal_completion",
            [proposal(True, "allow")],
        )
        verified_with_post_fail["post_run_grade"] = {
            "available": True,
            "valid": True,
            "passed": False,
        }
        explicit_with_post_pass = ledger(
            "explicit",
            "goal_gated",
            "explicit_failure",
            [],
        )
        explicit_with_post_pass["post_run_grade"] = {
            "available": True,
            "valid": True,
            "passed": True,
        }

        summary = aggregate_runs(
            [verified_with_post_fail, explicit_with_post_pass]
        )["overall"]

        self.assertEqual(summary["runs"]["verified_terminal_completion"], 1)
        self.assertEqual(summary["runs"]["explicit_failure"], 1)
        post = summary["diagnostics"]["post_run_grade"]
        self.assertEqual(post["pass"], 1)
        self.assertEqual(post["fail"], 1)
        explicit_post = summary["diagnostics"][
            "explicit_failure_post_run_grade"
        ]
        self.assertEqual(explicit_post["pass"], 1)


class PilotLedgerAndCliTest(unittest.TestCase):
    def test_continuation_is_derived_from_root_event_order(self):
        proposals = [
            {"proposal": 1, "turn": 1, "valid": True, "passed": False},
            {"proposal": 2, "turn": 2, "valid": True, "passed": True},
        ]
        events = [
            {"event_type": "stop_proposed", "data": {"turn": 1}},
            {
                "event_type": "model_requested",
                "data": {"purpose": "main", "agent_scope": "subagent"},
            },
            {
                "event_type": "model_requested",
                "data": {"purpose": "goal_evaluation"},
            },
            {"event_type": "stop_decided", "data": {"action": "block"}},
            {
                "event_type": "model_requested",
                "data": {"purpose": "main", "turn": 2},
            },
            {"event_type": "stop_proposed", "data": {"turn": 2}},
            {"event_type": "stop_decided", "data": {"action": "allow"}},
        ]

        enriched = enrich_proposals(proposals, events)

        self.assertTrue(enriched[0]["agent_continued"])
        self.assertFalse(enriched[1]["agent_continued"])

    def test_tool_denied_is_counted_offline_by_scope_and_category(self):
        events = [
            {"event_type": "tool_denied", "data": {"tool_name": "bash"}},
            {
                "event_type": "tool_denied",
                "data": {"tool_name": "edit_file"},
            },
            {
                "event_type": "tool_denied",
                "data": {"tool_name": "bash", "agent_scope": "subagent"},
            },
            {
                "event_type": "tool_denied",
                "data": {
                    "tool_name": "read_file",
                    "agent_scope": "subagent",
                },
            },
            {"event_type": "tool_started", "data": {"tool_name": "bash"}},
        ]

        denied = count_tool_denied(events)

        self.assertEqual(denied["total"], 4)
        self.assertEqual(denied["root"], {"total": 2, "bash": 1, "other": 1})
        self.assertEqual(
            denied["subagent"],
            {"total": 2, "bash": 1, "other": 1},
        )

    def test_run_ledger_and_offline_summarize_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            run_root = results_root / "runs" / "case-baseline-1"
            run_root.mkdir(parents=True)
            (run_root / "proposal_grades.json").write_text(
                json.dumps(
                    [
                        {
                            "proposal": 1,
                            "turn": 1,
                            "valid": True,
                            "passed": True,
                            "exit_code": 0,
                            "elapsed_ms": 1,
                            "snapshot_digest": "digest",
                            "gate_action": "allow",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (run_root / "events.jsonl").write_text(
                json.dumps(
                    {
                        "event_type": "stop_proposed",
                        "data": {"turn": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_root / POST_RUN_GRADE_FILENAME).write_text(
                json.dumps(
                    {
                        "available": True,
                        "valid": True,
                        "passed": False,
                        "exit_code": 1,
                        "snapshot_digest": "post-digest",
                        "elapsed_ms": 2,
                    }
                ),
                encoding="utf-8",
            )
            result = EvalResult(
                category="real_coding",
                case_id="case",
                profile="baseline",
                verified_success=True,
                agent_returned=True,
                elapsed_ms=123,
                metrics=RunMetrics(
                    main_model_attempts=2,
                    goal_model_attempts=0,
                    turns=1,
                    tool_calls=1,
                ),
            )

            run = write_run_ledger(result, run_root)
            with (run_root / "events.jsonl").open("a", encoding="utf-8") as events:
                events.write(
                    json.dumps(
                        {
                            "event_type": "tool_denied",
                            "data": {
                                "tool_name": "bash",
                                "agent_scope": "subagent",
                            },
                        }
                    )
                    + "\n"
                )
            exit_code = main(["summarize", str(results_root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(run["outcome"], "verified_terminal_completion")
            self.assertTrue((run_root / "run.json").is_file())
            self.assertTrue((run_root / "final_grade.json").is_file())
            self.assertTrue((run_root / POST_RUN_GRADE_FILENAME).is_file())
            self.assertTrue((results_root / "summary.json").is_file())
            self.assertTrue((results_root / "summary.md").is_file())
            self.assertTrue((results_root / "runs.csv").is_file())
            summary = json.loads(
                (results_root / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["overall"]["runs"]["verified_terminal_completion"],
                1,
            )
            self.assertEqual(
                summary["overall"]["diagnostics"]["tool_denied"][
                    "subagent"
                ]["bash"],
                1,
            )
            self.assertEqual(
                summary["overall"]["diagnostics"]["post_run_grade"]["fail"],
                1,
            )

    def test_task_selection_supports_one_or_all_tasks(self):
        all_cases = select_cases(None)
        selected = select_cases([all_cases[2].id])

        self.assertEqual(len(all_cases), 6)
        self.assertEqual([case.id for case in selected], [all_cases[2].id])
        with self.assertRaises(ValueError):
            select_cases(["missing-task"])

    def test_single_task_single_profile_runner_writes_all_artifacts(self):
        class ImmediateStopProvider:
            def complete(self, messages, tools):
                del messages, tools
                return ModelResponse("done", None, [], "stop")

        case = load_cases(PILOT_CASES)[0]
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)

            runs = run_pilot(
                cases=[case],
                profiles=["baseline"],
                repetitions=1,
                results_root=results_root,
                provider_factory=ImmediateStopProvider,
            )

            run_root = (
                results_root / "runs" / f"{case.id}-baseline-1"
            )
            self.assertEqual(len(runs), 1)
            self.assertEqual(
                runs[0]["outcome"],
                "premature_terminal_completion",
            )
            for name in (
                "run.json",
                "proposal_grades.json",
                "final_grade.json",
                POST_RUN_GRADE_FILENAME,
            ):
                self.assertTrue((run_root / name).is_file())
            for name in ("summary.json", "summary.md", "runs.csv"):
                self.assertTrue((results_root / name).is_file())


class FailingProvider:
    def complete(self, messages, tools):
        del messages, tools
        raise ModelProviderError(ModelErrorKind.FATAL)


class PilotInfrastructureClassificationTest(unittest.TestCase):
    def test_provider_error_is_invalid_not_explicit_or_premature(self):
        case = load_cases(PILOT_CASES)[0]
        with tempfile.TemporaryDirectory() as directory:
            result = run_real_case(
                case,
                profile="baseline",
                repetition=1,
                provider=FailingProvider(),
                fixtures_root=PILOT_FIXTURES,
                results_root=Path(directory),
            )

        self.assertTrue(result.invalid_run)
        self.assertFalse(result.explicit_failure)
        self.assertFalse(result.false_success)
        self.assertFalse(result.verified_success)


class OrderedProvider:
    def __init__(self, order):
        self.order = order
        self.calls = []
        self.responses = [
            ModelResponse("candidate", None, [], "stop"),
            ModelResponse(
                '{"ok":true,"reason":"done","impossible":false}',
                None,
                [],
                "stop",
            ),
        ]

    def complete(self, messages, tools):
        self.order.append("model")
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        return self.responses.pop(0)


class PostRunDiagnosticIsolationTest(unittest.TestCase):
    def test_post_run_pass_does_not_change_premature_outcome_or_leak(self):
        order = []
        provider = OrderedProvider(order)
        grades = [
            GradeResult(False, 1, snapshot_digest="PROPOSAL_SENTINEL"),
            GradeResult(True, 0, snapshot_digest="POST_RUN_SENTINEL"),
        ]

        def grade(_prepared):
            number = 2 - len(grades) + 1
            order.append(f"grade-{number}")
            return grades.pop(0)

        case = load_cases(PILOT_CASES)[0]
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            with patch("evals.graders.run_hidden_grader", side_effect=grade):
                result = run_real_case(
                    case,
                    profile="goal_gated",
                    repetition=1,
                    provider=provider,
                    fixtures_root=PILOT_FIXTURES,
                    results_root=results_root,
                )
            run_root = (
                results_root / "runs" / f"{case.id}-goal_gated-1"
            )
            run = write_run_ledger(result, run_root)

            self.assertEqual(order, ["model", "grade-1", "model", "grade-2"])
            self.assertTrue(result.false_success)
            self.assertFalse(result.verified_success)
            self.assertEqual(run["outcome"], "premature_terminal_completion")
            self.assertTrue(run["post_run_grade"]["passed"])
            serialized_calls = json.dumps(provider.calls)
            serialized_events = (run_root / "events.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("POST_RUN_SENTINEL", serialized_calls)
            self.assertNotIn("PROPOSAL_SENTINEL", serialized_calls)
            self.assertNotIn("POST_RUN_SENTINEL", serialized_events)
            self.assertNotIn("PROPOSAL_SENTINEL", serialized_events)

    def test_post_run_grading_error_does_not_invalidate_verified_outcome(self):
        class ImmediateStopProvider:
            def complete(self, messages, tools):
                del messages, tools
                return ModelResponse("done", None, [], "stop")

        case = load_cases(PILOT_CASES)[0]
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            with patch(
                "evals.graders.run_hidden_grader",
                side_effect=[
                    GradeResult(True, 0, snapshot_digest="proposal-pass"),
                    RuntimeError("PRIVATE_POST_RUN_ERROR"),
                ],
            ):
                result = run_real_case(
                    case,
                    profile="baseline",
                    repetition=1,
                    provider=ImmediateStopProvider(),
                    fixtures_root=PILOT_FIXTURES,
                    results_root=results_root,
                )
            run_root = (
                results_root / "runs" / f"{case.id}-baseline-1"
            )
            run = write_run_ledger(result, run_root)

        self.assertTrue(result.verified_success)
        self.assertFalse(result.invalid_run)
        self.assertEqual(run["outcome"], "verified_terminal_completion")
        self.assertTrue(run["post_run_grade"]["available"])
        self.assertFalse(run["post_run_grade"]["valid"])
        self.assertIsNone(run["post_run_grade"]["passed"])

    def test_explicit_failure_can_have_post_run_pass_without_terminal_proposal(self):
        case = load_cases(DEFAULT_CASES)[0]

        def finish_workspace_then_exhaust(provider, workspace, messages, **kwargs):
            del provider, messages, kwargs
            (workspace / "text_utils.py").write_text(
                "import re\n\n"
                "def slugify(value: str) -> str:\n"
                "    return re.sub(r'\\s+', '-', value.strip().lower())\n",
                encoding="utf-8",
            )
            raise MaxTurnsExceededError("Maximum model turns reached: 12")

        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            with patch("evals.run.agent_loop", side_effect=finish_workspace_then_exhaust):
                result = run_real_case(
                    case,
                    profile="baseline",
                    repetition=1,
                    provider=object(),
                    fixtures_root=DEFAULT_FIXTURES,
                    results_root=results_root,
                )
            run_root = (
                results_root / "runs" / f"{case.id}-baseline-1"
            )
            run = write_run_ledger(result, run_root)

        self.assertTrue(result.explicit_failure)
        self.assertFalse(result.invalid_run)
        self.assertEqual(run["outcome"], "explicit_failure")
        self.assertEqual(run["proposals"], [])
        self.assertTrue(run["post_run_grade"]["valid"])
        self.assertTrue(run["post_run_grade"]["passed"])


if __name__ == "__main__":
    unittest.main()
