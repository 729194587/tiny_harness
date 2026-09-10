"""Evaluator-only reference calibration before any model rollout."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .data import SweEvaluationBundle, SweTask
from .docker_workspace import DockerTaskEnvironment

CALIBRATED = "CALIBRATED"
CALIBRATION_FAILED = "CALIBRATION_FAILED"


class CalibrationGrader(Protocol):
    def __call__(
        self,
        task: SweTask,
        bundle: SweEvaluationBundle,
        log_path: Path,
        label: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CalibrationResult:
    instance_id: str
    status: str
    baseline_ftp_failed: bool
    baseline_ptp_passed: bool
    gold_ftp_passed: bool
    gold_ptp_passed: bool
    error_type: str | None = None

    @property
    def calibrated(self) -> bool:
        return self.status == CALIBRATED


def official_log_grader(
    task: SweTask,
    bundle: SweEvaluationBundle,
    log_path: Path,
    label: str,
) -> Mapping[str, Any]:
    """Grade a captured official eval script log with SWE-bench itself."""

    try:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.utils import make_test_spec
    except ImportError as error:
        raise RuntimeError(
            "Reference calibration requires the official swebench package"
        ) from error

    record = bundle.official_record()
    record["image"] = task.image
    prediction = {
        "instance_id": task.instance_id,
        "model_name_or_path": f"reference-{label}",
        "model_patch": bundle.patch if label == "gold" else "baseline",
    }
    report = get_eval_report(
        test_spec=make_test_spec(record),
        prediction=prediction,
        test_log_path=str(log_path),
        include_tests_status=True,
    )
    return report[task.instance_id]


def _all_in(report: Mapping[str, Any], category: str, outcome: str, expected: tuple[str, ...]) -> bool:
    tests_status = report.get("tests_status")
    if not isinstance(tests_status, Mapping):
        return False
    section = tests_status.get(category)
    if not isinstance(section, Mapping):
        return False
    observed = section.get(outcome)
    return isinstance(observed, list) and set(expected).issubset(observed)


def _run_phase(
    task: SweTask,
    bundle: SweEvaluationBundle,
    output_dir: Path,
    label: str,
    environment_factory: Callable[[SweTask], DockerTaskEnvironment],
    grader: CalibrationGrader,
) -> Mapping[str, Any]:
    log_path = output_dir / f"{label}.log"
    with environment_factory(task) as environment:
        assert environment.workspace is not None
        private_dir = environment.workspace / ".tinyharness"
        private_dir.mkdir(parents=True, exist_ok=True)
        if label == "gold":
            gold_path = private_dir / "gold.patch"
            gold_path.write_text(bundle.patch, encoding="utf-8", newline="\n")
            environment.exec(
                "git apply --binary --whitespace=nowarn .tinyharness/gold.patch",
                check=True,
            )
        script_path = private_dir / "eval.sh"
        script_path.write_text(bundle.eval_script, encoding="utf-8", newline="\n")
        completed = environment.exec("bash .tinyharness/eval.sh", check=False)
        log_path.write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
            newline="\n",
        )
    return grader(task, bundle, log_path, label)


def calibrate_task(
    task: SweTask,
    bundle: SweEvaluationBundle,
    output_dir: Path,
    *,
    environment_factory: Callable[[SweTask], DockerTaskEnvironment] = DockerTaskEnvironment,
    grader: CalibrationGrader = official_log_grader,
) -> CalibrationResult:
    """Admit a task only when clean and gold oracle behavior both reproduce."""

    if task.instance_id != bundle.instance_id:
        raise ValueError("Task and evaluator bundle instance IDs do not match")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline = _run_phase(
            task, bundle, output_dir, "baseline", environment_factory, grader
        )
        gold = _run_phase(task, bundle, output_dir, "gold", environment_factory, grader)
        result = CalibrationResult(
            instance_id=task.instance_id,
            status=CALIBRATION_FAILED,
            baseline_ftp_failed=_all_in(
                baseline, "FAIL_TO_PASS", "failure", bundle.fail_to_pass
            ),
            baseline_ptp_passed=_all_in(
                baseline, "PASS_TO_PASS", "success", bundle.pass_to_pass
            ),
            gold_ftp_passed=_all_in(
                gold, "FAIL_TO_PASS", "success", bundle.fail_to_pass
            ),
            gold_ptp_passed=_all_in(
                gold, "PASS_TO_PASS", "success", bundle.pass_to_pass
            ),
        )
        if all(
            (
                result.baseline_ftp_failed,
                result.baseline_ptp_passed,
                result.gold_ftp_passed,
                result.gold_ptp_passed,
            )
        ):
            result = CalibrationResult(
                **{**asdict(result), "status": CALIBRATED}
            )
    except BaseException as error:
        result = CalibrationResult(
            task.instance_id,
            CALIBRATION_FAILED,
            False,
            False,
            False,
            False,
            type(error).__name__,
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    (output_dir / "result.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
