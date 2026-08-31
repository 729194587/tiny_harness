"""Fresh workspace preparation and model-independent external grading."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.core import EvalCase
from tiny_harness.runtime.events import EventLogger, EventType


@dataclass(frozen=True)
class PreparedCase:
    run_root: Path
    workspace: Path
    hidden_grader: Path
    event_log: Path
    grader_digest_before: str


@dataclass(frozen=True)
class GradeResult:
    passed: bool | None
    exit_code: int | None
    valid: bool = True
    snapshot_digest: str | None = None


@dataclass(frozen=True)
class TerminalGrade:
    """External grade captured at the root Agent's natural final answer."""

    turn: int
    valid: bool
    passed: bool | None
    exit_code: int | None
    elapsed_ms: int
    snapshot_digest: str | None


SnapshotGrader = Callable[[PreparedCase], GradeResult]
POST_RUN_GRADE_FILENAME = "post_run_grade.json"


def directory_digest(directory: Path) -> str:
    """Hash relative names and bytes without following directory symlinks."""

    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        if path.is_symlink():
            digest.update(b"SYMLINK")
            digest.update(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            digest.update(path.read_bytes())
        elif path.is_dir():
            digest.update(b"DIRECTORY")
    return digest.hexdigest()


def prepare_case(
    case: EvalCase,
    *,
    fixtures_root: Path,
    results_root: Path,
    profile: str,
    repetition: int,
) -> PreparedCase:
    """Copy visible workspace and hidden grader into sibling directories."""

    fixture = fixtures_root / case.id
    source_workspace = fixture / "workspace"
    source_grader = fixture / "hidden_grader"
    if not source_workspace.is_dir() or not source_grader.is_dir():
        raise FileNotFoundError(f"Incomplete eval fixture: {fixture}")

    run_root = results_root / "runs" / f"{case.id}-{profile}-{repetition}"
    if run_root.exists():
        raise FileExistsError(f"Eval run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    workspace = run_root / "workspace"
    hidden_grader = run_root / "hidden_grader"
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(source_workspace, workspace, ignore=ignored)
    shutil.copytree(source_grader, hidden_grader, ignore=ignored)
    return PreparedCase(
        run_root=run_root,
        workspace=workspace,
        hidden_grader=hidden_grader,
        event_log=run_root / "events.jsonl",
        grader_digest_before=directory_digest(hidden_grader),
    )


def hidden_grader_changed(prepared: PreparedCase) -> bool:
    return directory_digest(prepared.hidden_grader) != prepared.grader_digest_before


def run_hidden_grader(
    prepared: PreparedCase,
    *,
    timeout_seconds: int = 30,
) -> GradeResult:
    """Grade a physical workspace copy and return metadata only."""

    try:
        if hidden_grader_changed(prepared):
            return GradeResult(None, None, valid=False)
        with tempfile.TemporaryDirectory(prefix="tinyharness-eval-grade-") as root:
            grading_root = Path(root)
            snapshot = grading_root / "workspace"
            grader = grading_root / "hidden_grader"
            ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
            for tree in (prepared.workspace, prepared.hidden_grader):
                if any(path.is_symlink() for path in tree.rglob("*")):
                    return GradeResult(None, None, valid=False)
            shutil.copytree(prepared.workspace, snapshot, ignore=ignored)
            shutil.copytree(prepared.hidden_grader, grader, ignore=ignored)
            snapshot_digest = directory_digest(snapshot)

            environment = os.environ.copy()
            environment["TINYHARNESS_EVAL_WORKSPACE"] = str(snapshot)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(grader),
                    "-v",
                ],
                cwd=grading_root,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            return GradeResult(
                passed=completed.returncode == 0,
                exit_code=completed.returncode,
                snapshot_digest=snapshot_digest,
            )
    except (OSError, shutil.Error, subprocess.SubprocessError):
        return GradeResult(None, None, valid=False)


def run_post_run_grade(prepared: PreparedCase) -> dict[str, Any]:
    """Grade final workspace state after the Agent has completely stopped."""

    started = time.monotonic()
    try:
        grade = run_hidden_grader(prepared)
    except Exception:
        grade = GradeResult(None, None, valid=False)
    payload = {
        "available": True,
        "valid": grade.valid,
        "passed": grade.passed if grade.valid else None,
        "exit_code": grade.exit_code if grade.valid else None,
        "snapshot_digest": grade.snapshot_digest,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }
    try:
        (prepared.run_root / POST_RUN_GRADE_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
    return payload


class FinalAnswerGradingEventLogger:
    """Synchronously grade a root final answer without publishing results."""

    def __init__(
        self,
        logger: EventLogger,
        prepared: PreparedCase,
        *,
        grader: SnapshotGrader | None = None,
    ) -> None:
        self._logger = logger
        self._prepared = prepared
        self._grader = grader or run_hidden_grader
        self._record: TerminalGrade | None = None

    @property
    def record(self) -> TerminalGrade | None:
        return self._record

    @property
    def invalid(self) -> bool:
        return self._record is not None and not self._record.valid

    def emit(
        self,
        event_type: EventType,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        event_data = dict(data or {})
        self._logger.emit(event_type, event_data)
        if event_data.get("agent_scope") is not None:
            return
        if event_type is EventType.RUN_FINISHED:
            started = time.monotonic()
            try:
                grade = self._grader(self._prepared)
            except Exception:
                grade = GradeResult(None, None, valid=False)
            self._record = TerminalGrade(
                turn=int(event_data.get("turns", 0)),
                valid=grade.valid,
                passed=grade.passed if grade.valid else None,
                exit_code=grade.exit_code if grade.valid else None,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                snapshot_digest=grade.snapshot_digest,
            )

    def write_record(self, path: Path) -> None:
        """Persist the sanitized terminal grade after the Agent run is over."""

        path.write_text(
            json.dumps(
                asdict(self._record) if self._record is not None else None,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
