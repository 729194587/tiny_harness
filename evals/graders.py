"""Fresh workspace preparation and model-independent external grading."""

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from evals.core import EvalCase


@dataclass(frozen=True)
class PreparedCase:
    run_root: Path
    workspace: Path
    hidden_grader: Path
    event_log: Path
    grader_digest_before: str


@dataclass(frozen=True)
class GradeResult:
    passed: bool
    exit_code: int


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
    shutil.copytree(source_workspace, workspace)
    shutil.copytree(source_grader, hidden_grader)
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
    """Run trusted hidden tests after the Agent has stopped."""

    environment = os.environ.copy()
    environment["TINYHARNESS_EVAL_WORKSPACE"] = str(prepared.workspace)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(prepared.hidden_grader),
            "-v",
        ],
        cwd=prepared.run_root,
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
    )
