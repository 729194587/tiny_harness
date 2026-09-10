"""Leak-resistant loaders for agent and evaluator views of SWE instances."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweTask:
    """The complete and deliberately small data surface visible to an agent run."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    image: str


@dataclass(frozen=True)
class SweEvaluationBundle:
    """Evaluator-only oracle data; never accepted by the agent rollout function."""

    instance_id: str
    repo: str
    version: str
    base_commit: str
    patch: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    eval_script: str
    log_parser: str
    eval_type: str

    def official_record(self) -> dict[str, Any]:
        """Return the record shape consumed by SWE-bench's official grader."""

        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "version": self.version,
            "FAIL_TO_PASS": list(self.fail_to_pass),
            "PASS_TO_PASS": list(self.pass_to_pass),
            "eval_script": self.eval_script,
            "log_parser": self.log_parser,
            "eval_type": self.eval_type,
        }


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                records.append(value)
    return records


def _test_ids(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def load_agent_tasks(path: Path) -> list[SweTask]:
    """Load only fields permitted to cross into Agent execution."""

    return [
        SweTask(
            instance_id=str(row["instance_id"]),
            repo=str(row["repo"]),
            base_commit=str(row["base_commit"]),
            problem_statement=str(row["problem_statement"]),
            image=str(row["image"]),
        )
        for row in _records(path)
    ]


def load_evaluation_bundles(path: Path) -> list[SweEvaluationBundle]:
    """Load oracle fields on the evaluator side of the boundary."""

    return [
        SweEvaluationBundle(
            instance_id=str(row["instance_id"]),
            repo=str(row["repo"]),
            version=str(row["version"]),
            base_commit=str(row["base_commit"]),
            patch=str(row["patch"]),
            test_patch=str(row["test_patch"]),
            fail_to_pass=_test_ids(row["FAIL_TO_PASS"], "FAIL_TO_PASS"),
            pass_to_pass=_test_ids(row["PASS_TO_PASS"], "PASS_TO_PASS"),
            eval_script=str(row["eval_script"]),
            log_parser=str(row["log_parser"]),
            eval_type=str(row["eval_type"]),
        )
        for row in _records(path)
    ]
