"""Thin subprocess wrapper around the official SWE-bench evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class OfficialEvaluationResult:
    run_id: str
    output_dir: Path
    returncode: int


def new_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def run_official_evaluation(
    predictions_path: Path,
    dataset_path: Path,
    results_root: Path,
    *,
    instance_ids: tuple[str, ...] = (),
    evaluation_run_id: str | None = None,
    timeout_seconds: float | None = None,
    split: str = "dev",
    local_images_only: bool = False,
) -> OfficialEvaluationResult:
    """Invoke the official harness without reproducing benchmark scoring."""

    if split not in {"dev", "test"}:
        raise ValueError("Expected dev or test split")
    run_id = evaluation_run_id or new_run_id("official")
    output_dir = (results_root / "official_evaluation" / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    argv = [
        sys.executable,
        "-m",
        "evals.swe_bench_lite.official_local" if local_images_only else "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset_path.resolve()),
        "--split",
        split,
        "--predictions_path",
        str(predictions_path.resolve()),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
    ]
    if instance_ids:
        argv.extend(["--instance_ids", *instance_ids])
    completed = subprocess.run(
        argv,
        shell=False,
        cwd=output_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
        **({"env": {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])
                    + os.pathsep + os.environ.get("PYTHONPATH", "")}} if local_images_only else {}),
    )
    (output_dir / "harness.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    result = OfficialEvaluationResult(run_id, output_dir, completed.returncode)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                **asdict(result),
                "output_dir": str(output_dir),
                "predictions_path": str(predictions_path.resolve()),
                "dataset_path": str(dataset_path.resolve()),
                "instance_ids": list(instance_ids),
                "split": split,
                "local_images_only": local_images_only,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Official SWE-bench evaluation failed with exit code {completed.returncode}; "
            f"see {output_dir / 'harness.log'}"
        )
    return result
