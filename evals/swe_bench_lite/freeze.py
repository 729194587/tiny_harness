"""Download pinned official data and freeze a deterministic Test subset.

No model, Docker or grading entrypoints are imported. Only ``agent/`` may be
given to an agent; the remainder of a bundle is evaluator/controller-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter
from importlib.metadata import version
from pathlib import Path

DATASET = "SWE-bench/SWE-bench_Lite"
SEED = "20261015"
POPULATION = 300
SAMPLE = 60
AGENT_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement", "image")
STRING_FIELDS = (*AGENT_FIELDS, "version", "patch", "test_patch", "eval_script",
                 "log_parser", "eval_type", "environment_setup_commit", "created_at",
                 "hints_text", "difficulty")
TEST_FIELDS = ("FAIL_TO_PASS", "PASS_TO_PASS")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                    for row in rows)


def revision_value(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("An exact 40-character dataset revision is required")
    return value


def fetch_source(revision: str, output: Path) -> None:
    """Fetch only official dataset metadata and Test parquet, never overwrite."""
    revision_value(revision)
    output.mkdir(parents=True, exist_ok=False)

    def fetch(url):
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    api_url = f"https://huggingface.co/api/datasets/{DATASET}/revision/{revision}"
    metadata = fetch(api_url)
    info = json.loads(metadata)
    if info["sha"] != revision or info["id"] != DATASET:
        raise ValueError("Official metadata identity does not match the pin")
    shards = sorted(item["rfilename"] for item in info["siblings"]
                    if re.fullmatch(r"data/test-\d+-of-\d+\.parquet", item["rfilename"]))
    if not shards:
        raise ValueError("No official Test parquet shards")
    files = {"metadata.json": metadata}
    urls = {"metadata.json": api_url}
    for relative in ["README.md", *shards]:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{relative}"
        files[relative] = fetch(url)
        urls[relative] = url
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (output / "source.json").write_bytes(json_bytes({
        "dataset": DATASET, "revision": revision, "split": "test",
        "parquet_files": shards,
        "files": {name: {"sha256": digest(content), "url": urls[name]}
                  for name, content in sorted(files.items())},
    }))


def load_source(source: Path):
    import pyarrow.parquet as pq  # supplied by the evaluation environment

    if source.is_symlink():
        raise ValueError("Source symlinks are not supported")
    descriptor = json.loads((source / "source.json").read_bytes())
    revision_value(descriptor["revision"])
    if descriptor["dataset"] != DATASET or descriptor["split"] != "test":
        raise ValueError("Expected the official Lite Test source")
    files = {"source.json": (source / "source.json").read_bytes()}
    for name, record in descriptor["files"].items():
        path = source / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or path.is_symlink() or not path.resolve().is_relative_to(source.resolve())):
            raise ValueError("Unsafe source artifact path")
        content = path.read_bytes()
        if digest(content) != record["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
        files[name] = content
    metadata = json.loads(files["metadata.json"])
    if metadata["id"] != DATASET or metadata["sha"] != descriptor["revision"]:
        raise ValueError("Metadata identity mismatch")
    expected_shards = sorted(item["rfilename"] for item in metadata["siblings"]
                             if re.fullmatch(r"data/test-\d+-of-\d+\.parquet", item["rfilename"]))
    if not expected_shards or descriptor["parquet_files"] != expected_shards:
        raise ValueError("Test shard inventory mismatch")
    if set(descriptor["files"]) != {"metadata.json", "README.md", *expected_shards}:
        raise ValueError("Unexpected source artifact inventory")
    rows = []
    schemas = []
    for name in expected_shards:
        table = pq.read_table(io.BytesIO(files[name]))
        schemas.append(str(table.schema))
        rows.extend(table.to_pylist())
    validate_rows(rows)
    return descriptor, rows, files, schemas


def validate_rows(rows) -> None:
    if len(rows) != POPULATION:
        raise ValueError("Expected exactly 300 Test tasks")
    ids = []
    for row in rows:
        for field in STRING_FIELDS:
            if not isinstance(row.get(field), str):
                raise ValueError(f"Expected string field: {field}")
        for field in AGENT_FIELDS:
            if not row[field].strip():
                raise ValueError(f"Empty required field: {field}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise ValueError("Invalid base commit")
        for field in TEST_FIELDS:
            value = row.get(field)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"Expected list of test IDs: {field}")
        ids.append(row["instance_id"])
    if len(set(ids)) != POPULATION:
        raise ValueError("Duplicate Test instance IDs")


def select_rows(rows):
    """Membership depends ONLY on repo, instance ID and fixed seed."""
    counts = Counter(row["repo"] for row in rows)
    if len(counts) > SAMPLE:
        raise ValueError("Cannot cover every repository")
    quotas = {repo: 1 for repo in sorted(counts)}
    while sum(quotas.values()) < SAMPLE:
        # Integer deficits avoid platform-dependent floating point tie handling.
        repo = min((r for r in counts if quotas[r] < counts[r]),
                   key=lambda r: (-(SAMPLE * counts[r] - POPULATION * quotas[r]), r))
        quotas[repo] += 1
    ranks = {}
    selected = set()
    for repo in sorted(counts):
        candidates = [row for row in rows if row["repo"] == repo]
        candidates.sort(key=lambda row: (digest(
            f"{SEED}\0{repo}\0{row['instance_id']}".encode("utf-8")), row["instance_id"]))
        for rank, row in enumerate(candidates, 1):
            ranks[row["instance_id"]] = rank
            if rank <= quotas[repo]:
                selected.add(row["instance_id"])
    return [(index, row) for index, row in enumerate(rows) if row["instance_id"] in selected], quotas, ranks


def protocol():
    return {
        "protocol_version": 1, "benchmark_label": "SWE-bench Lite Test stratified subset (60/300)",
        "seed": SEED, "population": POPULATION, "sample_size": SAMPLE,
        "sampling": {
            "inputs": ["repo", "instance_id", "original row order"],
            "allocation": "Start each repo at 1; repeatedly increment the non-full repo with largest 60*n_r-300*q_r; ties by repo ascending",
            "ranking": "Ascending SHA256(UTF8(seed + NUL + repo + NUL + instance_id)); ties by instance_id",
            "selection_and_execution_order": "Original parquet shard order (shards sorted by filename), then row order; source_row is zero-based",
            "prohibited_inputs": ["difficulty", "gold patch", "test patch", "test counts", "historical results", "image cache", "pilot outcomes"],
            "replacement": "Never replace or drop tasks because of outcomes, difficulty, unavailable images, calibration, service or grading failures",
            "interpretation": "Repo coverage imposes unequal inclusion probabilities; report this subset, never a full 300-task score",
        },
        "isolation": {
            "agent_directory": "agent", "agent_allowlist": list(AGENT_FIELDS),
            "controller_only": ["protocol.json", "manifest.json", "selection.csv", "SHA256SUMS"],
            "evaluator_only_directory": "evaluator",
            "mount_policy": "Only agent/agent_tasks.jsonl may cross the controller-to-agent input boundary. Never mount the bundle, evaluator data, calibration outputs or answer artifacts in a task workspace",
            "limit": "Allowlisting removes evaluator fields; original public issue text is preserved verbatim and may itself contain solution hints. This is data isolation, not an OS sandbox or a model contamination guarantee",
        },
        "paired_design": {
            "profiles": {"A": {"code_outline": False, "new_search_filtering": False},
                         "B": {"code_outline": True, "new_search_filtering": True}},
            "pairing_key": "instance_id", "attempts_per_profile": 1,
            "profile_order": "Alternate A,B and B,A by zero-based selection_index parity",
            "controls": ["same pinned model/version and endpoint", "same prompts, skills, recovery and turn/context budgets", "same image digest and fresh base workspace", "memory disabled", "profile applies to parent and children", "identical safety/path constraints"],
            "filter_scope": "Includes automatic ignored-directory behavior and new search parameters; hiding a schema argument alone does not disable filtering",
            "analysis": "Report paired wins/losses, both-resolved and neither-resolved; all-task costs plus costs on common successes",
            "execution_gate": "Model configuration, tool profiles, image digests and Harness source must be frozen in a separately approved phase-2 execution manifest; not yet runnable",
        },
        "accounting": {
            "primary_denominator": 60, "primary_score": "resolved / 60 per profile; pending runs are provisional",
            "categories": ["resolved", "unresolved", "infrastructure_failure", "official_evaluation_error", "harness_or_protocol_failure", "pending_or_not_run"],
            "category_definitions": {
                "resolved": "Explicit official resolved verdict bound to the prediction and evaluation fingerprints",
                "unresolved": "Valid non-resolved task outcome, including an empty patch; retain subtype and raw official verdict",
                "infrastructure_failure": "Evidence of environment/calibration/image availability, Docker or exhausted transient model-service failure, not an inferred agent failure",
                "official_evaluation_error": "Evaluator nonzero exit, missing/conflicting report or grader/parser exception not established as infrastructure failure",
                "harness_or_protocol_failure": "Harness crash or invalid model response; do not automatically classify as infrastructure",
                "pending_or_not_run": "No completed attempt, including interruption awaiting inspection",
            },
            "classification": "Preserve raw official verdict separately. Use evidence-based exclusive analysis categories; official infra/error sets can overlap. Empty patches are non-resolved task outcomes with an explicit subtype; do not relabel the raw official status",
            "secondary_score": "resolved / validly_graded with coverage; never substitutes for fixed-denominator primary score",
            "metrics": ["input/output/cache-hit/cache-miss tokens and usage coverage", "physical requests vs logical turns", "retries and auxiliary/child requests", "tools called/started/failed by name", "per-task first mutation root turn with never-mutated distinct from unknown", "per-repo outcomes"],
        },
        "service_failures": {
            "transient_kinds": ["rate_limit", "connection", "server_unavailable"],
            "max_transient_retries_per_logical_request": 2,
            "backoff_seconds": {"base": 1, "maximum": 8, "jitter_ratio": 0.25},
            "context_recovery": "At most one reactive compaction per logical request; identical policy across profiles; account for auxiliary calls",
            "exhaustion": "Record failed formal attempt and stop that task; never automatically restart a rollout or replace a task",
            "interruption": "Preserve receipts and partial artifacts. Unclear launch state requires inspection, never silent replay",
            "later_attempts": "Any separately approved rerun must retain original attempt and reason, use a new ID, and be reported separately rather than selecting the best result",
            "usage": "Count every observed request including failures and retries; missing billed usage stays unknown, not zero",
        },
        "freeze_policy": "No result-driven Test tuning or membership changes. Protocol/data changes require a new versioned bundle, retaining the old bundle. SHA256SUMS covers every artifact except itself; manifest excludes itself and SHA256SUMS to avoid circular hashes",
    }


def freeze(source: Path, output: Path) -> dict:
    descriptor, rows, raw_files, schemas = load_source(source)
    selected, quotas, ranks = select_rows(rows)
    artifacts = {f"evaluator/source/{name}": content for name, content in raw_files.items()}
    artifacts["evaluator/population.jsonl"] = jsonl_bytes(rows)
    artifacts["evaluator/selection.jsonl"] = jsonl_bytes([row for _, row in selected])
    artifacts["agent/agent_tasks.jsonl"] = jsonl_bytes([
        {field: row[field] for field in AGENT_FIELDS} for _, row in selected])
    artifacts["protocol.json"] = json_bytes(protocol())
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["selection_index", "source_row", "instance_id", "repo", "base_commit",
                     "repo_rank", "sampling_sha256", "profile_order"])
    for index, (source_row, row) in enumerate(selected):
        writer.writerow([index, source_row, row["instance_id"], row["repo"], row["base_commit"],
                         ranks[row["instance_id"]], digest(f"{SEED}\0{row['repo']}\0{row['instance_id']}".encode()),
                         "A,B" if index % 2 == 0 else "B,A"])
    artifacts["selection.csv"] = stream.getvalue().encode("utf-8")
    manifest = {
        "format_version": 1, "dataset": DATASET, "revision": descriptor["revision"], "split": "test",
        "population_count": len(rows), "selected_count": len(selected), "seed": SEED,
        "repository_population": dict(sorted(Counter(row["repo"] for row in rows).items())),
        "repository_selection": quotas, "parquet_schemas": schemas,
        "selected_instance_ids": [row["instance_id"] for _, row in selected],
        "generator": {"path": "evals/swe_bench_lite/freeze.py", "sha256": digest(Path(__file__).read_bytes())},
        "parquet_decoder": {"package": "pyarrow", "version": version("pyarrow")},
        "artifacts": {name: {"sha256": digest(content), "bytes": len(content)}
                      for name, content in sorted(artifacts.items())},
        "execution_configuration": "Not frozen in phase 1; no rollout or grading is authorized by this data bundle",
    }
    artifacts["manifest.json"] = json_bytes(manifest)
    artifacts["SHA256SUMS"] = "".join(f"{digest(content)}  {name}\n"
                                     for name, content in sorted(artifacts.items())).encode("utf-8")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download pinned official Test data only")
    fetch.add_argument("--revision", required=True, type=revision_value)
    fetch.add_argument("--output", required=True, type=Path)
    build = commands.add_parser("freeze", help="Offline deterministic generation; refuses existing output")
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_source(args.revision, args.output)
    else:
        manifest = freeze(args.source, args.output)
        print(json.dumps({"revision": manifest["revision"], "repository_selection": manifest["repository_selection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
