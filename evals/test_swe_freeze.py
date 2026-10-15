"""Offline checks for Test freezing. No model, Docker or network is required."""

import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.swe_bench_lite.freeze import (
    AGENT_FIELDS, DATASET, STRING_FIELDS, digest, freeze, json_bytes,
    load_source, select_rows, validate_rows,
)

ROOT = Path(__file__).parent / "swe_bench_lite" / "frozen" / "test60-v1"
REVISION = "b0dde1093fe417d83b7184254edf8199c1f0dff5"


def fixture_rows():
    rows = []
    for repo, count in [("a/one", 150), ("b/two", 90), ("c/three", 50), ("d/four", 10)]:
        for n in range(count):
            row = {field: f"PRIVATE_CANARY_{field}" for field in STRING_FIELDS}
            row.update(instance_id=f"{repo.replace('/', '__')}-{n}", repo=repo,
                       base_commit="a" * 40, problem_statement=f"Public issue {n}",
                       image="public-image:latest", FAIL_TO_PASS=["PRIVATE_FTP"],
                       PASS_TO_PASS=["PRIVATE_PTP"])
            rows.append(row)
    return rows


def tree(path):
    return {p.relative_to(path).as_posix(): p.read_bytes()
            for p in path.rglob("*") if p.is_file()}


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Requires evaluation parquet dependency")
class FreezeTest(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        shard = "data/test-00000-of-00001.parquet"
        stream = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(fixture_rows()), stream)
        files = {
            shard: stream.getvalue(), "README.md": b"Fixture, never benchmark evidence\n",
            "metadata.json": json_bytes({"id": DATASET, "sha": REVISION,
                                          "siblings": [{"rfilename": shard}]}),
        }
        for name, content in files.items():
            p = self.source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        self.descriptor = {"dataset": DATASET, "revision": REVISION, "split": "test",
                           "parquet_files": [shard],
                           "files": {name: {"sha256": digest(content), "url": "fixture"}
                                     for name, content in files.items()}}
        self.write_descriptor()

    def write_descriptor(self):
        (self.source / "source.json").write_bytes(json_bytes(self.descriptor))

    def test_independent_generation_and_all_checksums(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("Offline only")):
            freeze(self.source, self.root / "one")
            freeze(self.source, self.root / "two")
        a, b = tree(self.root / "one"), tree(self.root / "two")
        self.assertEqual(a, b)
        self.assertEqual({k: digest(v) for k, v in a.items()},
                         {k: digest(v) for k, v in b.items()})
        checks = dict(line.split("  ", 1)[::-1] for line in a["SHA256SUMS"].decode().splitlines())
        self.assertEqual(set(checks), set(a) - {"SHA256SUMS"})
        for name, expected in checks.items():
            self.assertEqual(digest(a[name]), expected)
        manifest = json.loads(a["manifest.json"])
        for name, record in manifest["artifacts"].items():
            self.assertEqual(record, {"sha256": digest(a[name]), "bytes": len(a[name])})

    def test_agent_allowlist_and_private_canaries(self):
        freeze(self.source, self.root / "bundle")
        p = self.root / "bundle" / "agent" / "agent_tasks.jsonl"
        tasks = [json.loads(line) for line in p.read_bytes().splitlines()]
        self.assertEqual(len(tasks), 60)
        self.assertEqual(list(p.parent.iterdir()), [p])
        self.assertNotIn(b"PRIVATE", p.read_bytes())
        originals = {r["instance_id"]: r for r in fixture_rows()}
        for task in tasks:
            self.assertEqual(set(task), set(AGENT_FIELDS))
            self.assertEqual(task, {k: originals[task["instance_id"]][k] for k in AGENT_FIELDS})

    def test_quota_and_source_order(self):
        selected, quotas, _ = select_rows(fixture_rows())
        self.assertEqual(quotas, {"a/one": 30, "b/two": 18, "c/three": 10, "d/four": 2})
        self.assertEqual(len(selected), 60)
        self.assertEqual([i for i, _ in selected], sorted(i for i, _ in selected))
        freeze(self.source, self.root / "bundle")
        rows = list(csv.DictReader(io.StringIO(
            (self.root / "bundle" / "selection.csv").read_text(encoding="utf-8"))))
        self.assertEqual([r["instance_id"] for _, r in selected], [r["instance_id"] for r in rows])

    def test_selection_ignores_all_oracle_and_environment_fields(self):
        rows = fixture_rows()
        before = select_rows(rows)
        for row in rows:
            for field in set(row) - {"repo", "instance_id"}:
                row[field] = "ALL VALUES CHANGED INCLUDING PATCH DIFFICULTY IMAGE"
        after = select_rows(rows)
        self.assertEqual([i for i, _ in before[0]], [i for i, _ in after[0]])
        self.assertEqual(before[1:], after[1:])

    def test_small_repo_is_covered_and_ties_are_lexical(self):
        rows = fixture_rows()
        rows[-1]["repo"] = "z/tiny"
        _, quotas, _ = select_rows(rows)
        self.assertEqual(sum(quotas.values()), 60)
        self.assertEqual(quotas["z/tiny"], 1)
        # Equal deficits select the lexical first repo for the extra slot.
        balanced = [{"repo": f"repo/{i:02}", "instance_id": str(n)}
                    for i in range(11) for n in range(i * 30, (i + 1) * 30)]
        q = select_rows(balanced)[1]
        self.assertEqual([q[f"repo/{i:02}"] for i in range(11)], [6] * 5 + [5] * 6)

    def test_validation_count_ids_schema_and_pin(self):
        for mutate in [lambda rows: rows.pop(),
                       lambda rows: rows[1].update(instance_id=rows[0]["instance_id"]),
                       lambda rows: rows[0].pop("patch"),
                       lambda rows: rows[0].update(FAIL_TO_PASS="[]"),
                       lambda rows: rows[0].update(base_commit="not-a-commit")]:
            rows = fixture_rows()
            mutate(rows)
            with self.assertRaises(ValueError):
                validate_rows(rows)
        self.descriptor["revision"] = "main"
        self.write_descriptor()
        with self.assertRaises(ValueError):
            load_source(self.source)

    def test_corruption_and_overwrite_fail_closed(self):
        freeze(self.source, self.root / "bundle")
        before = tree(self.root / "bundle")
        with self.assertRaises(FileExistsError):
            freeze(self.source, self.root / "bundle")
        self.assertEqual(before, tree(self.root / "bundle"))
        (self.source / "README.md").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            freeze(self.source, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_source_path_escape_is_rejected(self):
        self.descriptor["files"]["../outside"] = {"sha256": "0" * 64}
        self.write_descriptor()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            load_source(self.source)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Requires evaluation parquet dependency")
class FrozenOfficialTest(unittest.TestCase):
    def test_official_bundle_independently_reproduces_byte_for_byte(self):
        self.assertTrue(ROOT.is_dir(), "Official frozen bundle must be present")
        with tempfile.TemporaryDirectory() as temporary, patch(
            "urllib.request.urlopen", side_effect=AssertionError("Offline only")
        ):
            one, two = Path(temporary) / "one", Path(temporary) / "two"
            for hash_seed, target in [("1", one), ("2", two)]:
                subprocess.run([
                    sys.executable, "-B", "-m", "evals.swe_bench_lite.freeze", "freeze",
                    "--source", str(ROOT / "evaluator" / "source"), "--output", str(target),
                ], cwd=ROOT.parents[3], env=dict(os.environ, PYTHONHASHSEED=hash_seed,
                                               PYTHONDONTWRITEBYTECODE="1"),
                    check=True, capture_output=True, timeout=60)
            self.assertEqual(tree(one), tree(two))
            self.assertEqual(tree(one), tree(ROOT))
        manifest = json.loads((ROOT / "manifest.json").read_bytes())
        self.assertEqual(manifest["revision"], REVISION)
        self.assertEqual(manifest["population_count"], 300)
        self.assertEqual(manifest["selected_count"], 60)
        population = [json.loads(line) for line in (ROOT / "evaluator" / "population.jsonl").read_bytes().splitlines()]
        dev_ids = {json.loads(line)["instance_id"] for line in
                   (ROOT.parents[1] / "dev.jsonl").read_bytes().splitlines() if line.strip()}
        self.assertFalse(dev_ids & {r["instance_id"] for r in population})
        agent = [json.loads(line) for line in (ROOT / "agent" / "agent_tasks.jsonl").read_bytes().splitlines()]
        self.assertTrue(all(set(row) == set(AGENT_FIELDS) for row in agent))
        self.assertEqual([r["instance_id"] for r in agent], manifest["selected_instance_ids"])


if __name__ == "__main__":
    unittest.main()
