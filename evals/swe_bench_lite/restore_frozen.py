"""Restore omitted public Test60 data without resampling or overwriting."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import urllib.request

from .benchmark import FROZEN, FROZEN_SUMS, safe_file, sha, verify_bundle
from .freeze import jsonl_bytes, validate_rows

PARQUET = "evaluator/source/data/test-00000-of-00001.parquet"
POPULATION = "evaluator/population.jsonl"


def restore(bundle=FROZEN, *, parquet_cache=None, download=None):
    bundle = Path(bundle).resolve()
    sums = safe_file(bundle, "SHA256SUMS").read_bytes()
    if sha(sums) != FROZEN_SUMS:
        raise ValueError("Frozen checksum inventory changed")
    expected = dict(line.split("  ", 1)[::-1] for line in sums.decode().splitlines())
    missing = []
    for name, fingerprint in expected.items():
        path = safe_file(bundle, name)
        if path.exists():
            if sha(path.read_bytes()) != fingerprint:
                raise ValueError(f"Existing frozen artifact changed: {name}")
        elif name in {PARQUET, POPULATION}:
            missing.append(name)
        else:
            raise ValueError(f"Required Git artifact missing: {name}")
    if not missing:
        verify_bundle(bundle)
        return []
    descriptor = json.loads((bundle / "evaluator/source/source.json").read_bytes())
    record = descriptor["files"]["data/test-00000-of-00001.parquet"]
    url = ("https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite/resolve/"
           + descriptor["revision"] + "/data/test-00000-of-00001.parquet")
    if record["url"] != url or record["sha256"] != expected[PARQUET]:
        raise ValueError("Pinned source identity mismatch")
    if PARQUET not in missing:
        content = (bundle / PARQUET).read_bytes()
    elif parquet_cache is not None:
        content = Path(parquet_cache).read_bytes()
    else:
        if download is None:
            def download(address):
                with urllib.request.urlopen(address, timeout=60) as response:
                    return response.read()
        content = download(url)
    if sha(content) != expected[PARQUET]:
        raise ValueError("Downloaded/cached parquet checksum mismatch")
    import pyarrow.parquet as pq
    rows = pq.read_table(io.BytesIO(content)).to_pylist()
    validate_rows(rows)
    payloads = {PARQUET: content, POPULATION: jsonl_bytes(rows)}
    for name, data in payloads.items():
        if sha(data) != expected[name]:
            raise ValueError(f"Reconstructed artifact checksum mismatch: {name}")
    # Validate every byte before writing; exclusive creation never repairs files.
    for name in missing:
        path = safe_file(bundle, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payloads[name])
    verify_bundle(bundle)
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=FROZEN)
    parser.add_argument("--parquet-cache", type=Path,
                        help="Optional original public parquet; SHA-256 required")
    args = parser.parse_args(argv)
    print(json.dumps({"restored": restore(args.bundle, parquet_cache=args.parquet_cache)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
