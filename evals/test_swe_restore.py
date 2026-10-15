"""Offline restore tests using the pinned local public parquet as a cache."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from evals.swe_bench_lite.benchmark import FROZEN, verify_bundle
from evals.swe_bench_lite.restore_frozen import PARQUET, POPULATION, restore


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name) / "bundle"
        for path in FROZEN.rglob("*"):
            if path.is_file() and path.relative_to(FROZEN).as_posix() not in {PARQUET, POPULATION}:
                target = self.bundle / path.relative_to(FROZEN)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())

    def test_cache_restore_matches_original_bytes_and_is_idempotent(self):
        no_network = Mock(side_effect=AssertionError("No network"))
        restored = restore(self.bundle, parquet_cache=FROZEN / PARQUET, download=no_network)
        self.assertEqual(set(restored), {PARQUET, POPULATION})
        for path in FROZEN.rglob("*"):
            if path.is_file():
                self.assertEqual((self.bundle / path.relative_to(FROZEN)).read_bytes(), path.read_bytes())
        self.assertEqual(restore(self.bundle, download=no_network), [])
        no_network.assert_not_called()
        self.assertEqual(len(verify_bundle(self.bundle)[0]), 60)

    def test_pinned_download_interface(self):
        download = Mock(return_value=(FROZEN / PARQUET).read_bytes())
        restore(self.bundle, download=download)
        self.assertEqual(download.call_count, 1)
        self.assertIn("/resolve/b0dde1093fe417d83b7184254edf8199c1f0dff5/", download.call_args.args[0])

    def test_bad_download_does_not_write_missing_files(self):
        with self.assertRaisesRegex(ValueError, "parquet checksum"):
            restore(self.bundle, download=lambda url: b"corrupt fixture")
        self.assertFalse((self.bundle / PARQUET).exists())
        self.assertFalse((self.bundle / POPULATION).exists())

    def test_existing_artifact_is_never_repaired(self):
        path = self.bundle / "manifest.json"
        path.write_bytes(b"changed fixture")
        no_network = Mock(side_effect=AssertionError("No network"))
        with self.assertRaisesRegex(ValueError, "Existing frozen artifact changed"):
            restore(self.bundle, download=no_network)
        self.assertEqual(path.read_bytes(), b"changed fixture")
        no_network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
