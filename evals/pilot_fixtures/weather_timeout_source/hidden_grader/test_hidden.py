import ast
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(os.environ["TINYHARNESS_EVAL_WORKSPACE"])
sys.path.insert(0, str(WORKSPACE / "src"))

from weather.client import WeatherClient
from weather.loader import load_settings

EXPECTED_TEST_HASHES = {
    "test_client.py": "719ce78fb804700ba7e9bc7fd6d3d4640f7ed466b7d219f8a01f4a9bf6b95c35",
    "test_loader.py": "266fa52e8af711067e5b4d0c1483ebfe1d3fcb01d06aeac4155b2681c404d803",
}


class WeatherTimeoutHiddenTest(unittest.TestCase):
    def test_runtime_default_is_thirty(self):
        self.assertEqual(load_settings({}).timeout_seconds, 30)
        self.assertEqual(WeatherClient(environment={}).timeout_seconds, 30)

    def test_environment_override_is_unchanged(self):
        environment = {"WEATHER_TIMEOUT": "14"}
        self.assertEqual(load_settings(environment).timeout_seconds, 14)
        self.assertEqual(
            WeatherClient(environment=environment).timeout_seconds,
            14,
        )

    def test_explicit_constructor_value_has_highest_precedence(self):
        client = WeatherClient(
            timeout_seconds=8,
            environment={"WEATHER_TIMEOUT": "14"},
        )
        self.assertEqual(client.timeout_seconds, 8)

    def test_example_value_stays_five(self):
        example = json.loads(
            (WORKSPACE / "config" / "settings.example.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(example["timeout_seconds"], 5)

    def test_no_duplicate_timeout_default_source(self):
        weather_root = WORKSPACE / "src" / "weather"
        assignments = []
        for path in weather_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(
                        isinstance(target, ast.Name)
                        and target.id == "DEFAULT_TIMEOUT_SECONDS"
                        for target in targets
                    ):
                        assignments.append(path.name)
        self.assertLessEqual(len(assignments), 1)
        if (weather_root / "defaults.py").exists():
            self.assertFalse((weather_root / "config.py").exists())
            self.assertFalse((weather_root / "settings.py").exists())

    def test_visible_tests_are_unchanged(self):
        tests = WORKSPACE / "tests"
        self.assertEqual(
            {path.name for path in tests.glob("test_*.py")},
            set(EXPECTED_TEST_HASHES),
        )
        for name, expected in EXPECTED_TEST_HASHES.items():
            content = (tests / name).read_text(encoding="utf-8")
            actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
