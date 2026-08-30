import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from importer.pipeline import import_customers


class CustomerPipelineTest(unittest.TestCase):
    def test_pipeline_cleans_each_record(self):
        result = import_customers(
            [
                {"email": " a@example.com ", "age": "20"},
                {"email": " b@example.com ", "age": 30},
            ]
        )

        self.assertEqual(result[0]["email"], "a@example.com")
        self.assertEqual(result[0]["age"], 20)
        self.assertEqual(result[1]["email"], "b@example.com")


if __name__ == "__main__":
    unittest.main()
