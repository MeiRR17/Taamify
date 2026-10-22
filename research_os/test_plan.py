"""Standard-library tests for the Research OS planning registry. No machine/GPU access."""
import json
import tempfile
import unittest
from pathlib import Path
from research_os.plan import ROOT, calculate, load_json


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_json(ROOT / "config/pipelines.json")
        cls.augmentations = load_json(ROOT / "config/augmentations.json")

    def test_pipeline_ids_and_totals(self):
        result = calculate(self.config)
        self.assertEqual(result["status"], "planning_only")
        self.assertEqual(result["pipeline_count"], 13)
        self.assertEqual(result["planned_fits"], 51)
        self.assertEqual(result["planned_fit_epochs"], 1350)
        self.assertEqual(result["pipelines_with_benchmarks"], 0)

    def test_human_hours(self):
        h = calculate(self.config)["human_hours_planned"]
        self.assertEqual(h["academic"], {"minimum": 176, "maximum": 342})
        self.assertEqual(h["product"], {"minimum": 76, "maximum": 148})
        self.assertEqual(h["engineering"], {"minimum": 105, "maximum": 207})

    def test_hours_require_measured_input(self):
        result = calculate(self.config, {"P00": {"minutes_per_epoch": 2, "setup_hours": 1.5}})
        self.assertEqual(result["pipelines_with_benchmarks"], 1)
        # P00 = 3 fits x 30 epochs x 2 minutes, plus 1.5h prep
        self.assertAlmostEqual(result["hours_from_measured_inputs_only"], 4.5)
        for p in result["pipelines"]:
            if p["id"] != "P00":
                self.assertIsNone(p["hours_measured_inputs"])

    def test_bad_benchmark_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate(self.config, {"P99": {"minutes_per_epoch": 2}})
        with self.assertRaises(ValueError):
            calculate(self.config, {"P00": {"minutes_per_epoch": -1}})
        with self.assertRaises(ValueError):
            calculate(self.config, {"P00": {"minutes_per_epoch": float("nan")}})

    def test_augmentation_registry_is_present(self):
        modules = self.augmentations["modules"]
        self.assertEqual(len(modules), 9)
        self.assertEqual(len({a["id"] for a in modules}), 9)
        self.assertIn("root_recording_id", self.augmentations["policy"])


if __name__ == "__main__":
    unittest.main()
