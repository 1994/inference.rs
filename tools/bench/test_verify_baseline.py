"""A signed baseline's numbers have to follow from the evidence it froze."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "verify_baseline", Path(__file__).with_name("verify-baseline.py")
)
verify_baseline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_baseline)
FROZEN = ROOT / "benchmarks/baselines/serving/2b-mtp0-serving-v1"


def entry(case="short", metric="wall_seconds", ratio=1.5, available=True):
    # The three numbers are independent so a test can move one and expect exactly one finding.
    return {
        "case": case,
        "metric": metric,
        "available": available,
        "baseline": 1.0,
        "candidate": 2.0,
        "ratio": ratio,
    }


class NumbersTest(unittest.TestCase):
    def test_an_identical_recomputation_reports_nothing(self):
        recorded = [entry(), entry(case="long", ratio=1.25)]
        self.assertEqual(verify_baseline.compare_numbers(recorded, recorded), [])

    def test_a_changed_ratio_is_reported(self):
        recorded = [entry(ratio=1.5)]
        recomputed = [entry(ratio=1.6)]
        problems = verify_baseline.compare_numbers(recorded, recomputed)
        self.assertEqual(len(problems), 1)
        self.assertIn("ratio", problems[0])

    def test_a_metric_that_is_not_recomputed_is_reported(self):
        recorded = [entry(), entry(case="long", ratio=1.25)]
        problems = verify_baseline.compare_numbers(recorded, [entry()])
        self.assertIn("not recomputed", problems[0])

    def test_a_metric_that_is_not_recorded_is_reported(self):
        problems = verify_baseline.compare_numbers([entry()], [entry(), entry(case="long")])
        self.assertIn("recomputed but not recorded", problems[0])

    def test_an_unavailable_metric_needs_no_numbers(self):
        recorded = [{"case": "long", "metric": "tpot_seconds", "available": False}]
        recomputed = [{"case": "long", "metric": "tpot_seconds", "available": False}]
        self.assertEqual(verify_baseline.compare_numbers(recorded, recomputed), [])


class FrozenTest(unittest.TestCase):
    def copied(self):
        directory = Path(tempfile.mkdtemp()) / "2b-mtp0-serving-v1"
        shutil.copytree(FROZEN, directory)
        return directory

    def test_the_frozen_baseline_recomputes(self):
        name, problems = verify_baseline.verify(self.copied())
        self.assertEqual(name, "2b-mtp0-serving-v1")
        self.assertEqual(problems, [])

    def test_a_tampered_manifest_is_caught(self):
        directory = self.copied()
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["gate"]["performance"]["metrics"][0]["ratio"] += 0.25
        (directory / "manifest.json").write_text(json.dumps(manifest))
        _, problems = verify_baseline.verify(directory)
        self.assertTrue(problems)
        self.assertIn("ratio", problems[0])

    def test_the_tree_recomputes(self):
        for directory in verify_baseline.baselines():
            name, problems = verify_baseline.verify(directory)
            self.assertEqual(problems, [], name)


if __name__ == "__main__":
    unittest.main()
