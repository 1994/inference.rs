"""The gate must catch fast-but-wrong kernels and incomplete benchmark evidence."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from attention_gate import evaluate


def evidence():
    case = {"id": "tail72", "tokens": 33, "heads": 2, "head_dim": 72, "elements": 4752}
    manifest = {
        "fixture_sha256": "fixture-digest",
        "contract": "sdpa-f32-v1",
        "scope": "case-defined-sdpa-pipeline-f32-output-cuda-graph",
        "cases": [case],
    }
    baseline = {
        **manifest,
        "schema": 1,
        "device": "test-gpu",
        "driver": "test-driver",
        "toolkit": "test-toolkit",
        "build": "release",
        "implementation": "candle",
        "precision": "f16",
        "run_id": "paired-run",
        "records": [
            {
                "case": case["id"],
                "shape": case,
                "round": r,
                "warmup": 100,
                "replays_per_sample": 8,
                "samples_ms": [1.0] * 60,
                "core_samples_ms": [1.0] * 60,
                "max_abs_error": 1e-6,
                "reference_max_abs": 1.0,
                "finite": True,
                "elements": case["elements"],
            }
            for r in range(5)
        ],
    }
    candidate = copy.deepcopy(baseline)
    candidate["implementation"] = "native"
    candidate["precision"] = "compensated-f32"
    for row in candidate["records"]:
        row["samples_ms"] = [0.8] * 60
        row["core_samples_ms"] = [0.8] * 60
    return manifest, baseline, candidate


class GateTests(unittest.TestCase):
    def test_core_geometric_mean_must_not_regress(self):
        m, a, b = evidence()
        for row in b["records"]:
            row["core_samples_ms"] = [1.02] * 60
        report = evaluate(m, a, b)
        self.assertEqual(report["decision"], "review")
        self.assertLess(report["core_geomean_speedup"], 1.0)

    def test_core_timing_drift_is_not_hidden_by_pipeline_stability(self):
        m, a, b = evidence()
        a["records"][0]["core_samples_ms"] = [1.2] * 60
        report = evaluate(m, a, b)
        self.assertEqual(report["decision"], "review")
        self.assertTrue(
            any("core timing drift" in issue for issue in report["performance_reviews"])
        )

    def test_framework_overhead_cannot_hide_a_slower_native_kernel(self):
        m, a, b = evidence()
        for row in b["records"]:
            row["core_samples_ms"] = [1.1] * 60
        self.assertEqual(evaluate(m, a, b)["decision"], "review")

    def test_cli_exit_code_and_report_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for decision, status in (("pass", 0), ("reject", 1), ("invalid", 1)):
                m, a, b = evidence()
                if decision == "reject":
                    b["records"][0]["max_abs_error"] = 1.0
                elif decision == "invalid":
                    b["records"].pop()
                command = [sys.executable, str(Path(__file__).with_name("attention_gate.py"))]
                for name, document in (("manifest", m), ("baseline", a), ("candidate", b)):
                    path = root / f"{name}.json"
                    path.write_text(json.dumps(document))
                    command.extend([f"--{name}", str(path)])
                command.extend(["--out", str(root / "report.json")])
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                with self.subTest(decision=decision):
                    self.assertEqual(result.returncode, status, result.stderr)
                    self.assertEqual(
                        json.loads((root / "report.json").read_text())["decision"], decision
                    )

    def test_valid_speedup_passes_without_changing_production(self):
        report = evaluate(*evidence())
        self.assertEqual(report["decision"], "pass")
        self.assertFalse(report["changes_production"])

    def test_fast_but_wrong_is_rejected(self):
        m, a, b = evidence()
        b["records"][0]["max_abs_error"] = 0.01
        self.assertEqual(evaluate(m, a, b)["decision"], "reject")

    def test_invalid_baseline_cannot_approve_candidate(self):
        m, a, b = evidence()
        a["records"][0]["max_abs_error"] = 0.01
        self.assertEqual(evaluate(m, a, b)["decision"], "reject")

    def test_regression_and_small_gain_require_review(self):
        for latency in (1.06, 1.01):
            m, a, b = evidence()
            for row in b["records"]:
                row["samples_ms"] = [latency] * 60
            self.assertEqual(evaluate(m, a, b)["decision"], "review")

    def test_tail_regression_is_not_hidden_by_median(self):
        m, a, b = evidence()
        for row in b["records"]:
            row["samples_ms"][-10:] = [2.0] * 10
        self.assertEqual(evaluate(m, a, b)["decision"], "review")

    def test_unstable_round_requires_review(self):
        m, a, b = evidence()
        b["records"][0]["samples_ms"] = [1.5] * 60
        self.assertEqual(evaluate(m, a, b)["decision"], "review")

    def test_missing_duplicate_and_unknown_cases_fail_closed(self):
        for mutation in ("missing", "duplicate", "unknown"):
            m, a, b = evidence()
            if mutation == "missing":
                b["records"].pop()
            elif mutation == "duplicate":
                b["records"].append(b["records"][0])
            else:
                b["records"][0]["case"] = "not-in-manifest"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evaluate(m, a, b)

    def test_nonfinite_error_scale_latency_and_output_fail_closed(self):
        for field in ("max_abs_error", "reference_max_abs", "samples_ms", "finite"):
            m, a, b = evidence()
            b["records"][0][field] = (
                [float("nan")] * 60
                if field == "samples_ms"
                else False
                if field == "finite"
                else float("nan")
            )
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate(m, a, b)

    def test_mismatched_environment_or_precision_is_invalid(self):
        for field in (
            "device",
            "driver",
            "toolkit",
            "fixture_sha256",
            "scope",
            "contract",
            "build",
        ):
            m, a, b = evidence()
            b[field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate(m, a, b)

    def test_incomplete_samples_and_wrong_output_shape_fail(self):
        for field, value in (("samples_ms", [1.0]), ("elements", 0), ("warmup", 0)):
            m, a, b = evidence()
            b["records"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate(m, a, b)

    def test_candidate_cannot_relax_tolerance_by_inflating_reference_scale(self):
        m, a, b = evidence()
        b["records"][0]["reference_max_abs"] = 1000
        with self.assertRaises(ValueError):
            evaluate(m, a, b)

    def test_one_regressing_shape_is_not_hidden_by_other_speedups(self):
        m, a, b = evidence()
        second = {**m["cases"][0], "id": "second"}
        m["cases"].append(second)
        for run in (a, b):
            extra = copy.deepcopy(run["records"])
            for row in extra:
                row.update(case="second", shape=second, samples_ms=[1.0] * 60)
            run["records"].extend(extra)
        for row in b["records"][:5]:
            row["samples_ms"] = [1.06] * 60
        for row in b["records"][5:]:
            row["samples_ms"] = [0.1] * 60
        self.assertEqual(evaluate(m, a, b)["decision"], "review")


if __name__ == "__main__":
    unittest.main()
