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
NATIVE = "qwen3vl-2b-mtp0-native-2b-mtp0-native-g1.json"


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

    def tamper(self, edit):
        directory = self.copied()
        edit(directory)
        return verify_baseline.verify(directory)[1]

    def test_a_dropped_matrix_case_is_caught(self):
        def edit(directory):
            path = directory / "2b-mtp0.json"
            profile = json.loads(path.read_text())
            del profile["workload"]["matrix"]["hot_long"]
            path.write_text(json.dumps(profile))

        self.assertIn("matrix differs", " ".join(self.tamper(edit)))

    def test_a_relaxed_threshold_is_caught(self):
        def edit(directory):
            path = directory / "2b-mtp0.json"
            profile = json.loads(path.read_text())
            profile["measurement"]["max_ratio"] = 5.0
            path.write_text(json.dumps(profile))

        self.assertIn("not the profile's", " ".join(self.tamper(edit)))

    def test_a_changed_workload_is_caught(self):
        def edit(directory):
            path = directory / "qwen3vl-2b-mtp0-native-2b-mtp0-native-g1.json"
            report = json.loads(path.read_text())
            report["max_new_tokens"] = 8
            path.write_text(json.dumps(report))

        self.assertIn("max_new_tokens", " ".join(self.tamper(edit)))

    def test_a_report_without_a_re_measurement_is_caught(self):
        def edit(directory):
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["statistical_basis"]["reproductions"] = []
            manifest["statistical_basis"]["paired_units"] = 1
            path.write_text(json.dumps(manifest))

        self.assertIn("no independent re-measurement", " ".join(self.tamper(edit)))

    def test_a_re_measurement_beyond_the_declared_drift_is_caught(self):
        def edit(directory):
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["statistical_basis"]["reproductions"][0]["within_declared_tolerance"] = False
            path.write_text(json.dumps(manifest))

        self.assertIn("beyond the declared drift", " ".join(self.tamper(edit)))

    def test_a_paired_unit_count_that_disagrees_is_caught(self):
        def edit(directory):
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["statistical_basis"]["paired_units"] = 9
            path.write_text(json.dumps(manifest))

        self.assertIn("does not match", " ".join(self.tamper(edit)))

    def report(self, directory):
        path = directory / NATIVE
        return path, json.loads(path.read_text())

    def test_a_dirty_tree_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            report["identity"]["source"]["dirty"] = True
            path.write_text(json.dumps(report))

        self.assertIn("clean tree", " ".join(self.tamper(edit)))

    def test_a_missing_source_revision_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            report["identity"]["source"]["revision"] = ""
            path.write_text(json.dumps(report))

        self.assertIn("no source revision", " ".join(self.tamper(edit)))

    def test_a_failed_checklist_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            report["checklist"]["compliant"] = False
            path.write_text(json.dumps(report))

        self.assertIn("checklist did not pass", " ".join(self.tamper(edit)))

    def test_a_checklist_mismatch_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            report["checklist"]["mismatches"] = [{"field": "numeric"}]
            path.write_text(json.dumps(report))

        self.assertIn("records mismatches", " ".join(self.tamper(edit)))

    def test_a_checklist_for_another_profile_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            report["checklist"]["profile_id"] = "something-else"
            path.write_text(json.dumps(report))

        self.assertIn("another profile", " ".join(self.tamper(edit)))

    def test_a_readback_without_a_source_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            report["config_readback"] = {"readback": {}}
            path.write_text(json.dumps(report))

        self.assertIn("names no source", " ".join(self.tamper(edit)))

    def test_an_enabled_cache_without_reuse_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            report["hot_prefix_tokens_reused"] = 0.0
            path.write_text(json.dumps(report))

        self.assertIn("reused no tokens", " ".join(self.tamper(edit)))

    def test_a_null_readback_is_a_note_not_a_failure(self):
        # The frozen vLLM reports record the command line as the configuration source.
        directory = self.copied()
        self.assertEqual(verify_baseline.verify(directory)[1], [])

    def log(self, directory):
        return directory / (NATIVE.replace(".json", ".telemetry.jsonl"))

    def test_a_telemetry_log_that_changed_is_caught(self):
        def edit(directory):
            with self.log(directory).open("a") as handle:
                handle.write('{"type": "sample", "timestamp": 0}\n')

        self.assertIn("does not match its recorded fingerprint", " ".join(self.tamper(edit)))

    def test_a_telemetry_log_with_the_wrong_sample_count_is_caught(self):
        def edit(directory):
            path = self.log(directory)
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            removed = next(index for index, line in enumerate(lines) if '"type": "sample"' in line)
            path.write_text("\n".join(lines[:removed] + lines[removed + 1 :]) + "\n")

        problems = " ".join(self.tamper(edit))
        self.assertIn("does not match its recorded fingerprint", problems)
        self.assertIn("samples", problems)

    def test_a_missing_telemetry_log_is_caught(self):
        def edit(directory):
            self.log(directory).unlink()

        self.assertIn("is missing", " ".join(self.tamper(edit)))

    def test_a_report_without_telemetry_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            del report["telemetry"]
            path.write_text(json.dumps(report))

        self.assertIn("no hardware telemetry", " ".join(self.tamper(edit)))

    def test_a_telemetry_log_that_does_not_parse_is_caught(self):
        def edit(directory):
            self.log(directory).write_text("not json\n")

        self.assertIn("not JSON lines", " ".join(self.tamper(edit)))

    def test_a_report_without_a_fingerprint_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            report["telemetry"]["sha256"] = ""
            path.write_text(json.dumps(report))

        self.assertIn("not fingerprinted", " ".join(self.tamper(edit)))

    def test_a_report_whose_trials_attribute_no_device_is_caught(self):
        def edit(directory):
            path, report = self.report(directory)
            for trial in report["trials"]:
                trial.pop("gpu", None)
            path.write_text(json.dumps(report))

        self.assertIn("attributes device telemetry", " ".join(self.tamper(edit)))

    def test_the_shortfall_is_a_note_not_a_failure(self):
        directory = self.copied()
        self.assertEqual(verify_baseline.verify(directory)[1], [])

    def test_the_tree_recomputes(self):
        for directory in verify_baseline.baselines():
            name, problems = verify_baseline.verify(directory)
            self.assertEqual(problems, [], name)


if __name__ == "__main__":
    unittest.main()
