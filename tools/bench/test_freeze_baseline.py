"""A frozen baseline must be complete, immutable and re-verifiable."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "freeze_baseline", Path(__file__).with_name("freeze-baseline.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Reuse the gate's own fixtures so the frozen reports are the ones the gate accepts.
gate_spec = importlib.util.spec_from_file_location(
    "gate_fixture", Path(__file__).with_name("test_compare_results.py")
)
gate = importlib.util.module_from_spec(gate_spec)
gate_spec.loader.exec_module(gate)


class FreezeTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.reports = self.write_reports()
        self.evidence = self.directory / "workload.json"
        self.evidence.write_text('[{"case": "short"}]')

    def tearDown(self):
        for path in sorted(self.directory.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        self.directory.rmdir()

    def write_reports(self, candidate=None):
        baseline = gate.report()
        candidate = candidate if candidate is not None else gate.report()
        self.baseline_path = self.directory / "native.json"
        self.candidate_path = self.directory / "vllm.json"
        self.baseline_path.write_text(json.dumps(baseline))
        self.candidate_path.write_text(json.dumps(candidate))
        return (self.baseline_path, self.candidate_path)

    def freeze(self, **overrides):
        arguments = {
            "baseline_id": "profile-v1",
            "profile": "27b-mtp2",
            "out": self.directory / "baselines",
            "baseline": self.baseline_path,
            "candidate": self.candidate_path,
            "evidence": [self.evidence],
            "max_ratio": 1.1,
            "require_identical_tokens": False,
            "reproduction": [],
            "max_drift": 0.2,
        }
        arguments.update(overrides)
        return module.freeze(type("Args", (), arguments)())

    def test_a_valid_pair_freezes_with_a_manifest_and_hashes(self):
        self.assertEqual(self.freeze(), 0)
        root = self.directory / "baselines" / "profile-v1"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["baseline_id"], "profile-v1")
        self.assertEqual(manifest["profile_id"], "27b-mtp2")
        self.assertTrue(manifest["validity"]["identity_verified"])
        self.assertEqual(len(manifest["reports"]), 2)
        self.assertEqual([entry["file"] for entry in manifest["evidence"]], ["workload.json"])
        # Every referenced file is present and its hash matches.
        self.assertEqual(module.verify(root), 0)
        self.assertEqual(module.sha256_file(root / "native.json"), manifest["reports"][0]["sha256"])

    def test_an_invalid_pair_never_produces_a_baseline(self):
        # A debug build fails validity, so nothing is frozen even though it was measured.
        candidate = gate.report()
        candidate["identity"]["build"]["release_like"] = False
        candidate["identity"]["verified"] = False
        self.write_reports(candidate)
        with self.assertRaises(ValueError):
            self.freeze()
        self.assertFalse((self.directory / "baselines").exists())

    def test_a_failed_performance_verdict_is_still_a_valid_baseline(self):
        candidate = gate.report()
        for trial in candidate["trials"]:
            trial["wall_seconds"] = 3.0
        self.write_reports(candidate)
        self.assertEqual(self.freeze(), 0)
        manifest = json.loads(
            (self.directory / "baselines" / "profile-v1" / "manifest.json").read_text()
        )
        # The result is recorded honestly rather than being rejected as invalid.
        self.assertFalse(manifest["gate"]["passed"])
        self.assertTrue(manifest["validity"]["identity_verified"])

    def test_an_independent_unit_is_recorded_with_its_drift(self):
        # A second fresh-process pair for the same profile must be gated and its movement from
        # the primary ratios recorded, so a reader can see the statistical basis.
        second_native = self.directory / "native-b.json"
        second_reference = self.directory / "vllm-b.json"
        shifted = gate.report()
        for trial in shifted["trials"]:
            trial["wall_seconds"] = 1.05
        second_native.write_text(json.dumps(shifted))
        second_reference.write_text(json.dumps(gate.report()))
        self.assertEqual(self.freeze(reproduction=[(second_native, second_reference)]), 0)
        root = self.directory / "baselines" / "profile-v1"
        manifest = json.loads((root / "manifest.json").read_text())
        basis = manifest["statistical_basis"]
        self.assertEqual(basis["paired_units"], 2)
        self.assertEqual(len(basis["reproductions"]), 1)
        unit = basis["reproductions"][0]
        self.assertTrue(unit["passed"])
        self.assertTrue(unit["within_declared_tolerance"])
        self.assertGreater(unit["worst_drift"], 0.0)
        # The reproduction's own reports are frozen alongside the primary evidence.
        self.assertIn("native-b.json", [entry["file"] for entry in manifest["evidence"]])
        self.assertEqual(module.verify(root), 0)

    def test_a_reproduction_beyond_the_declared_tolerance_is_flagged(self):
        second_native = self.directory / "native-b.json"
        second_reference = self.directory / "vllm-b.json"
        shifted = gate.report()
        for trial in shifted["trials"]:
            trial["wall_seconds"] = 2.0
        second_native.write_text(json.dumps(shifted))
        second_reference.write_text(json.dumps(gate.report()))
        self.freeze(reproduction=[(second_native, second_reference)], max_drift=0.01)
        root = self.directory / "baselines" / "profile-v1"
        manifest = json.loads((root / "manifest.json").read_text())
        unit = manifest["statistical_basis"]["reproductions"][0]
        # The measurement is kept and flagged rather than rejected: it is a result.
        self.assertFalse(unit["within_declared_tolerance"])
        self.assertGreater(unit["worst_drift"], 0.01)

    def test_a_baseline_without_hardware_evidence_is_rejected(self):
        for mutate in (
            lambda r: r.pop("telemetry"),
            lambda r: r.update(telemetry={"samples": 0}),
            lambda r: r.update(telemetry={"samples": 10, "sha256": None}),
            lambda r: r["telemetry"].pop("active_window"),
            lambda r: [trial.update(gpu=None) for trial in r["trials"]],
        ):
            with self.subTest(mutate=mutate):
                self.write_reports()
                report = json.loads(self.baseline_path.read_text())
                mutate(report)
                self.baseline_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "telemetry"):
                    self.freeze()
                self.assertFalse((self.directory / "baselines").exists())

    def test_a_dirty_source_tree_cannot_be_frozen(self):
        # The revision would not determine the measured binary, so the baseline could not be
        # rebuilt from what it records.
        for role, mutate in (
            ("baseline", lambda r: r["identity"]["source"].update(dirty=True)),
            ("candidate", lambda r: r["identity"]["source"].update(dirty=True)),
        ):
            with self.subTest(role=role):
                self.write_reports()
                report = (
                    json.loads(self.baseline_path.read_text())
                    if role == "baseline"
                    else json.loads(self.candidate_path.read_text())
                )
                mutate(report)
                path = self.baseline_path if role == "baseline" else self.candidate_path
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "uncommitted tree"):
                    self.freeze()
                self.assertFalse((self.directory / "baselines").exists())

    def test_an_existing_baseline_id_is_never_overwritten(self):
        self.freeze()
        with self.assertRaises(SystemExit):
            self.freeze()

    def test_verification_detects_a_changed_file(self):
        self.freeze()
        root = self.directory / "baselines" / "profile-v1"
        (root / "native.json").write_text("{}")
        with self.assertRaises(SystemExit):
            module.verify(root)

    def test_a_baseline_without_a_manifest_is_incomplete(self):
        root = self.directory / "incomplete"
        root.mkdir()
        with self.assertRaises(SystemExit):
            module.verify(root)

    def test_a_path_unsafe_baseline_id_is_rejected(self):
        for baseline_id in ("../escape", "a/b", "", "."):
            with self.subTest(baseline_id=baseline_id), self.assertRaises(SystemExit):
                self.freeze(baseline_id=baseline_id)


if __name__ == "__main__":
    unittest.main()
