"""Reject incomparable experiments and performance/numerical regressions."""

import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare_results", Path(__file__).with_name("compare-results.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def identity(engine="native", **overrides):
    block = {
        "run_id": "run-1",
        "profile_id": "profile-1",
        "engine": engine,
        "engine_version": "0.1.0",
        "source": {"revision": "deadbeef", "dirty": False},
        "release_profile": {"lto": "thin", "codegen-units": 1},
        "build": {"path": "/bin/infer", "sha256": "abc", "release_like": True},
        "model": {"path": "/models/x", "files": {"config.json": "aaa"}, "shards": {"a": 1}},
        "hardware": {"devices": [{"uuid": "GPU-1", "name": "RTX 5090", "driver_version": "1"}]},
        "limits": {"source": "native runtime inspection", "effective_max_model_len": 8192},
        "cache": {"requested": True, "effective": True},
        "verified": True,
    }
    for key, value in overrides.items():
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    return block


def report(case="hot_long", concurrency=1, tokens=2, repeats=3, engine="native"):
    """One report with the identity, matrix and per-request evidence the gate now requires."""
    return {
        "completed": True,
        "model": "fixture",
        "inputs_sha256": "same-input",
        "max_new_tokens": tokens,
        "temperature": 0,
        "mtp_depth": 2,
        "prefix_cache_enabled": True,
        "eos_tokens": [9],
        "hot_prefix_tokens_reused": 20,
        "gpu_memory_utilization": 0.88,
        "matrix": {case: {"concurrency": concurrency, "repeats": repeats}},
        "identity": identity(engine),
        # Both sides of a pair must have matched the same declared profile.
        "checklist": {"path": "profile.json", "sha256": "checklist-1", "compliant": True},
        "telemetry": {"log": "run.telemetry.jsonl", "sha256": "telemetry-1", "samples": 40},
        "trials": [
            {
                "case": case,
                "repeat": repeat,
                "warmup": False,
                "concurrency": concurrency,
                "wall_seconds": 1.0,
                "prefix_tokens_reused": 20 if case == "hot_long" else 0,
                "gpu": {
                    "utilization_gpu_mean": 42.0,
                    "utilization_gpu_max": 88.0,
                    "utilization_memory_mean": 30.0,
                    "sm_clock_mean_mhz": 2400.0,
                },
                "server_process": {"cpu_percent_one_core_mean": 120.0},
                "results": [
                    {
                        "slot": slot,
                        "input_tokens": 20,
                        "output_tokens_excluding_eos": tokens,
                        "token_ids": list(range(1, tokens + 1)),
                        "ttft_seconds": 0.1,
                        "tpot_seconds": 0.9 if tokens > 1 else None,
                        "finish_reason": "length",
                        "truncated": False,
                        "server_measurement": {
                            "ttft_seconds": 0.05,
                            "e2e_seconds": 0.5,
                            "output_tokens": tokens,
                        },
                    }
                    for slot in range(concurrency)
                ],
            }
            for repeat in range(repeats)
        ],
    }


class GateTests(unittest.TestCase):
    def test_equal_and_regressed_latency(self):
        old = report()
        new = copy.deepcopy(old)
        self.assertTrue(module.compare(old, new)["passed"])
        for trial in new["trials"]:
            for result in trial["results"]:
                result["tpot_seconds"] *= 1.2
        result = module.compare(old, new)
        self.assertFalse(result["passed"])
        self.assertFalse(result["performance"]["passed"])

    def test_shorter_generation_cannot_pass_as_faster(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["output_tokens_excluding_eos"] = 1
        new["trials"][0]["results"][0]["token_ids"] = [1]
        new["trials"][0]["results"][0]["server_measurement"]["output_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "request work differs"):
            module.compare(old, new)

    def test_actual_cache_reuse_and_input_alignment_required(self):
        for field, value in [
            ("inputs_sha256", "other"),
            ("mtp_depth", 0),
            ("completed", False),
            ("hot_prefix_tokens_reused", 0),
        ]:
            with self.subTest(field=field):
                old, new = report(), report()
                new[field] = value
                with self.assertRaises(ValueError):
                    module.compare(old, new)

    def test_numerical_gate(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["token_ids"] = [1, 3]
        result = module.compare(old, new, identical=True)
        self.assertFalse(result["passed"])
        self.assertFalse(result["numeric"]["identical"])
        self.assertEqual(result["token_mismatches"], 1)
        # Performance and numeric results stay separate conclusions.
        self.assertTrue(result["performance"]["passed"])

    def test_missing_samples_and_nan_rejected(self):
        old, new = report(), report()
        new["trials"].pop()
        with self.assertRaises(ValueError):
            module.compare(old, new)
        new = report()
        new["trials"][0]["wall_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            module.compare(old, new)

    def test_a_single_slot_group_is_not_the_declared_case(self):
        old = report(case="batch4", concurrency=4)
        new = copy.deepcopy(old)
        new["trials"][0]["results"] = new["trials"][0]["results"][:1]
        with self.assertRaisesRegex(ValueError, "expected 4"):
            module.compare(old, new)

    def test_a_hot_repeat_without_reuse_is_rejected(self):
        old, new = report(), report()
        new["trials"][1]["prefix_tokens_reused"] = 0
        with self.assertRaisesRegex(ValueError, "without observed reuse"):
            module.compare(old, new)

    def test_a_trial_without_hardware_telemetry_is_rejected(self):
        # A latency difference with no device evidence cannot be attributed to the engine.
        old, new = report(), report()
        new["trials"][0]["gpu"] = None
        with self.assertRaisesRegex(ValueError, "hardware telemetry"):
            module.compare(old, new)
        new = report()
        new["trials"][0]["gpu"].pop("utilization_gpu_mean")
        with self.assertRaisesRegex(ValueError, "hardware telemetry"):
            module.compare(old, new)

    def test_the_result_compares_device_utilisation(self):
        old, new = report(), report()
        for trial in new["trials"]:
            trial["gpu"]["utilization_gpu_mean"] = 70.0
        result = module.compare(old, new)
        row = next(entry for entry in result["hardware"] if entry["case"] == "hot_long")
        self.assertEqual(row["baseline_utilization_gpu_mean"], 42.0)
        self.assertEqual(row["candidate_utilization_gpu_mean"], 70.0)
        self.assertIn("baseline_sm_clock_mean_mhz", row)

    def test_a_truncated_request_is_rejected(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["truncated"] = True
        with self.assertRaisesRegex(ValueError, "truncated"):
            module.compare(old, new)
        new = report()
        new["trials"][0]["results"][0]["finish_reason"] = None
        with self.assertRaises(ValueError):
            module.compare(old, new)

    def test_a_server_that_reports_more_tokens_than_it_streamed_is_rejected(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["server_measurement"]["output_tokens"] = 5
        with self.assertRaisesRegex(ValueError, "were streamed"):
            module.compare(old, new)

    def test_a_matrix_row_that_was_never_measured_is_rejected(self):
        old, new = report(), report()
        new["matrix"]["hot_long"]["repeats"] = 4
        with self.assertRaisesRegex(ValueError, "was not measured"):
            module.compare(old, new)

    def test_a_single_visible_token_leaves_tpot_unavailable(self):
        old = report(tokens=1)
        new = copy.deepcopy(old)
        result = module.compare(old, new)
        self.assertTrue(result["passed"])
        tpot = next(m for m in result["metrics"] if m["metric"] == "tpot_seconds")
        self.assertFalse(tpot["available"])
        # The other metrics are still judged.
        self.assertTrue(any(m["available"] for m in result["metrics"]))

    def test_an_unavailable_metric_does_not_mask_a_regression(self):
        old = report(tokens=1)
        new = copy.deepcopy(old)
        for trial in new["trials"]:
            trial["wall_seconds"] = 2.0
        self.assertFalse(module.compare(old, new)["passed"])


class IdentityAlignmentTest(unittest.TestCase):
    def test_a_missing_identity_block_is_rejected(self):
        old, new = report(), report()
        del new["identity"]
        with self.assertRaisesRegex(ValueError, "identity block"):
            module.compare(old, new)

    def test_an_unverified_or_debug_build_is_rejected(self):
        for mutate in (
            lambda block: block.update(verified=False),
            lambda block: block["build"].update(release_like=False),
            lambda block: block["build"].pop("release_like"),
            lambda block: block.update(source={"revision": None, "dirty": True}),
            lambda block: block.update(hardware={"devices": []}),
            lambda block: block.update(model={"path": "/models/x", "files": {}}),
        ):
            with self.subTest(mutate=mutate):
                old, new = report(), report()
                mutate(new["identity"])
                with self.assertRaises(ValueError):
                    module.compare(old, new)

    def test_a_different_model_artifact_is_rejected(self):
        old, new = report(), report()
        new["identity"]["model"]["files"]["config.json"] = "bbb"
        with self.assertRaisesRegex(ValueError, "model identity"):
            module.compare(old, new)

    def test_a_different_gpu_is_rejected(self):
        old, new = report(), report()
        new["identity"]["hardware"]["devices"][0]["uuid"] = "GPU-2"
        with self.assertRaisesRegex(ValueError, "hardware identity"):
            module.compare(old, new)

    def test_a_different_resource_constraint_is_rejected(self):
        old, new = report(), report()
        new["gpu_memory_utilization"] = 0.5
        with self.assertRaisesRegex(ValueError, "gpu_memory_utilization"):
            module.compare(old, new)

    def test_same_engine_requires_identical_serving_limits(self):
        old, new = report(), report()
        new["identity"]["limits"] = {
            "source": "native runtime inspection",
            "effective_max_model_len": 4096,
        }
        with self.assertRaisesRegex(ValueError, "serving limits"):
            module.compare(old, new)

    def test_cross_engine_needs_a_profile_and_an_aligned_context(self):
        old = report()
        new = copy.deepcopy(old)
        new["identity"] = identity(
            "vllm",
            limits={"source": "command line", "max_model_len": 8192},
            build={"path": "/venv/bin/python", "sha256": "x"},
        )
        module.compare(old, new)
        # A missing shared profile makes the pairing unverifiable.
        new["identity"]["profile_id"] = None
        with self.assertRaisesRegex(ValueError, "profile id"):
            module.compare(old, new)
        # A different context ceiling is a different experiment.
        new["identity"] = identity(
            "vllm",
            limits={"source": "command line", "max_model_len": 4096},
            build={"path": "/venv/bin/python", "sha256": "x"},
        )
        with self.assertRaisesRegex(ValueError, "not aligned across engines"):
            module.compare(old, new)


if __name__ == "__main__":
    unittest.main()
