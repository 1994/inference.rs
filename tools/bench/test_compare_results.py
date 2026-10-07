"""Reject incomparable workloads and performance/numerical regressions."""

import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare_results", Path(__file__).with_name("compare-results.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def report():
    return {
        "completed": True,
        "model": "fixture",
        "inputs_sha256": "same-input",
        "max_new_tokens": 2,
        "temperature": 0,
        "mtp_depth": 2,
        "prefix_cache_enabled": True,
        "eos_tokens": [9],
        "hot_prefix_tokens_reused": 20,
        "trials": [
            {
                "case": "hot_long",
                "repeat": repeat,
                "warmup": False,
                "concurrency": 1,
                "wall_seconds": 1.0,
                "results": [
                    {
                        "slot": 0,
                        "input_tokens": 20,
                        "output_tokens_excluding_eos": 2,
                        "token_ids": [1, 2],
                        "ttft_seconds": 0.1,
                        "tpot_seconds": 0.9,
                    }
                ],
            }
            for repeat in range(3)
        ],
    }


class GateTests(unittest.TestCase):
    def test_equal_and_regressed_latency(self):
        old = report()
        new = copy.deepcopy(old)
        self.assertTrue(module.compare(old, new)["passed"])
        for trial in new["trials"]:
            trial["results"][0]["tpot_seconds"] *= 1.2
        self.assertFalse(module.compare(old, new)["passed"])

    def test_shorter_generation_cannot_pass_as_faster(self):
        old, new = report(), report()
        new["trials"][0]["results"][0]["output_tokens_excluding_eos"] = 1
        new["trials"][0]["results"][0]["token_ids"] = [1]
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
        self.assertEqual(result["token_mismatches"], 1)

    def test_missing_samples_and_nan_rejected(self):
        old, new = report(), report()
        new["trials"].pop()
        with self.assertRaises(ValueError):
            module.compare(old, new)
        new = report()
        new["trials"][0]["wall_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            module.compare(old, new)


if __name__ == "__main__":
    unittest.main()
