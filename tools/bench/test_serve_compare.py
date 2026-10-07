"""Contract tests for the portable serving harness without a GPU."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "serve_compare", Path(__file__).with_name("serve-compare.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

workload_path = Path(__file__).with_name("serve-workloads.py")
workload_spec = importlib.util.spec_from_file_location("serve_workloads", workload_path)
workloads = importlib.util.module_from_spec(workload_spec)
workload_spec.loader.exec_module(workloads)


def write(rows):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(rows, handle)
        return Path(handle.name)


def complete_rows(repeats=3):
    return [
        {"case": case, "repeat": repeat, "slot": 0, "tokens": [1, 2, 3]}
        for case in module.CASES
        for repeat in range(-1, repeats)
    ]


class LoadWorkloadTest(unittest.TestCase):
    def test_complete_workload_passes(self):
        path = write(complete_rows())
        self.assertEqual(len(module.load_workload(path, 3)), 16)
        path.unlink()

    def test_missing_repeat_is_rejected(self):
        rows = [row for row in complete_rows() if row["repeat"] != 1]
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_empty_tokens_are_rejected(self):
        rows = complete_rows()
        rows[0]["tokens"] = []
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_input_cap_is_enforced(self):
        path = write(complete_rows())
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3, max_input_tokens=2)
        path.unlink()


class WorkloadMatrixTest(unittest.TestCase):
    def test_long_prompt_is_a_superset_of_short(self):
        short = workloads.content("short", 0, 0)
        long = workloads.content("long", 0, 0)
        body = workloads.BODY
        self.assertEqual(short.count(body), 1)
        self.assertEqual(long.count(body), workloads.CASES["long"][0])
        self.assertIn(body, short)
        self.assertIn(body, long)

    def test_cases_declare_their_concurrency(self):
        for case, (copies, concurrency) in workloads.CASES.items():
            self.assertGreaterEqual(copies, 1, case)
            self.assertGreaterEqual(concurrency, 1, case)
        self.assertEqual(workloads.CASES["batch4"][1], 4)

    def test_hot_prompt_is_longer_than_the_long_case(self):
        hot = workloads.hot_content()
        long = workloads.content("long", 0, 0)
        self.assertGreater(len(hot), len(long))


class CommandTest(unittest.TestCase):
    def arguments(self, **overrides):
        values = {
            "engine": "native",
            "model": Path("/models/x"),
            "executable": Path("/bin/infer"),
            "mtp": 2,
            "gpu_memory_utilization": 0.88,
            "prefix_cache": True,
            "max_model_len": 8192,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 2048,
            "extra": [],
        }
        values.update(overrides)
        return type("Args", (), values)()

    def test_native_command_is_model_and_engine_independent(self):
        command = module.build_command(self.arguments(), 1234)
        self.assertIn("--num-speculative-tokens", command)
        self.assertIn("2", command)
        self.assertNotIn("--enable-prefix-caching", command)

    def test_vllm_command_enables_speculation_and_prefix_cache(self):
        arguments = self.arguments(engine="vllm", executable=Path("/venv/bin/python"))
        command = module.build_command(arguments, 1234)
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--speculative-config", command)

    def test_sglang_command_uses_nextn(self):
        arguments = self.arguments(engine="sglang", executable=Path("/venv/bin/python"))
        command = module.build_command(arguments, 1234)
        self.assertIn("--speculative-algorithm", command)
        self.assertIn("NEXTN", command)


if __name__ == "__main__":
    unittest.main()
