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


# The harness reads the declared concurrency from the workload, exactly as serve-workloads.py
# writes it, so the fixtures must carry it too.
CONCURRENCY = {"short": 1, "long": 1, "batch4": 4, "hot_long": 1}


def complete_rows(repeats=3):
    return [
        {
            "case": case,
            "repeat": repeat,
            "slot": slot,
            "concurrency": CONCURRENCY[case],
            "tokens": [1, 2, 3],
        }
        for case in module.CASES
        for repeat in range(-1, repeats)
        for slot in range(CONCURRENCY[case])
    ]


class LoadWorkloadTest(unittest.TestCase):
    def test_complete_workload_passes(self):
        path = write(complete_rows())
        rows = module.load_workload(path, 3)
        self.assertEqual(len(rows), (1 + 1 + 4 + 1) * 4)
        self.assertEqual(module.workload_matrix(rows, 3)["batch4"]["concurrency"], 4)
        path.unlink()

    def test_a_row_without_declared_concurrency_is_rejected(self):
        rows = complete_rows()
        del rows[0]["concurrency"]
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_a_single_slot_batch4_group_is_rejected(self):
        # Four concurrent requests are the point of the case; one slot is not that case.
        rows = [row for row in complete_rows() if not (row["case"] == "batch4" and row["slot"] > 0)]
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_inconsistent_concurrency_within_a_case_is_rejected(self):
        rows = complete_rows()
        for row in rows:
            if row["case"] == "short" and row["repeat"] == 1:
                row["concurrency"] = 2
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
        path.unlink()

    def test_a_case_missing_one_slot_of_a_group_is_rejected(self):
        rows = [
            row
            for row in complete_rows()
            if not (row["case"] == "batch4" and row["repeat"] == 2 and row["slot"] == 3)
        ]
        path = write(rows)
        with self.assertRaises(SystemExit):
            module.load_workload(path, 3)
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

    def test_native_receives_the_same_serving_limits_as_the_reference(self):
        # Otherwise the comparison would measure two different service constraints.
        command = module.build_command(self.arguments(), 1234)
        for flag, value in (("--max-model-len", "8192"), ("--max-num-seqs", "16")):
            self.assertIn(flag, command, flag)
            self.assertEqual(command[command.index(flag) + 1], value, flag)

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

    def test_vllm_command_disables_prefix_caching_when_asked(self):
        arguments = self.arguments(
            engine="vllm", executable=Path("/venv/bin/python"), prefix_cache=False
        )
        command = module.build_command(arguments, 1234)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertNotIn("--enable-prefix-caching", command)

    def test_native_cannot_claim_a_disabled_cache(self):
        arguments = self.arguments(prefix_cache=False)
        with self.assertRaises(RuntimeError):
            module.require_cache_switch(arguments)
        module.require_cache_switch(self.arguments(prefix_cache=True))


def elf(section_names):
    """A minimal ELF64 image whose section table lists exactly the given names."""
    strings = b"\0"
    offsets = {}
    for name in section_names:
        offsets[name] = len(strings)
        strings += name.encode() + b"\0"
    shstrtab = len(section_names) + 1
    count = shstrtab + 1
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # 64-bit
    header[5] = 1  # little endian
    header[6] = 1  # version
    strings_offset = 64 + count * 64
    header[0x28:0x30] = (64).to_bytes(8, "little")
    header[0x34:0x36] = (64).to_bytes(2, "little")
    header[0x3A:0x3C] = (64).to_bytes(2, "little")
    header[0x3C:0x3E] = count.to_bytes(2, "little")
    header[0x3E:0x40] = shstrtab.to_bytes(2, "little")
    sections = bytearray(count * 64)
    for index, name in enumerate(section_names, start=1):
        base = index * 64
        sections[base : base + 4] = offsets[name].to_bytes(4, "little")
    base = shstrtab * 64
    sections[base + 0x18 : base + 0x20] = strings_offset.to_bytes(8, "little")
    sections[base + 0x20 : base + 0x28] = len(strings).to_bytes(8, "little")
    return bytes(header) + bytes(sections) + strings


class IdentityTest(unittest.TestCase):
    def test_the_section_table_decides_whether_a_build_is_release(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            release.write_bytes(elf([".text", ".rodata"]))
            debug = Path(directory) / "debug"
            debug.write_bytes(elf([".text", ".debug_info", ".debug_str"]))
            self.assertEqual(module.binary_identity(release)["image"], "elf")
            self.assertTrue(module.binary_identity(release)["release_like"])
            self.assertEqual(module.binary_identity(release)["debug_sections"], [])
            self.assertFalse(module.binary_identity(debug)["release_like"])
            self.assertIn(".debug_info", module.binary_identity(debug)["debug_sections"])
            self.assertEqual(
                module.binary_identity(release)["sha256"],
                module.sha256_bytes(release.read_bytes()),
            )

    def test_the_section_names_are_read_rather_than_searched_for(self):
        # A release image whose read-only data merely mentions the names must still be release:
        # std's backtrace symbolizer embeds them, and a byte scan would misread that.
        payload = elf([".text"]) + b".debug_info in a string table"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release"
            path.write_bytes(payload)
            self.assertTrue(module.binary_identity(path)["release_like"])
            self.assertEqual(module.binary_identity(path)["debug_sections"], [])

    def test_an_image_whose_format_is_unknown_is_unverifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opaque"
            path.write_bytes(b"not an executable image")
            identity = module.binary_identity(path)
            self.assertEqual(identity["image"], "unknown")
            self.assertIsNone(identity["release_like"])

    def test_model_identity_fingerprints_the_artifacts_that_matter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "tokenizer.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"weights")
            identity = module.model_identity(root)
            self.assertIn("config.json", identity["files"])
            self.assertIn("tokenizer.json", identity["files"])
            self.assertNotIn("golden.json", identity["files"])
            self.assertEqual(identity["shards"], {"model.safetensors": 7})
            # Two packages that differ only in weights are not the same artifact.
            (root / "model.safetensors").write_bytes(b"other")
            self.assertNotEqual(identity["shards"], module.model_identity(root)["shards"])


if __name__ == "__main__":
    unittest.main()
