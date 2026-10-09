"""The migration ratchet: it must catch growth, and it must not fire on progress."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "test_inventory", Path(__file__).with_name("test-inventory.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inventory(src=10, tests=4, consumers=("a.rs",)):
    return {
        "crates": {"crate-a": {"src_inline": src, "src_near": 0, "tests": tests, "benches": 0}},
        "totals": {
            "src_inline": src,
            "src_near": 0,
            "tests": tests,
            "benches": 0,
            "src": src,
            "all": src + tests,
        },
        "ignored_reasons": {},
        "gated": {},
        "path_mounts": [],
        "executor_consumers": [
            {"path": f"crates/g/{name}", "crate": "g", "patterns": ["test-backends"]}
            for name in consumers
        ],
    }


class ClassifyTest(unittest.TestCase):
    def test_the_plan_s_layout_names(self):
        self.assertEqual(module.classify("tests/integration.rs"), "tests")
        self.assertEqual(module.classify("benches/serving.rs"), "benches")
        self.assertEqual(module.classify("src/resource/waiters/tests.rs"), "src_near")
        self.assertEqual(module.classify("src/storage/shards_tests.rs"), "src_near")
        self.assertEqual(module.classify("src/resident/prefill_gemm/bench_check.rs"), "src_near")
        self.assertEqual(module.classify("src/pipeline/lifecycle.rs"), "src_inline")


class ScanTest(unittest.TestCase):
    def test_entries_are_counted_where_they_live(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src" / "deep").mkdir(parents=True)
            (root / "tests").mkdir()
            (root / "src" / "lib.rs").write_text(
                "#[test]\nfn a() {}\n#[tokio::test]\nasync fn b() {}"
            )
            (root / "src" / "deep" / "tests.rs").write_text("#[test]\nfn c() {}")
            (root / "tests" / "integration.rs").write_text("#[test]\nfn d() {}")
            entries, _ignored, _gated, mounts = module.scan_crate(root)
            self.assertEqual(entries["src_inline"], 2)
            self.assertEqual(entries["src_near"], 1)
            self.assertEqual(entries["tests"], 1)
            self.assertEqual(mounts, [])

    def test_ignore_reasons_and_feature_gates_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text(
                '#[test]\n#[ignore = "needs a GPU"]\nfn a() {}\n'
                '#[cfg(feature = "test-backends")]\n#[test]\nfn b() {}\n'
                '#[cfg(target_os = "macos")]\n#[test]\nfn c() {}\n'
            )
            _entries, ignored, gated, _mounts = module.scan_crate(root)
            self.assertEqual(ignored, {"needs a GPU": 1})
            self.assertEqual(gated, {'feature = "test-backends"': 1, 'target_os = "macos"': 1})

    def test_path_mounts_are_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "mod.rs").write_text('#[cfg(test)]\n#[path = "tests.rs"]\nmod tests;\n')
            (root / "src" / "tests.rs").write_text("#[test]\nfn a() {}")
            _entries, _i, _g, mounts = module.scan_crate(root)
            self.assertEqual(
                mounts,
                [{"crate": root.name, "file": "src/mod.rs", "mount": "tests.rs"}],
            )


class MountTest(unittest.TestCase):
    def tree(self, directory, files):
        root = Path(directory)
        crate = root / "crates" / "group" / "crate-a" / "src"
        crate.mkdir(parents=True)
        for name in files:
            (crate / name).write_text("")

    def test_a_missing_mount_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tree(directory, ["mod.rs"])
            data = inventory()
            data["path_mounts"] = [
                {"crate": "crate-a", "file": "src/mod.rs", "mount": "missing.rs"}
            ]
            problems = module.check_mounts(data, Path(directory))
            self.assertTrue(any("missing" in problem for problem in problems))

    def test_a_resolving_mount_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tree(directory, ["mod.rs", "cases.rs"])
            data = inventory()
            data["path_mounts"] = [{"crate": "crate-a", "file": "src/mod.rs", "mount": "cases.rs"}]
            self.assertEqual(module.check_mounts(data, Path(directory)), [])

    def test_a_duplicate_mount_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tree(directory, ["mod.rs", "cases.rs"])
            mount = {"crate": "crate-a", "file": "src/mod.rs", "mount": "cases.rs"}
            data = inventory()
            data["path_mounts"] = [mount, dict(mount)]
            problems = module.check_mounts(data, Path(directory))
            self.assertTrue(any("times" in problem for problem in problems))


class TargetTest(unittest.TestCase):
    def crate(self, root, name, manifest, files):
        crate = root / "crates" / "group" / name
        (crate / "tests").mkdir(parents=True)
        (crate / "Cargo.toml").write_text(manifest)
        for path in files:
            target = crate / "tests" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("#[test]\nfn a() {}")
        return crate

    def test_a_helper_directory_without_autotests_off_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.crate(root, "crate-a", '[package]\nname = "crate-a"\n', ["unit/a.rs"])
            problems = module.check_targets(root)
            self.assertTrue(any("autotests" in problem for problem in problems))

    def test_an_undeclared_integration_target_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.crate(
                root,
                "crate-a",
                '[package]\nname = "crate-a"\nautotests = false\n',
                ["hardware.rs"],
            )
            problems = module.check_targets(root)
            self.assertTrue(any("hardware.rs" in problem for problem in problems))

    def test_declared_targets_and_helpers_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.crate(
                root,
                "crate-a",
                '[package]\nname = "crate-a"\nautotests = false\n\n'
                '[[test]]\nname = "hardware"\npath = "tests/hardware.rs"\n',
                ["hardware.rs", "unit/a.rs"],
            )
            self.assertEqual(module.check_targets(root), [])


class RatchetTest(unittest.TestCase):
    def test_growth_in_src_is_refused(self):
        recorded = inventory(src=10)
        problems = module.regressions(inventory(src=11), recorded)
        self.assertTrue(any("test bodies in src grew" in problem for problem in problems))

    def test_moving_tests_out_of_src_is_progress(self):
        recorded = inventory(src=10, tests=4)
        # Four entries left src and joined tests/: the plan's direction.
        moved = inventory(src=6, tests=8)
        self.assertEqual(module.regressions(moved, recorded), [])

    def test_losing_cases_is_refused_until_the_destination_is_recorded(self):
        recorded = inventory(src=10, tests=4)
        shrunk = inventory(src=10, tests=2)
        problems = module.regressions(shrunk, recorded)
        self.assertTrue(any("test entries fell" in problem for problem in problems))

    def test_a_relocated_consumer_is_not_a_new_dependency(self):
        # The migration moves these files by design; only the dependency surface matters.
        recorded = inventory(consumers=("a.rs",))
        moved = inventory(consumers=("unit/a.rs",))
        self.assertEqual(module.regressions(moved, recorded), [])

    def test_a_new_executor_consumer_is_refused(self):
        recorded = inventory(consumers=("a.rs",))
        problems = module.regressions(inventory(consumers=("a.rs", "b.rs")), recorded)
        self.assertTrue(any("new consumer" in problem for problem in problems))

    def test_the_recorded_tree_passes_its_own_ratchet(self):
        # The committed snapshot must hold for the committed tree, or the gate is red on arrival.
        recorded = json.loads(module.SNAPSHOT.read_text())
        self.assertEqual(module.regressions(module.collect(), recorded), [])


class RenderTest(unittest.TestCase):
    def test_the_table_ends_with_the_total(self):
        table = module.render(inventory())
        self.assertIn("| crate |", table)
        self.assertIn("**total**", table.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
