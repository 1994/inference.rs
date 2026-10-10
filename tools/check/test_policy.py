"""The algorithm-neutrality rule: code in the common layers must not name one algorithm."""

import tempfile
import unittest
from pathlib import Path

import policy


class AlgorithmNeutralityTest(unittest.TestCase):
    def build(self, code: str, layers=None) -> Path:
        root = Path(tempfile.mkdtemp())
        for layer in policy.COMMON_LAYERS if layers is None else layers:
            directory = root / layer
            directory.mkdir(parents=True)
            (directory / "lib.rs").write_text(code)
        return root

    def test_code_naming_an_algorithm_is_rejected(self):
        root = self.build("pub fn draft() -> bool { mtp_depth > 0 }\n")
        with self.assertRaises(SystemExit) as raised:
            policy.check_algorithm_neutrality(root, {})
        self.assertIn("must not name a speculation algorithm", str(raised.exception))

    def test_a_comment_may_name_it(self):
        root = self.build("// MTP draft layers declared by the configuration.\npub fn draft() {}\n")
        policy.check_algorithm_neutrality(root, {})

    def test_a_test_module_may_name_it(self):
        root = self.build("pub fn draft() {}\n#[cfg(test)]\nmod tests {\n    fn mtp() {}\n}\n")
        policy.check_algorithm_neutrality(root, {})

    def test_a_baselined_symbol_passes(self):
        root = self.build("pub mtp_layers: usize,\n", layers=[policy.COMMON_LAYERS[0]])
        policy.check_algorithm_neutrality(root, {"crates/foundation/ir/src/lib.rs": {"mtp_layers"}})

    def test_a_stale_entry_fails_once_the_migration_removes_it(self):
        root = self.build("pub fn draft() {}\n")
        with self.assertRaises(SystemExit) as raised:
            policy.check_algorithm_neutrality(
                root, {"crates/foundation/ir/src/lib.rs": {"mtp_layers"}}
            )
        self.assertIn("Drop the entries", str(raised.exception))

    def test_the_tree_passes(self):
        policy.check_algorithm_neutrality()


if __name__ == "__main__":
    unittest.main()
