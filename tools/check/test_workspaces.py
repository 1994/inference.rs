"""An independent workspace has to be registered with its reason, and stay independent."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "workspaces", Path(__file__).with_name("workspaces.py")
)
workspaces = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workspaces)

ROOT_MANIFEST = """
[workspace]
members = ["crates/one"]
"""

UNPUBLISHED_PACKAGE = """[package]
name = "infer-reference"
version = "0.1.0"

[workspace]
"""

INDEPENDENT = """
[package]
name = "infer-reference"
version = "0.1.0"
publish = false

[workspace]
"""


class CheckTest(unittest.TestCase):
    def tree(self, manifest=INDEPENDENT, lock=True):
        root = Path(tempfile.mkdtemp())
        (root / "Cargo.toml").write_text(ROOT_MANIFEST)
        (root / "crates/one").mkdir(parents=True)
        (root / "crates/one/Cargo.toml").write_text('[package]\nname = "infer-one"\n')
        (root / "tools/reference").mkdir(parents=True)
        (root / "tools/reference/Cargo.toml").write_text(manifest)
        if lock:
            (root / "tools/reference/Cargo.lock").write_text("")
        return root

    def test_a_registered_workspace_passes(self):
        workspaces.check(
            self.tree(), registered={"tools/reference/Cargo.toml": "a heavy reference framework"}
        )

    def test_an_unregistered_workspace_fails(self):
        with self.assertRaises(SystemExit) as raised:
            workspaces.check(self.tree(), registered={})
        self.assertIn("no recorded reason", str(raised.exception))

    def test_a_stale_entry_fails(self):
        with self.assertRaises(SystemExit) as raised:
            workspaces.check(
                self.tree(),
                registered={
                    "tools/reference/Cargo.toml": "a heavy reference framework",
                    "tools/gone/Cargo.toml": "a workspace that no longer exists",
                },
            )
        self.assertIn("no longer independent", str(raised.exception))

    def test_a_missing_lockfile_fails(self):
        with self.assertRaises(SystemExit) as raised:
            workspaces.check(
                self.tree(lock=False),
                registered={"tools/reference/Cargo.toml": "a heavy reference framework"},
            )
        self.assertIn("without a lockfile", str(raised.exception))

    def test_a_published_reference_workspace_fails(self):
        with self.assertRaises(SystemExit) as raised:
            workspaces.check(
                self.tree(manifest=UNPUBLISHED_PACKAGE),
                registered={"tools/reference/Cargo.toml": "a heavy reference framework"},
            )
        self.assertIn("not published", str(raised.exception))

    def test_a_short_reason_fails(self):
        with self.assertRaises(SystemExit) as raised:
            workspaces.check(self.tree(), registered={"tools/reference/Cargo.toml": "isolation"})
        self.assertIn("has to say something", str(raised.exception))

    def test_the_tree_passes(self):
        workspaces.check()


if __name__ == "__main__":
    unittest.main()
