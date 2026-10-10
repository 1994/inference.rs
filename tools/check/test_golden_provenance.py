"""A golden has to say what produced it, or be listed as one that predates the rule."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "golden_provenance", Path(__file__).with_name("golden-provenance.py")
)
golden_provenance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(golden_provenance)

MINIMAL = {
    "producer": "fixture producer",
    "torch": "2.14.1",
    "seed": 1,
    "values": [1.0],
}


class RegistrationTest(unittest.TestCase):
    def test_a_complete_registration_has_no_gaps(self):
        self.assertEqual(golden_provenance.missing(MINIMAL), [])

    def test_a_generator_script_counts_as_the_producer(self):
        document = {
            "generator": {"script": "tools/fixtures/x.py", "python": "3.13"},
            "formula": "value = index",
        }
        self.assertEqual(golden_provenance.missing(document), [])

    def test_each_missing_clause_is_named(self):
        self.assertEqual(
            golden_provenance.missing({"values": [1.0]}),
            ["producer", "environment version", "input identity"],
        )
        self.assertEqual(
            golden_provenance.missing({"producer": "x", "torch": "1", "values": []}),
            ["input identity"],
        )


class CheckTest(unittest.TestCase):
    def tree(self, document):
        root = Path(tempfile.mkdtemp())
        (root / "examples/case").mkdir(parents=True)
        (root / "examples/case/golden.json").write_text(json.dumps(document))
        return root

    def test_a_tree_of_registered_goldens_passes(self):
        golden_provenance.check(self.tree(MINIMAL), legacy={})

    def test_an_unregistered_golden_fails(self):
        with self.assertRaises(SystemExit) as raised:
            golden_provenance.check(self.tree({"values": []}), legacy={})
        self.assertIn("environment version", str(raised.exception))

    def test_a_legacy_entry_explains_instead(self):
        root = self.tree({"values": []})
        golden_provenance.check(root, legacy={"examples/case/golden.json": "predates the rule"})

    def test_a_stale_entry_fails(self):
        with self.assertRaises(SystemExit) as raised:
            golden_provenance.check(
                self.tree(MINIMAL), legacy={"examples/case/golden.json": "predates the rule"}
            )
        self.assertIn("drop the entries", str(raised.exception))

    def test_the_tree_passes(self):
        golden_provenance.check()


if __name__ == "__main__":
    unittest.main()
