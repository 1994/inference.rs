"""A golden has to say what produced it, or be listed as one that predates the rule."""

import hashlib
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


class FixtureSourceTest(unittest.TestCase):
    def tree(self, source, config=b'{"hidden_size": 1}'):
        root = Path(tempfile.mkdtemp())
        directory = root / "examples/case"
        directory.mkdir(parents=True)
        (directory / "source.json").write_text(json.dumps(source))
        if config is not None:
            (directory / "config.json").write_bytes(config)
        return root

    def source(self, **overrides):
        document = {
            "repository": "owner/model",
            "source": "https://example.invalid/config.json",
            "config_sha256": hashlib.sha256(b'{"hidden_size": 1}').hexdigest(),
        }
        document.update(overrides)
        return document

    def test_a_matching_hash_passes(self):
        self.assertEqual(golden_provenance.check_fixture_sources(self.tree(self.source())), [])

    def test_a_hash_that_does_not_match_is_caught(self):
        root = self.tree(self.source(config_sha256="0" * 64))
        problems = golden_provenance.check_fixture_sources(root)
        self.assertIn("does not match", " ".join(problems))

    def test_an_input_identity_is_required(self):
        root = self.tree(self.source(config_sha256=None))
        problems = golden_provenance.check_fixture_sources(root)
        self.assertIn("neither an input hash nor a revision", " ".join(problems))

    def test_a_revision_alone_identifies_the_input(self):
        root = self.tree(self.source(config_sha256=None, revision="abc123"))
        self.assertEqual(golden_provenance.check_fixture_sources(root), [])

    def test_a_missing_origin_is_caught(self):
        root = self.tree(self.source(repository=None, source=None))
        problems = golden_provenance.check_fixture_sources(root)
        self.assertIn("records no repository", " ".join(problems))
        self.assertIn("records no source", " ".join(problems))

    def test_a_missing_licence_is_only_a_note(self):
        root = self.tree(self.source())
        self.assertEqual(golden_provenance.check_fixture_sources(root), [])


if __name__ == "__main__":
    unittest.main()
