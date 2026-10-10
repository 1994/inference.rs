"""The production dependency graph record and the ways it can disagree."""

import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "release_dependencies", Path(__file__).with_name("release-dependencies.py")
)
release_dependencies = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_dependencies)

TREE = """anyhow v1.0.100
hashbrown v0.15.5
serde v1.0.228
hashbrown v0.17.1
syn v2.0.119
syn v3.0.6 (*)
infer-backend-host v0.1.0
"""


class ParseTreeTest(unittest.TestCase):
    def test_counts_crates_and_edges(self):
        parsed = release_dependencies.parse_tree(TREE)
        # Five names: `hashbrown` and `syn` each appear twice, and `syn`'s second line is cargo's
        # repeat marker rather than another edge.
        self.assertEqual(parsed["crates"], 5)
        self.assertEqual(parsed["edges"], 7)

    def test_duplicates_keep_every_version(self):
        duplicates = release_dependencies.parse_tree(TREE)["duplicates"]
        self.assertEqual(duplicates["hashbrown"], ["0.15.5", "0.17.1"])
        self.assertEqual(duplicates["syn"], ["2.0.119", "3.0.6"])
        self.assertNotIn("serde", duplicates)

    def test_the_record_is_only_about_the_graph(self):
        parsed = release_dependencies.parse_tree(TREE)
        self.assertNotIn("infer-backend-host", parsed["duplicates"])


class MismatchTest(unittest.TestCase):
    def setUp(self):
        self.current = {
            "lockfile_sha256": "aa",
            "configurations": [
                {
                    "name": "linux-cuda",
                    "target": "x86_64-unknown-linux-gnu",
                    "features": ["cuda"],
                    "crates": 212,
                    "edges": 704,
                    "duplicates": {"syn": ["2.0.119", "3.0.6"]},
                }
            ],
        }

    def test_an_unchanged_graph_has_no_problems(self):
        self.assertEqual(release_dependencies.mismatches(self.current, self.current), [])

    def test_a_changed_lockfile_needs_a_deliberate_re_record(self):
        record = {**self.current, "lockfile_sha256": "bb"}
        problems = release_dependencies.mismatches(record, self.current)
        self.assertTrue(any("Cargo.lock changed" in problem for problem in problems))

    def test_a_new_duplicate_is_reported(self):
        record = {
            **self.current,
            "configurations": [{**self.current["configurations"][0], "duplicates": {}}],
        }
        problems = release_dependencies.mismatches(record, self.current)
        self.assertTrue(any("duplicates changed" in problem for problem in problems))

    def test_a_test_executor_in_a_production_graph_is_reported(self):
        current = {
            **self.current,
            "configurations": [
                {
                    **self.current["configurations"][0],
                    "duplicates": {"infer-backend-reference": ["0.1.0"]},
                }
            ],
        }
        problems = release_dependencies.mismatches(self.current, current)
        self.assertTrue(any("infer-backend-reference" in problem for problem in problems))

    def test_a_configuration_without_a_record_is_reported(self):
        current = {
            **self.current,
            "configurations": [
                {**self.current["configurations"][0], "name": "macos-metal"},
            ],
        }
        problems = release_dependencies.mismatches(self.current, current)
        self.assertTrue(any("no record" in problem for problem in problems))

    def test_the_host_checks_the_configuration_it_can_resolve_warm(self):
        # Each runner checks the graph its own target matches, so CI covers both without fetching
        # the other platform's crates on every run.
        selected = release_dependencies.release_configurations()
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["target"], release_dependencies.HOST_TARGET[__import__("sys").platform]
        )
        self.assertEqual(len(release_dependencies.release_configurations(everything=True)), 2)

    def test_the_tree_record_is_current(self):
        import json

        record = json.loads(release_dependencies.RECORD.read_text())
        self.assertEqual(
            release_dependencies.mismatches(record, release_dependencies.collect()), []
        )


if __name__ == "__main__":
    unittest.main()
