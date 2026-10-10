"""The per-entry totals a test entry reports."""

import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "test_report", Path(__file__).with_name("test-report.py")
)
test_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(test_report)

TWO_BINARIES = """running 3 tests
test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
running 2 tests
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out
"""

FAILING = """running 2 tests
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
"""

COMPILED_ONLY = """   Compiling infer-ir v0.1.0
    Finished `test` profile [unoptimized] target(s)
"""


class CommandTest(unittest.TestCase):
    def test_only_the_documented_separator_is_removed(self):
        # `python3 test-report.py --entry x -- cargo test -- --ignored` must reach cargo with its
        # own separator intact, or the flag becomes a test-name filter.
        import subprocess
        import sys

        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("test-report.py")),
                "--entry",
                "inner",
                "--",
                sys.executable,
                "-c",
                "import sys; print('argv:', sys.argv[1:])",
                "--",
                "--ignored",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("['--', '--ignored']", completed.stdout)


class SummarizeTest(unittest.TestCase):
    def test_totals_every_binary(self):
        summary = test_report.summarize("workspace", TWO_BINARIES.splitlines())
        self.assertEqual(summary["state"], "ran")
        self.assertEqual(summary["binaries"], 2)
        self.assertEqual(summary["collected"], 5)
        self.assertEqual(summary["passed"], 5)
        self.assertEqual(summary["ignored"], 1)
        self.assertEqual(summary["result"], "ok")

    def test_a_failure_is_reported(self):
        summary = test_report.summarize("workspace", FAILING.splitlines())
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["result"], "failed")

    def test_compiling_without_running_says_compiled(self):
        summary = test_report.summarize("cross", COMPILED_ONLY.splitlines())
        self.assertEqual(summary["state"], "compiled")
        self.assertEqual(summary["binaries"], 0)
        self.assertEqual(summary["result"], "ok")

    def test_the_line_names_every_number(self):
        text = test_report.line(test_report.summarize("ir", TWO_BINARIES.splitlines()))
        for field in ("ir", "ran", "5 collected", "5 passed", "0 failed", "1 ignored"):
            self.assertIn(field, text)


if __name__ == "__main__":
    unittest.main()
