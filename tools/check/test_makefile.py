"""Help text that offers a target the Makefile lacks is how verb meanings drift."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("makefile", Path(__file__).with_name("makefile.py"))
makefile = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(makefile)

CONSISTENT = """
help:
\t@echo 'Tests: make test'
\t@echo 'Quality gates: check, check-rust,'
\t@echo '  test-cuda, bench'

test:
\t@echo test

check check-rust test-cuda bench:
\t@echo $@
"""

# Only the help line is wrong; the target it should have named exists.
MISSING = CONSISTENT.replace("check, check-rust,", "check, check-rust-typo,")


class DocumentationTest(unittest.TestCase):
    def tree(self, manifest):
        root = Path(tempfile.mkdtemp())
        (root / "Makefile").write_text(manifest)
        return root

    def test_a_consistent_makefile_passes(self):
        makefile.check(self.tree(CONSISTENT))

    def test_a_help_line_without_a_target_fails(self):
        with self.assertRaises(SystemExit) as raised:
            makefile.check(self.tree(MISSING))
        self.assertIn("does not provide", str(raised.exception))

    def test_both_shapes_are_read(self):
        names = makefile.documented(
            "Tests: make test\nQuality gates: check, check-rust,\n  test-cuda, bench\n"
        )
        self.assertEqual(names, ["bench", "check", "check-rust", "test", "test-cuda"])

    def test_the_word_make_in_prose_is_not_an_entry(self):
        names = makefile.documented(
            "Tests: make test\n"
            "  DIST_DIR=artifacts/packages overrides the artifact directory.\n"
            "Quality gates: check\n"
        )
        self.assertEqual(names, ["check", "test"])

    def test_the_tree_agrees(self):
        makefile.check()


if __name__ == "__main__":
    unittest.main()
