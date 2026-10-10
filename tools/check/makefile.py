"""The help text and the real targets agree.

The build plan asks for the responsibilities of build, test, check, package and accept to be
unambiguous. A help line that names a target that does not exist - or a verb whose meaning
drifts, like `test` quietly being a packaging step - is exactly how that goes wrong, so this
checks every entry the help offers against the Makefile that has to provide it.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# An entry is a line that offers one - `make <name>` at the start of a help line - not the word
# "make" inside prose, which is how the artifact-directory sentence once read as `make directory`.
USAGE = re.compile(r"^make ([a-z][a-z0-9-]*)")


def documented(output):
    """Every target the help offers, whether it is written as `make x` or in the gate list."""
    names = set()
    for line in output.splitlines():
        # A help line may head its entry with a label, as `Tests: make test` does.
        stripped = re.sub(r"^[A-Za-z][A-Za-z ]*: ", "", line.strip())
        match = USAGE.match(stripped)
        if match:
            names.add(match.group(1))
    listing = False
    for line in output.splitlines():
        cleaned = re.sub(r"\([^)]*\)", "", line)
        if "Quality gates:" in cleaned:
            listing = True
            cleaned = cleaned.split("Quality gates:", 1)[1]
        elif listing and cleaned.lstrip().startswith("make "):
            break
        if listing:
            names.update(part for part in re.split(r"[,\s]+", cleaned) if part)
    return sorted(name for name in names if re.fullmatch(r"[a-z][a-z0-9-]*", name))


def targets(text):
    """Every target the Makefile defines, including the ones a `$(call ...)` template expands."""
    names = set()
    for line in text.splitlines():
        if line.startswith("\t") or line.lstrip().startswith("#"):
            continue
        head = line.split(":", 1)[0] if ":" in line else ""
        for name in head.split():
            if re.fullmatch(r"[a-z][a-z0-9-]*", name) and "=" not in name:
                names.add(name)
        for expanded in re.findall(r"\$\(eval \$\(call platform_targets,([a-z0-9-]+)\)\)", line):
            names.update(f"{verb}-{expanded}" for verb in ("build", "test", "package"))
    return names


def check(root=ROOT):
    makefile = (Path(root) / "Makefile").read_text()
    help_text = subprocess.run(
        ["make", "help"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    defined = targets(makefile)
    problems = [
        f"help offers `make {name}`, which the Makefile does not provide"
        for name in documented(help_text)
        if name not in defined
    ]
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Makefile entries: {len(documented(help_text))} documented entries all exist")


if __name__ == "__main__":
    sys.exit(check())
