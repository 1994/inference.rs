"""Every registered artifact records what produced it.

The build plan requires the reference environment's versions, inputs and generation command
to travel with a golden, so it can be regenerated rather than trusted. A golden that cannot
say where it came from is a number nobody can re-derive.

A golden predating the rule is listed with the clause it misses; the list may only shrink, and
a stale entry fails.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ("examples",)
# Fixture directories state where their inputs came from; the hash of the copied input is the
# identity a reader can check without network access.
SOURCES = "source.json"
SOURCE_KEYS = ("repository", "source")
# Goldens produced before the rule, with the clause they cannot answer yet. Regenerating one
# through its recorded exporter removes its entry.
LEGACY = {
    "examples/qwen3.8-27b/text-golden.json": (
        "no environment version: its exporter needs the target tokenizer and was not recorded"
    ),
    "examples/qwen-hybrid-tiny/readout-golden.json": (
        "no provenance: regenerating with today's libraries produces different values, which is "
        "what recording the versions exists to prevent"
    ),
    "examples/qwen-hybrid-grouped/readout-golden.json": (
        "no provenance: regenerating with today's libraries produces different values"
    ),
}
VERSION_KEYS = ("torch", "transformers", "python", "version")
INPUT_KEYS = ("seed", "revision", "formula", "inputs_sha256", "declared_inputs", "cases")


def nested(document, key):
    """A key at the top level, or under `generator`/`model`, which is where they are recorded."""
    for scope in (document, document.get("generator", {}), document.get("model", {})):
        if isinstance(scope, dict) and key in scope:
            return scope[key]
    return None


def registration(document):
    """Which of the three identities a golden records."""
    producer = document.get("producer") or nested(document, "script")
    versions = [key for key in VERSION_KEYS if nested(document, key)]
    inputs = [key for key in INPUT_KEYS if nested(document, key)]
    return {
        "producer": producer,
        "versions": versions,
        "inputs": inputs,
    }


def missing(document):
    recorded = registration(document)
    gaps = []
    if not recorded["producer"]:
        gaps.append("producer")
    if not recorded["versions"]:
        gaps.append("environment version")
    if not recorded["inputs"]:
        gaps.append("input identity")
    return gaps


def goldens(root=ROOT):
    for directory in GOLDENS:
        yield from sorted((root / directory).rglob("*golden*.json"))


def sources(root=ROOT):
    for directory in GOLDENS:
        yield from sorted((root / directory).rglob(SOURCES))


def check_fixture_sources(root=ROOT):
    """A fixture's provenance has to name its origin and still match the file it describes.

    The upstream revision is not always recorded - a local download has none - so the copied
    input's hash is what makes the fixture checkable offline; a fixture needs one of the two. When
    the hash is there it has to be the real one, or the note describes a file the fixture no longer
    holds. The licence is reported when it is absent rather than required, because a fixture whose
    repository cannot be reached is not a fixture whose licence is known, and inventing one would
    be worse than saying so.
    """
    problems = []
    for path in sources(root):
        name = str(path.relative_to(root))
        document = json.loads(path.read_text())
        problems.extend(f"{name}: records no {key}" for key in SOURCE_KEYS if not document.get(key))
        if not document.get("config_sha256") and not document.get("revision"):
            problems.append(f"{name}: records neither an input hash nor a revision")
        if not document.get("license"):
            print(f"{name}: no licence recorded for its repository")
        digest = document.get("config_sha256")
        config = path.parent / "config.json"
        if digest and config.exists():
            actual = __import__("hashlib").sha256(config.read_bytes()).hexdigest()
            if actual != digest:
                problems.append(f"{name}: config_sha256 does not match the copied config.json")
        if not document.get("revision"):
            print(f"{name}: no upstream revision; the config hash is its identity")
    return problems


def check(root=ROOT, legacy=None):
    legacy = dict(LEGACY if legacy is None else legacy)
    seen = set()
    problems = []
    for path in goldens(root):
        name = str(path.relative_to(root))
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise SystemExit(f"{name}: not valid JSON: {error}") from error
        gaps = missing(document)
        if not gaps:
            continue
        reason = legacy.get(name)
        if reason is None:
            problems.append(
                f"{name}: records no {' and '.join(gaps)}; regenerate it through its exporter "
                "with that identity, or list it in LEGACY with the reason"
            )
            continue
        seen.add(name)
        print(f"{name}: legacy, {reason}")
    problems.extend(check_fixture_sources(root))
    if problems:
        raise SystemExit("\n".join(problems))
    stale = set(legacy) - seen
    if stale:
        raise SystemExit(f"drop the entries that now record their provenance: {sorted(stale)}")
    checked = len(list(goldens(root)))
    print(
        f"Golden provenance: {checked} goldens checked, {len(seen)} legacy, "
        f"{len(list(sources(root)))} fixture sources checked"
    )


if __name__ == "__main__":
    check()
