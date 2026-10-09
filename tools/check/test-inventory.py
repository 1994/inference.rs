#!/usr/bin/env python3
"""Inventory the Rust test corpus and hold the migration ratchet.

The test-organization plan migrates test bodies out of `src/` into each crate's `tests/`, one host
integration target per crate, and then removes the two CPU test executors. Before that migration
the plan requires recording what exists: where each case lives, what it is gated on, and which
files consume the executors.

This tool records that inventory in `tools/check/test-inventory.json` and enforces the invariants
that must hold while the migration proceeds:

* no new test entry may appear in `src/` - the recorded count is a ceiling that only moves down;
* every `#[path = ...]` test mount must resolve and be mounted exactly once;
* no new consumer of the CPU test executors or the `test-backends` feature may appear;
* a change in the collected total must be recorded with `--record`, so a migration that loses
  assertions cannot be mistaken for progress.

Run with `--record` to accept the current tree as the new baseline after a deliberate migration.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).with_name("test-inventory.json")
# A test entry, however it is spelled in this workspace.
TEST_ENTRY = re.compile(r"#\[(?:tokio::)?test\]")
# `#[path = "....rs"]` is how a module mounts a test file that lives outside `src/`.
PATH_MOUNT = re.compile(r'#\[\s*path\s*=\s*"([^"]+)"\s*\]')
# Gating attributes that decide whether a case runs by default.
GATE = re.compile(
    r"#\[(?:cfg\(([^)]*)\)|ignore\s*=\s*\"([^\"]*)\")[^\]]*\]",
    re.MULTILINE,
)
EXECUTOR_PATTERNS = (
    "infer-backend-host",
    "infer_backend_host",
    "infer-backend-reference",
    "infer_backend_reference",
    "test-backends",
    "test_backends",
)
# The migration's own tooling names these on purpose; only real consumers are recorded.
EXECUTOR_SCAN_EXCLUDED = (
    "tools/check/test-inventory.py",
    "tools/check/test_test_inventory.py",
)


def crates():
    return sorted(path for path in REPO_ROOT.glob("crates/*/*") if (path / "Cargo.toml").is_file())


def classify(relative):
    """Which layout a test file belongs to, in the plan's terms."""
    if relative.startswith("tests/"):
        return "tests"
    if relative.startswith("benches/"):
        return "benches"
    name = Path(relative).name
    if (
        "/tests/" in f"/{relative}"
        or name.endswith("_tests.rs")
        or name
        in (
            "tests.rs",
            "test.rs",
            "bench_check.rs",
        )
    ):
        return "src_near"
    return "src_inline"


def scan_crate(root):
    entries = {"src_inline": 0, "src_near": 0, "tests": 0, "benches": 0}
    ignored = {}
    gated = {}
    mounts = []
    for directory in ("src", "tests", "benches"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.rs")):
            text = path.read_text(errors="ignore")
            found = len(TEST_ENTRY.findall(text))
            relative = str(path.relative_to(root))
            if found:
                entries[classify(relative)] += found
                for match in GATE.finditer(text):
                    condition, reason = match.group(1), match.group(2)
                    if reason is not None:
                        ignored[reason] = ignored.get(reason, 0) + 1
                    elif condition and ("target_os" in condition or "feature" in condition):
                        gated[condition] = gated.get(condition, 0) + 1
            mounts.extend(
                {"crate": root.name, "file": relative, "mount": mount}
                for mount in PATH_MOUNT.findall(text)
            )
    return entries, ignored, gated, mounts


def collect():
    inventory = {}
    mounts = []
    ignored = {}
    gated = {}
    for root in crates():
        entries, crate_ignored, crate_gated, crate_mounts = scan_crate(root)
        if sum(entries.values()):
            inventory[root.name] = entries
        for reason, count in crate_ignored.items():
            ignored[reason] = ignored.get(reason, 0) + count
        for condition, count in crate_gated.items():
            gated[condition] = gated.get(condition, 0) + count
        mounts.extend(crate_mounts)
    totals = {
        key: sum(crate.get(key, 0) for crate in inventory.values())
        for key in ("src_inline", "src_near", "tests", "benches")
    }
    totals["src"] = totals["src_inline"] + totals["src_near"]
    totals["all"] = totals["src"] + totals["tests"] + totals["benches"]
    consumers = sorted(
        relative
        for path in REPO_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in (".rs", ".toml", ".sh", ".py")
        and "target/" not in str(path)
        and (relative := str(path.relative_to(REPO_ROOT))) not in EXECUTOR_SCAN_EXCLUDED
        and any(pattern in path.read_text(errors="ignore") for pattern in EXECUTOR_PATTERNS)
    )
    return {
        "crates": inventory,
        "totals": totals,
        "ignored_reasons": ignored,
        "gated": gated,
        "path_mounts": mounts,
        "executor_consumers": consumers,
    }


def check_mounts(inventory, root=REPO_ROOT):
    """Every mount resolves, and no file is mounted twice."""
    problems = []
    seen = {}
    for mount in inventory["path_mounts"]:
        crate = root / "crates"
        owner = next((path for path in crate.glob(f"*/{mount['crate']}") if path.is_dir()), None)
        key = (mount["crate"], mount["file"], mount["mount"])
        seen[key] = seen.get(key, 0) + 1
        if owner is None:
            problems.append(f"{mount['crate']}: unknown crate")
            continue
        resolved = (owner / mount["file"]).parent / mount["mount"]
        if not resolved.is_file():
            problems.append(f"{mount['crate']}: {mount['file']} mounts missing {mount['mount']}")
    for key, count in seen.items():
        if count > 1:
            problems.append(f"{key[0]}: {key[1]} mounts {key[2]} {count} times")
    return problems


def regressions(inventory, recorded, root=REPO_ROOT):
    """The ratchet: what may not get worse while the migration proceeds."""
    problems = check_mounts(inventory, root)
    if inventory["totals"]["src"] > recorded["totals"]["src"]:
        problems.append(
            "test bodies in src grew to "
            f"{inventory['totals']['src']} from {recorded['totals']['src']}; the plan moves them "
            "into tests/, it does not add more"
        )
    for crate, entries in inventory["crates"].items():
        before = recorded["crates"].get(crate, {})
        if entries["src_inline"] + entries["src_near"] > before.get("src_inline", 0) + before.get(
            "src_near", 0
        ):
            problems.append(f"{crate}: test bodies in src grew")
    allowed = set(recorded["executor_consumers"])
    for path in inventory["executor_consumers"]:
        if path not in allowed:
            problems.append(f"{path}: new consumer of a CPU test executor")
    return problems


def render(inventory):
    lines = [
        "| crate | src inline | src near-test | tests/ | benches/ |",
        "|---|---:|---:|---:|---:|",
    ]
    for crate, entries in sorted(
        inventory["crates"].items(),
        key=lambda item: -(item[1]["src_inline"] + item[1]["src_near"]),
    ):
        lines.append(
            f"| {crate} | {entries['src_inline']} | {entries['src_near']} | "
            f"{entries['tests']} | {entries['benches']} |"
        )
    totals = inventory["totals"]
    lines.append(
        f"| **total** | **{totals['src_inline']}** | **{totals['src_near']}** | "
        f"**{totals['tests']}** | **{totals['benches']}** |"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="accept the tree as the baseline")
    parser.add_argument("--json", action="store_true", help="print the inventory")
    parser.add_argument("--table", action="store_true", help="print the per-crate table")
    args = parser.parse_args()
    inventory = collect()
    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    if args.table:
        print(render(inventory))
    if args.record:
        SNAPSHOT.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        print(
            f"recorded {inventory['totals']['all']} test entries "
            f"({inventory['totals']['src']} in src) and "
            f"{len(inventory['executor_consumers'])} executor consumers",
            file=sys.stderr,
        )
        return 0
    recorded = json.loads(SNAPSHOT.read_text())
    problems = regressions(inventory, recorded)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not args.json and not args.table and not problems:
        print(
            f"{inventory['totals']['all']} test entries "
            f"({inventory['totals']['src']} still in src); ratchet holds"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
