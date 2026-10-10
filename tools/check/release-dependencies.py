"""Check the production dependency graphs, not just the host CLI's.

The baseline plan asks for the release feature graphs with normal *and* build edges,
with the duplicate crates, resolved features, lockfile and target identity recorded.
Checking the host CLI without features answers none of that: it hides the CUDA graph,
the target-conditional Metal dependencies and every build script.

The configurations come from the same `build.rs` plan the native build and packaging
use, so this file cannot drift from them. `--record` refreshes the record; the default
run compares against it and fails on a new duplicate, a stale record or a test-only
executor appearing in a production graph.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = Path(__file__).with_name("release-dependencies.json")
# Every production graph must stay free of the CPU test executors and the feature that pulls
# them.
FORBIDDEN = ("infer-backend-host", "infer-backend-reference")
CRATE = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+) v(?P<version>[0-9][^\s]*)")
TREE_SUFFIX = re.compile(r"\s+\(\*\)$")


def load_package_tools():
    """Load the packaging driver, which owns the plan's compilation of `build.rs`."""
    spec = importlib.util.spec_from_file_location(
        "package_tools", ROOT / "tools/package/package.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release_configurations():
    """The production targets and the features `build.rs` resolves for each."""
    tools = load_package_tools()
    configurations = []
    for target in ("x86_64-unknown-linux-gnu", "aarch64-apple-darwin"):
        plan = tools.resolve("auto", target)
        configurations.append(
            {
                "name": plan["platform"],
                "target": plan["target"],
                "backend": plan["backend"],
                "features": plan["features"],
            }
        )
    return configurations


def parse_tree(text):
    """The crates and build-script edges in `cargo tree --prefix none` output."""
    crates = {}
    for line in text.splitlines():
        line = TREE_SUFFIX.sub("", line.strip())
        if not line or line.endswith("(*)"):
            continue
        match = CRATE.match(line)
        if match is not None:
            crates.setdefault(match.group("name"), []).append(match.group("version"))
    duplicates = {name: sorted(set(versions)) for name, versions in crates.items()}
    return {
        "crates": len(crates),
        "edges": sum(len(versions) for versions in crates.values()),
        "duplicates": {
            name: versions for name, versions in duplicates.items() if len(versions) > 1
        },
    }


def cargo_tree(configuration):
    command = [
        "cargo",
        "tree",
        "--offline",
        "--locked",
        "-p",
        "infer-cli",
        "--target",
        configuration["target"],
        "--edges",
        "normal,build",
        "--prefix",
        "none",
    ]
    for feature in configuration["features"]:
        command += ["--features", feature]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return parse_tree(completed.stdout)


def lockfile_sha256():
    return hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest()


def collect():
    return {
        "schema": 1,
        "lockfile_sha256": lockfile_sha256(),
        "configurations": [
            {**configuration, **cargo_tree(configuration)}
            for configuration in release_configurations()
        ],
    }


def mismatches(record, current):
    """Every way a production graph disagrees with the record."""
    problems = []
    if record.get("lockfile_sha256") != current["lockfile_sha256"]:
        problems.append("Cargo.lock changed; re-record with --record and review the difference")
    expected = {entry["name"]: entry for entry in record.get("configurations", [])}
    for entry in current["configurations"]:
        name = entry["name"]
        problems.extend(
            f"{name}: production graph carries {crate}"
            for crate in FORBIDDEN
            if crate in entry["duplicates"]
        )
        previous = expected.get(name)
        if previous is None:
            problems.append(f"{name}: no record for this configuration; re-record")
            continue
        if previous["features"] != entry["features"] or previous["target"] != entry["target"]:
            problems.append(f"{name}: target or resolved features changed; re-record")
        problems.extend(
            f"{name}: {key} changed; re-record and review the difference"
            for key in ("crates", "edges", "duplicates")
            if previous[key] != entry[key]
        )
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="refresh the recorded graph")
    arguments = parser.parse_args()
    current = collect()
    if arguments.record:
        RECORD.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        for entry in current["configurations"]:
            print(
                f"{entry['name']}: {entry['crates']} crates, {entry['edges']} edges, "
                f"{len(entry['duplicates'])} duplicated"
            )
        return 0
    if not RECORD.exists():
        raise SystemExit(f"{RECORD.name} is missing; run with --record")
    problems = mismatches(json.loads(RECORD.read_text()), current)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    for entry in current["configurations"]:
        print(
            f"{entry['name']}: {entry['crates']} crates, {entry['edges']} edges, "
            f"{len(entry['duplicates'])} duplicated, no test executor"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
