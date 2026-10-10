"""Recompute a signed baseline's numbers from the evidence it froze.

`freeze-baseline.py verify` checks that a baseline's files still hash to what the manifest
recorded. That says the evidence has not changed; it says nothing about whether the numbers
in the manifest follow from it. This recomputes them with the same comparison the freeze
used, so a transcription slip, a report edited after signing, or a gate whose definition
moved away from the record all surface as a mismatch rather than as a number nobody re-derived.

A one-unit baseline predates the effective-profile rule, so the recomputation passes
`allow_legacy_baseline`; the frozen pair is a historical comparison by definition.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = Path("benchmarks/baselines")
TOLERANCE = 1e-9


def load_gate():
    """Load the comparison gate, which is a hyphenated script and not an importable module."""
    spec = importlib.util.spec_from_file_location(
        "compare_results", Path(__file__).with_name("compare-results.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def primary_pair(manifest):
    """The two reports the recorded gate compared, in role order."""
    reports = {entry["role"]: entry["file"] for entry in manifest["reports"]}
    missing = {"baseline", "candidate"} - set(reports)
    if missing:
        raise SystemExit(f"manifest records no {sorted(missing)} report")
    return reports["baseline"], reports["candidate"]


def keyed(metrics):
    return {(entry["case"], entry["metric"]): entry for entry in metrics}


def compare_numbers(recorded, recomputed):
    """Every recorded case and metric has to agree with the recomputation."""
    problems = []
    for key, entry in keyed(recorded).items():
        fresh = keyed(recomputed).get(key)
        if fresh is None:
            problems.append(f"{key[0]}:{key[1]} is recorded but not recomputed")
            continue
        if bool(entry.get("available")) != bool(fresh.get("available")):
            problems.append(f"{key[0]}:{key[1]} availability differs")
            continue
        if not entry.get("available"):
            continue
        problems.extend(
            f"{key[0]}:{key[1]} {field}: recorded {entry[field]!r}, recomputed {fresh[field]!r}"
            for field in ("baseline", "candidate", "ratio")
            if abs(entry[field] - fresh[field]) > TOLERANCE * max(1.0, abs(entry[field]))
        )
    extra = set(keyed(recomputed)) - set(keyed(recorded))
    if extra:
        problems.append(f"recomputed but not recorded: {sorted(extra)}")
    return problems


def verify(directory, gate=None, root=ROOT):
    gate = load_gate() if gate is None else gate
    manifest = json.loads((directory / "manifest.json").read_text())
    baseline_file, candidate_file = primary_pair(manifest)
    result = gate.compare(
        json.loads((directory / baseline_file).read_text()),
        json.loads((directory / candidate_file).read_text()),
        manifest["gate"]["max_latency_ratio"],
        allow_legacy_baseline=True,
    )
    problems = compare_numbers(manifest["gate"]["performance"]["metrics"], result["metrics"])
    recorded = manifest["gate"]["performance"].get("passed")
    if bool(recorded) != bool(result["passed"]):
        problems.append(f"verdict: recorded {recorded}, recomputed {result['passed']}")
    return manifest["baseline_id"], problems


def baselines(root=ROOT):
    for path in sorted((root / BASELINES).rglob("manifest.json")):
        yield path.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    failures = []
    for directory in baselines(args.root):
        name, problems = verify(directory, root=args.root)
        if problems:
            failures.append((name, problems))
            for problem in problems:
                print(f"{name}: {problem}", file=sys.stderr)
        else:
            print(f"{name}: numbers recomputed from its frozen reports")
    if failures:
        raise SystemExit(f"{len(failures)} baseline(s) do not recompute")
    print(f"Baselines: {len(list(baselines(args.root)))} recomputed from their evidence")


if __name__ == "__main__":
    main()
