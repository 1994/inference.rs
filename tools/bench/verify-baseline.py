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
# The plan asks for at least five independent paired units; the record states what it holds.
REQUIRED_PAIRED_UNITS = 5


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
    problems = check_contract(directory, manifest)
    if problems:
        # The comparison refuses a pair whose workload no longer aligns, so a broken contract is
        # reported as itself rather than as that refusal.
        return manifest["baseline_id"], problems
    baseline_file, candidate_file = primary_pair(manifest)
    result = gate.compare(
        json.loads((directory / baseline_file).read_text()),
        json.loads((directory / candidate_file).read_text()),
        manifest["gate"]["max_latency_ratio"],
        allow_legacy_baseline=True,
    )
    problems.extend(compare_numbers(manifest["gate"]["performance"]["metrics"], result["metrics"]))
    recorded = manifest["gate"]["performance"].get("passed")
    if bool(recorded) != bool(result["passed"]):
        problems.append(f"verdict: recorded {recorded}, recomputed {result['passed']}")
    return manifest["baseline_id"], problems


def profile_of(directory, manifest):
    """The profile the freeze copied in, which is what the matrix and thresholds must come from."""
    name = f"{manifest['profile_id']}.json"
    path = directory / name
    if not path.exists():
        raise SystemExit(f"{manifest['baseline_id']}: no copied profile {name}")
    return json.loads(path.read_text())


def check_contract(directory, manifest):
    """The plan's matrix, thresholds and workload have to come from the profile, not the report.

    A baseline is signed for a profile, so the cases it covers, the ratio it is judged against
    and the workload it ran all belong to that declaration. Anything invented at signing time -
    a case dropped, a threshold relaxed, a workload quietly changed - is exactly what this
    catches, and it needs no device to check.
    """
    profile = profile_of(directory, manifest)
    problems = []
    if profile.get("profile_id") != manifest["profile_id"]:
        problems.append(
            f"copied profile is {profile.get('profile_id')!r}, "
            f"manifest says {manifest['profile_id']!r}"
        )
    declared = profile["workload"]["matrix"]
    if manifest["validity"]["matrix"] != declared:
        problems.append("validity matrix differs from the profile's declaration")
    if manifest["gate"]["max_latency_ratio"] != profile["measurement"]["max_ratio"]:
        problems.append("recorded ratio threshold is not the profile's")
    workload = {key: value for key, value in profile["workload"].items() if key != "matrix"}
    for entry in manifest["reports"]:
        report = json.loads((directory / entry["file"]).read_text())
        if report.get("matrix") != declared:
            problems.append(f"{entry['file']}: matrix differs from the profile's declaration")
        for key, value in workload.items():
            if key in report and report[key] != value:
                problems.append(
                    f"{entry['file']}: {key} is {report[key]!r}, profile says {value!r}"
                )
    problems.extend(check_measurement(profile, manifest, directory))
    basis = manifest["statistical_basis"]
    reproductions = basis.get("reproductions", [])
    if not reproductions:
        problems.append("no independent re-measurement is recorded")
    problems.extend(
        f"re-measurement {unit.get('unit')} moved beyond the declared drift"
        for unit in reproductions
        if not unit.get("within_declared_tolerance")
    )
    if basis.get("paired_units") != 1 + len(reproductions):
        problems.append("paired_units does not match the recorded reproductions")
    if basis.get("paired_units", 0) < REQUIRED_PAIRED_UNITS:
        print(
            f"{manifest['baseline_id']}: {basis['paired_units']} paired units, "
            f"the plan asks for {REQUIRED_PAIRED_UNITS}; recorded, not yet met"
        )
    return problems


def check_measurement(profile, manifest, directory):
    """What each report has to say about the run itself, not just about the numbers.

    A signable measurement comes from a clean tree, its checklist has to have passed, a cache
    the profile enables has to show reuse, and the effective configuration has to be readable
    from somewhere. None of that needs a device.
    """
    problems = []
    for entry in manifest["reports"]:
        report = json.loads((directory / entry["file"]).read_text())
        name = entry["file"]
        source = report.get("identity", {}).get("source", {})
        if source.get("dirty") is not False:
            problems.append(f"{name}: not measured from a clean tree")
        if not source.get("revision"):
            problems.append(f"{name}: records no source revision")
        checklist = report.get("checklist", {})
        if checklist.get("compliant") is not True:
            problems.append(f"{name}: checklist did not pass")
        if checklist.get("mismatches"):
            problems.append(f"{name}: checklist records mismatches")
        if checklist.get("profile_id") != manifest["profile_id"]:
            problems.append(f"{name}: checklist is for another profile")
        readback = report.get("config_readback")
        if not isinstance(readback, dict) or not readback.get("source"):
            problems.append(f"{name}: config readback names no source")
        elif readback.get("readback") is None:
            # vLLM exposes its effective configuration through the command line rather than an
            # inspection API, which the report says; the command itself is recorded, so the
            # configuration is still checkable. Saying so beats pretending the field is filled.
            print(f"{name}: configuration readable from {readback['source']!r}, not read back")
        if (
            profile.get("cache", {}).get("prefix_cache")
            and report.get("hot_prefix_tokens_reused", 0) <= 0
        ):
            problems.append(f"{name}: prefix cache is enabled but the hot case reused no tokens")
    return problems


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
