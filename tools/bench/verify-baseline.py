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
import hashlib
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
    problems.extend(check_window(manifest, directory))
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
    problems.extend(check_reproductions(manifest, directory, gate))
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
    problems.extend(check_telemetry(manifest, directory))
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


def ratios(result):
    """Paired ratios keyed by case and metric, the shape the freeze records drift in."""
    return {
        (entry["case"], entry["metric"]): entry["ratio"]
        for entry in result["metrics"]
        if entry.get("available")
    }


def check_reproductions(manifest, directory, gate):
    """Re-derive how far the independent re-measurement moved the ratios.

    The freeze records the drift per metric, the worst of them and whether it stayed inside the
    declared tolerance. Recording it is not the same as it still following from the two reports,
    so this recomputes each number rather than checking that a number is present.
    """
    basis = manifest["statistical_basis"]
    problems = []
    if not basis.get("reproductions"):
        return problems
    baseline_file, candidate_file = primary_pair(manifest)
    primary = ratios(
        gate.compare(
            json.loads((directory / baseline_file).read_text()),
            json.loads((directory / candidate_file).read_text()),
            manifest["gate"]["max_latency_ratio"],
            allow_legacy_baseline=True,
        )
    )
    for unit in basis.get("reproductions", []):
        files = unit.get("reports", [])
        missing = [name for name in files if not (directory / name).exists()]
        if missing:
            problems.append(f"re-measurement {unit.get('unit')}: reports {missing} are missing")
            continue
        if len(files) != 2:
            problems.append(f"re-measurement {unit.get('unit')}: needs two reports")
            continue
        unit_result = gate.compare(
            json.loads((directory / files[0]).read_text()),
            json.loads((directory / files[1]).read_text()),
            manifest["gate"]["max_latency_ratio"],
            allow_legacy_baseline=True,
        )
        observed = ratios(unit_result)
        shared = sorted(set(primary) & set(observed))
        drift = {
            f"{case}:{metric}": abs(observed[(case, metric)] - primary[(case, metric)])
            / primary[(case, metric)]
            for case, metric in shared
        }
        recorded = unit.get("drift", {})
        if set(recorded) != set(drift):
            problems.append(
                f"re-measurement {unit.get('unit')}: drift keys do not match the reports"
            )
            continue
        problems.extend(
            f"re-measurement {unit.get('unit')} {key}: recorded {recorded[key]!r}, "
            f"recomputed {drift[key]!r}"
            for key in drift
            if abs(recorded[key] - drift[key]) > TOLERANCE * max(1.0, abs(recorded[key]))
        )
        worst = max(drift.values(), default=0.0)
        if abs(unit.get("worst_drift", -1) - worst) > TOLERANCE * max(1.0, worst):
            problems.append(
                f"re-measurement {unit.get('unit')}: worst drift recorded "
                f"{unit.get('worst_drift')!r}, recomputed {worst!r}"
            )
        within = all(value <= basis["declared_max_drift"] for value in drift.values())
        if bool(unit.get("within_declared_tolerance")) != within:
            problems.append(
                f"re-measurement {unit.get('unit')}: the recorded tolerance verdict disagrees"
            )
        if bool(unit.get("passed")) != bool(unit_result["passed"]):
            problems.append(f"re-measurement {unit.get('unit')}: the recorded verdict disagrees")
    return problems


def check_window(manifest, directory):
    """An active window is a subset of the run, so its numbers have to be a subset's numbers."""
    problems = []
    for entry in manifest["reports"]:
        report = json.loads((directory / entry["file"]).read_text())
        window = report.get("telemetry", {}).get("active_window", {})
        samples = report.get("telemetry", {}).get("samples", 0)
        if window.get("samples", 0) > samples:
            problems.append(f"{entry['file']}: active window holds more samples than the run")
        if window.get("utilization_gpu_mean", 0) > window.get("utilization_gpu_max", 0):
            problems.append(f"{entry['file']}: active window mean utilization exceeds its peak")
    return problems


def check_telemetry(manifest, directory):
    """The hardware evidence has to still be there, and still be the evidence that was signed.

    Signing refuses a report without telemetry. After freezing, nothing checked that the
    report's fingerprint still matches the log beside it, that the log still parses, or that it
    still holds the number of samples the report claims. Those are cheap to check and they are
    what makes the hardware evidence evidence rather than a filename.
    """
    problems = []
    for entry in manifest["reports"]:
        report = json.loads((directory / entry["file"]).read_text())
        name = entry["file"]
        telemetry = report.get("telemetry")
        if not isinstance(telemetry, dict) or not telemetry.get("samples"):
            problems.append(f"{name}: no hardware telemetry")
            continue
        if not telemetry.get("sha256"):
            problems.append(f"{name}: telemetry is not fingerprinted")
        if not isinstance(telemetry.get("active_window"), dict):
            problems.append(f"{name}: no active-window telemetry")
        if not any(trial.get("gpu") for trial in report.get("trials", [])):
            problems.append(f"{name}: no trial attributes device telemetry")
        log = directory / str(telemetry.get("log", ""))
        if not log.exists():
            problems.append(f"{name}: telemetry log {log.name} is missing")
            continue
        raw = log.read_bytes()
        if hashlib.sha256(raw).hexdigest() != telemetry["sha256"]:
            problems.append(f"{name}: telemetry log does not match its recorded fingerprint")
        try:
            records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            problems.append(f"{name}: telemetry log is not JSON lines: {error}")
            continue
        if not records or records[0].get("type") != "metadata":
            problems.append(f"{name}: telemetry log does not open with its metadata")
        samples = sum(1 for record in records if record.get("type") == "sample")
        if samples != telemetry["samples"]:
            problems.append(
                f"{name}: telemetry log holds {samples} samples, report records "
                f"{telemetry['samples']}"
            )
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
