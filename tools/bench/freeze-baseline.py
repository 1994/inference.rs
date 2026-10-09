#!/usr/bin/env python3
"""Freeze a gated serving comparison into an immutable, verifiable baseline.

A baseline is issued only when the *validity* checks pass: both reports carry a verified release
identity, the same model artifact and hardware, an aligned cache state and resource constraints,
a complete matrix and no failed or truncated request. The performance verdict is recorded as it
came out; measuring a regression accurately still forms a valid baseline.

The frozen directory holds the reports, the extra evidence (workload, server logs, telemetry), a
machine-readable manifest and the content hashes of everything it references. An existing
baseline id is never overwritten, and ``--verify`` recomputes every hash.

Usage:
    python3 tools/bench/freeze-baseline.py --baseline-id rtx5090-27b-mtp2-v1 \\
        --profile 27b-mtp2-nvfp4 --out benchmarks/baselines/serving \\
        --baseline artifacts/perf/native.json --candidate artifacts/perf/vllm.json \\
        --evidence artifacts/perf/workload.json artifacts/perf/native.server.log
    python3 tools/bench/freeze-baseline.py --verify benchmarks/baselines/serving/rtx5090-27b-mtp2-v1
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# A frozen baseline id names one experiment profile and its evidence; keep it path-safe.
ALLOWED_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gate():
    """Load the report gate, which is a hyphenated script and not an importable module."""
    spec = importlib.util.spec_from_file_location(
        "compare_results", Path(__file__).with_name("compare-results.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_id(baseline_id):
    if not baseline_id or not set(baseline_id) <= ALLOWED_ID or baseline_id in {".", ".."}:
        raise SystemExit(f"baseline id must be a plain path-safe name: {baseline_id!r}")


def copy_evidence(paths, destination):
    """Copy evidence into the baseline and return its recorded hashes."""
    recorded = []
    for path in paths:
        source = Path(path)
        if not source.is_file():
            raise SystemExit(f"evidence file does not exist: {source}")
        target = destination / source.name
        if target.exists():
            raise SystemExit(f"two evidence files share the name {source.name}")
        shutil.copy2(source, target)
        recorded.append({"file": target.name, "sha256": sha256_file(target)})
    return recorded


def require_clean_source(report, role):
    """A frozen baseline must be rebuildable from the revision it names.

    A dirty tree means the recorded revision does not determine the measured binary; the binary
    hash still identifies it, but the baseline could not be reconstructed from the revision.
    """
    source = (report.get("identity") or {}).get("source") or {}
    if source.get("dirty") is not False:
        raise ValueError(
            f"{role} report was measured from an uncommitted tree; commit and re-measure before "
            "freezing a baseline"
        )


def ratios(result):
    """Paired ratios keyed by case and metric, for comparing two measurement units."""
    return {
        (entry["case"], entry["metric"]): entry["ratio"]
        for entry in result["metrics"]
        if entry.get("available")
    }


def reproductions(gate, args, baseline):
    """Gate each independent re-measurement and record how far it moved the ratios.

    The plan requires a fresh-process unit and a declared tolerance; the tolerance is recorded
    next to the observed drift so a reader can see whether it was declared or chosen after the
    fact.
    """
    recorded = []
    primary = ratios(gate.compare(baseline, json.loads(args.candidate.read_text()), args.max_ratio))
    for index, (left, right) in enumerate(args.reproduction, start=1):
        first = json.loads(left.read_text())
        second = json.loads(right.read_text())
        unit = gate.compare(first, second, args.max_ratio, args.require_identical_tokens)
        observed = ratios(unit)
        shared = sorted(set(primary) & set(observed))
        drift = {
            f"{case}:{metric}": abs(observed[(case, metric)] - primary[(case, metric)])
            / primary[(case, metric)]
            for case, metric in shared
        }
        recorded.append(
            {
                "unit": index,
                "reports": [left.name, right.name],
                "passed": unit["passed"],
                "drift": drift,
                "worst_drift": max(drift.values(), default=0.0),
                "within_declared_tolerance": all(
                    value <= args.max_drift for value in drift.values()
                ),
            }
        )
    return recorded


def freeze(args):
    validate_id(args.baseline_id)
    gate = load_gate()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    require_clean_source(baseline, "baseline")
    require_clean_source(candidate, "candidate")
    # Validity failures raise; a failed performance verdict is a result, not an error.
    result = gate.compare(baseline, candidate, args.max_ratio, args.require_identical_tokens)
    units = reproductions(gate, args, baseline)
    for index, (left, right) in enumerate(args.reproduction, start=1):
        require_clean_source(json.loads(left.read_text()), f"reproduction {index} baseline")
        require_clean_source(json.loads(right.read_text()), f"reproduction {index} candidate")
    destination = args.out / args.baseline_id
    if destination.exists():
        raise SystemExit(f"baseline already exists and is never overwritten: {destination}")
    destination.mkdir(parents=True)

    reports = copy_evidence([args.baseline, args.candidate], destination)
    reproduction_files = [path for pair in args.reproduction for path in pair]
    evidence = copy_evidence([*args.evidence, *reproduction_files], destination)
    manifest = {
        "schema": 1,
        "baseline_id": args.baseline_id,
        "profile_id": args.profile,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engines": [
            baseline.get("identity", {}).get("engine"),
            candidate.get("identity", {}).get("engine"),
        ],
        "source": baseline.get("identity", {}).get("source"),
        "hardware": baseline.get("identity", {}).get("hardware"),
        "reports": [
            {"role": role, **record}
            for role, record in zip(("baseline", "candidate"), reports, strict=True)
        ],
        "evidence": evidence,
        "validity": {
            "identity_verified": True,
            "matrix": baseline.get("matrix"),
            "cache_effective": (baseline.get("identity", {}).get("cache") or {}).get("effective"),
        },
        "gate": {
            "max_latency_ratio": result["max_latency_ratio"],
            "passed": result["passed"],
            "performance": result["performance"],
            "numeric": result["numeric"],
            "quality": result["quality"],
        },
        "statistical_basis": {
            "paired_units": 1 + len(units),
            "declared_max_drift": args.max_drift,
            "note": (
                "The baseline plan asks for at least five independent paired units; this record "
                "states how many it actually holds so a consumer can weigh the interval."
            ),
            "reproductions": units,
        },
    }
    # The manifest is written last, so a directory without one is an incomplete freeze.
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"baseline": str(destination), "passed": result["passed"]}, indent=2))
    return 0


def verify(directory):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no manifest in {directory}")
    manifest = json.loads(manifest_path.read_text())
    entries = [*manifest["reports"], *manifest["evidence"]]
    mismatched = []
    for entry in entries:
        path = directory / entry["file"]
        if not path.is_file():
            mismatched.append(f"{entry['file']}: missing")
        elif sha256_file(path) != entry["sha256"]:
            mismatched.append(f"{entry['file']}: content changed")
    if mismatched:
        raise SystemExit("baseline verification failed: " + ", ".join(mismatched))
    print(
        json.dumps(
            {
                "baseline_id": manifest["baseline_id"],
                "profile_id": manifest["profile_id"],
                "verified_files": len(entries),
                "gate_passed": manifest["gate"]["passed"],
            },
            indent=2,
        )
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-id", help="unique id; an existing id is never overwritten")
    parser.add_argument("--profile", help="experiment profile this baseline belongs to")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "baselines" / "serving",
        help="directory that holds the baseline directories",
    )
    parser.add_argument("--baseline", type=Path, help="the fixed side's report")
    parser.add_argument("--candidate", type=Path, help="the candidate side's report")
    parser.add_argument(
        "--evidence",
        type=Path,
        nargs="*",
        default=[],
        help="workload, server logs and telemetry to freeze alongside the reports",
    )
    parser.add_argument("--max-ratio", type=float, default=1.1)
    parser.add_argument("--require-identical-tokens", action="store_true")
    parser.add_argument(
        "--reproduction",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("BASELINE", "CANDIDATE"),
        help="an independent re-measurement of the same profile; repeatable",
    )
    parser.add_argument(
        "--max-drift",
        type=float,
        default=0.2,
        help="declared tolerance for how far a reproduction may move a paired ratio",
    )
    parser.add_argument("--verify", type=Path, help="recheck an existing baseline's hashes")
    args = parser.parse_args()
    if args.verify is not None:
        return verify(args.verify)
    if not (args.baseline_id and args.profile and args.baseline and args.candidate):
        parser.error("--baseline-id, --profile, --baseline and --candidate are required to freeze")
    return freeze(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError) as error:
        print(f"freeze failed: {error}", file=sys.stderr)
        sys.exit(1)
