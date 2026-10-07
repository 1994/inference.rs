#!/usr/bin/env python3
"""Fail-closed native-versus-Candle performance gate for same-device, paired attention measurements.

This checks the native implementation against a performance baseline. Inputs must
include the full fixture matrix and repeated rounds; missing/invalid evidence is an error.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

ROUNDS = 5
SAMPLES = 30
ATOL = 1e-6
RTOL = 1e-4
MAX_REGRESSION = 1.05
MIN_SPEEDUP = 1.0
MAX_DRIFT = 0.10
BASELINE_TOLERANCES = {"f16": (2e-4, 2e-3), "bf16": (2e-3, 2e-2)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, minimum=0):
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= minimum,
        f"invalid {name}: {value}",
    )
    return value


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def index_run(run, manifest):
    require(run["schema"] == 1, "unsupported schema")
    for field in ("fixture_sha256", "contract", "scope"):
        require(run[field] == manifest[field], f"mismatched {field}")
    require(run["build"] == "release", "release build required")
    require(bool(run["implementation"]), "implementation identity required")
    indexed = {}
    cases = {case["id"]: case for case in manifest["cases"]}
    require(len(cases) == len(manifest["cases"]) and cases, "duplicate/empty manifest")
    for record in run["records"]:
        case_id, round_id = record["case"], record["round"]
        require(case_id in cases, f"unexpected case {case_id}")
        require(type(round_id) is int and 0 <= round_id < ROUNDS, "invalid round")
        require((case_id, round_id) not in indexed, "duplicate measurement")
        require(record["shape"] == cases[case_id], f"shape mismatch: {case_id}")
        require(record["warmup"] >= 100, "insufficient warmup")
        require(record["replays_per_sample"] == 8, "incomparable device repetition count")
        samples = record["samples_ms"]
        require(len(samples) >= SAMPLES, "insufficient samples")
        for sample in samples:
            require(finite(sample, "latency") > 0, "nonpositive latency")
        if not cases[case_id].get("rope", False):
            core = record["core_samples_ms"]
            require(len(core) >= SAMPLES, "insufficient core samples")
            for sample in core:
                require(finite(sample, "core latency") > 0, "nonpositive core latency")
        finite(record["max_abs_error"], "error")
        finite(record["reference_max_abs"], "reference scale")
        require(record["finite"] is True, f"nonfinite output: {case_id}")
        require(record["elements"] == cases[case_id]["elements"], "output length mismatch")
        indexed[case_id, round_id] = record
    expected = {(case, r) for case in cases for r in range(ROUNDS)}
    require(set(indexed) == expected, "missing cases or rounds")
    return indexed


def evaluate(manifest, baseline, candidate):
    require(manifest["contract"] == "sdpa-f32-v1", "unknown precision contract")
    for field in ("device", "driver", "toolkit", "fixture_sha256", "scope", "run_id"):
        require(baseline[field] == candidate[field], f"incomparable {field}")
        require(bool(baseline[field]), f"missing {field}")
    require(baseline["precision"] in BASELINE_TOLERANCES, "unknown Candle precision")
    require(candidate["precision"] == "compensated-f32", "native precision must remain F32")
    before = index_run(baseline, manifest)
    after = index_run(candidate, manifest)
    rows, failures, reviews = [], [], []
    for case in manifest["cases"]:
        name = case["id"]
        a = [before[name, r] for r in range(ROUNDS)]
        b = [after[name, r] for r in range(ROUNDS)]
        require(
            len({r["reference_max_abs"] for r in a + b}) == 1,
            f"inconsistent oracle scale: {name}",
        )
        for role, records in (("baseline", a), ("native", b)):
            atol, rtol = (
                BASELINE_TOLERANCES[baseline["precision"]] if role == "baseline" else (ATOL, RTOL)
            )
            failures.extend(
                f"{name}: {role} accuracy (round {record['round']})"
                for record in records
                if record["max_abs_error"] > atol + rtol * record["reference_max_abs"]
            )
        med_a = [statistics.median(r["samples_ms"]) for r in a]
        med_b = [statistics.median(r["samples_ms"]) for r in b]
        ratios = [y / x for x, y in zip(med_a, med_b, strict=True)]
        ratio = statistics.median(ratios)
        tail_ratio = statistics.median(
            percentile(y["samples_ms"], 0.95) / percentile(x["samples_ms"], 0.95)
            for x, y in zip(a, b, strict=True)
        )
        # Require repeatable evidence: a good aggregate cannot hide several bad rounds.
        if ratio > MAX_REGRESSION or tail_ratio > MAX_REGRESSION:
            reviews.append(f"{name}: latency regression")
        if sum(r <= MAX_REGRESSION for r in ratios) < ROUNDS - 1:
            reviews.append(f"{name}: inconsistent paired speedup")
        for role, medians in (("baseline", med_a), ("candidate", med_b)):
            center = statistics.median(medians)
            if max(abs(v / center - 1) for v in medians) > MAX_DRIFT:
                reviews.append(f"{name}: {role} timing drift")
        core_speedup = None
        if not case.get("rope", False):
            core_ratios = [
                statistics.median(y["core_samples_ms"]) / statistics.median(x["core_samples_ms"])
                for x, y in zip(a, b, strict=True)
            ]
            core_tail = statistics.median(
                percentile(y["core_samples_ms"], 0.95) / percentile(x["core_samples_ms"], 0.95)
                for x, y in zip(a, b, strict=True)
            )
            core_speedup = 1 / statistics.median(core_ratios)
            for role, records in (("baseline", a), ("native", b)):
                medians = [statistics.median(r["core_samples_ms"]) for r in records]
                center = statistics.median(medians)
                if max(abs(v / center - 1) for v in medians) > MAX_DRIFT:
                    reviews.append(f"{name}: {role} core timing drift")
            if core_speedup < 1 / MAX_REGRESSION or core_tail > MAX_REGRESSION:
                reviews.append(f"{name}: core latency regression")
            if sum(r <= MAX_REGRESSION for r in core_ratios) < ROUNDS - 1:
                reviews.append(f"{name}: inconsistent core speedup")
        rows.append(
            {
                "case": name,
                "baseline_ms": statistics.median(med_a),
                "native_ms": statistics.median(med_b),
                "speedup": 1 / ratio,
                "core_speedup": core_speedup,
                "p95_ratio": tail_ratio,
                "paired_ratios": ratios,
                "max_abs_error": max(r["max_abs_error"] for r in b),
                "baseline_max_abs_error": max(r["max_abs_error"] for r in a),
            }
        )
    speedup = math.exp(statistics.mean(math.log(r["speedup"]) for r in rows))
    if speedup < MIN_SPEEDUP:
        reviews.append("native geometric mean latency exceeds Candle")
    core_values = [r["core_speedup"] for r in rows if r["core_speedup"] is not None]
    core_speedup = (
        math.exp(statistics.mean(math.log(v) for v in core_values)) if core_values else None
    )
    if core_speedup is not None and core_speedup < MIN_SPEEDUP:
        reviews.append("native geometric mean core latency exceeds Candle")
    return {
        "decision": "reject" if failures else "review" if reviews else "pass",
        "changes_production": False,
        "baseline_precision": baseline["precision"],
        "native_precision": candidate["precision"],
        "geomean_speedup": speedup,
        "core_geomean_speedup": core_speedup,
        "accuracy_failures": failures,
        "performance_reviews": reviews,
        "cases": rows,
        "limits": {"atol": ATOL, "rtol": RTOL, "max_regression": MAX_REGRESSION},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline", "candidate", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(
            *[json.loads(p.read_text()) for p in (args.manifest, args.baseline, args.candidate)]
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as error:
        report = {"decision": "invalid", "changes_production": False, "error": str(error)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    raise SystemExit(0 if report["decision"] == "pass" else 1)


if __name__ == "__main__":
    main()
