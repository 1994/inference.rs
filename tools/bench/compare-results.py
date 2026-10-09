#!/usr/bin/env python3
"""Gate paired serving benchmark JSON reports; lower latency is better.

This judges three things separately and never lets one stand in for another:

* alignment - both reports describe the same model artifact, resource limits, cache state and
  hardware, and both were produced by a verifiable release build;
* performance - the paired latency ratios against the declared threshold;
* numeric - whether the two sides produced the same tokens, when the caller requires it.

Task quality is not measured here; a report that passes says nothing about answer quality.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

# Telemetry fields a measured trial must carry; without them a latency difference cannot be read
# as engine behaviour rather than an idle device or a CPU-bound stall.
REQUIRED_TELEMETRY = ("utilization_gpu_mean", "utilization_gpu_max", "sm_clock_mean_mhz")

# Per-request latency that must be present and positive; TPOT is optional because a request with
# one visible token has no inter-token interval.
REQUIRED_REQUEST_LATENCY = ("ttft_seconds",)


def require_release_identity(report):
    """Reject a report whose build, model or hardware identity cannot be verified."""
    identity = report.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("report has no identity block")
    if identity.get("verified") is not True:
        raise ValueError("report identity is not verified")
    engine = identity.get("engine")
    if engine == "native":
        build = identity.get("build")
        if not isinstance(build, dict) or build.get("release_like") is not True:
            raise ValueError("native report was not built as a release binary")
        if not (identity.get("source") or {}).get("revision"):
            raise ValueError("native report records no source revision")
    elif not identity.get("engine_version"):
        raise ValueError("reference report records no engine version")
    if not (identity.get("hardware") or {}).get("devices"):
        raise ValueError("report records no hardware identity")
    if not isinstance(identity.get("model"), dict) or not identity["model"].get("files"):
        raise ValueError("report records no model artifact fingerprint")


def align(baseline, candidate):
    """Require the two reports to describe the same experiment conditions."""
    for field in ("model", "hardware"):
        if baseline["identity"].get(field) != candidate["identity"].get(field):
            raise ValueError(f"unaligned {field} identity")
    left_cache = (baseline["identity"].get("cache") or {}).get("effective")
    right_cache = (candidate["identity"].get("cache") or {}).get("effective")
    if left_cache != right_cache:
        raise ValueError("unaligned prefix cache state")
    if baseline.get("gpu_memory_utilization") != candidate.get("gpu_memory_utilization"):
        raise ValueError("unaligned resource constraint: gpu_memory_utilization")
    # Both sides must have matched the same declared profile, or the comparison is between two
    # different experiments that happen to share a workload file.
    left_checklist = baseline.get("checklist")
    right_checklist = candidate.get("checklist")
    if not isinstance(left_checklist, dict) or not isinstance(right_checklist, dict):
        raise ValueError("both reports must be checked against an experiment checklist")
    for side, block in (("baseline", left_checklist), ("candidate", right_checklist)):
        if block.get("compliant") is not True:
            raise ValueError(f"{side} run did not match its experiment checklist")
    if left_checklist.get("sha256") != right_checklist.get("sha256"):
        raise ValueError("the two reports were measured against different checklists")
    left_engine = baseline["identity"].get("engine")
    right_engine = candidate["identity"].get("engine")
    if left_engine == right_engine:
        # An A/B of one engine must hold the serving limits fixed.
        if baseline["identity"].get("limits") != candidate["identity"].get("limits"):
            raise ValueError("unaligned serving limits")
        return
    # Across engines the internal limits differ by construction, so the shared context ceiling is
    # checked explicitly and the experiment profile has to name the pairing.
    profile = baseline["identity"].get("profile_id")
    if not profile or profile != candidate["identity"].get("profile_id"):
        raise ValueError("cross-engine comparison needs one shared profile id")
    native, reference = (baseline, candidate) if left_engine == "native" else (candidate, baseline)
    total = (native["identity"].get("limits") or {}).get("effective_max_model_len")
    declared = (reference["identity"].get("limits") or {}).get("max_model_len")
    if total is None or declared is None or total != declared:
        raise ValueError(
            f"context limits are not aligned across engines: native {total}, reference {declared}"
        )


def measured(report):
    """Validate one report and return its measured trials keyed by case and repeat."""
    if report.get("completed") is not True:
        raise ValueError("benchmark did not complete")
    matrix = report.get("matrix")
    if not isinstance(matrix, dict) or not matrix:
        raise ValueError("report does not declare the measured matrix")
    rows = {}
    for trial in report["trials"]:
        if trial["warmup"]:
            continue
        key = (trial["case"], trial["repeat"])
        if key in rows:
            raise ValueError("duplicate trial")
        declared = matrix.get(trial["case"])
        if not isinstance(declared, dict) or "concurrency" not in declared:
            raise ValueError(f"case is not declared in the matrix: {trial['case']}")
        expected = declared["concurrency"]
        # A partially measured concurrency group is not the case it claims to be.
        if trial["concurrency"] != expected or len(trial["results"]) != expected:
            raise ValueError(
                f"{trial['case']} measured {len(trial['results'])} requests, expected {expected}"
            )
        makespan = trial.get("wall_seconds")
        if makespan is None or not math.isfinite(makespan) or makespan <= 0:
            raise ValueError("missing or invalid group makespan")
        for result in trial["results"]:
            if not result.get("finish_reason") or result.get("truncated"):
                raise ValueError("failed or truncated request")
            for field in REQUIRED_REQUEST_LATENCY:
                value = result.get(field)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError("missing or invalid latency")
            # A single visible token has no inter-token interval; that is unavailable, not bad.
            tpot = result.get("tpot_seconds")
            if tpot is not None and (not math.isfinite(tpot) or tpot < 0):
                raise ValueError("invalid TPOT")
            # When the engine reports its own view, it must agree with what was streamed: a
            # server that generated more than it delivered measured a truncated request.
            server = result.get("server_measurement")
            reported = None if server is None else server.get("output_tokens")
            if reported is not None and reported != len(result["token_ids"]):
                raise ValueError(
                    f"server reported {reported} generated tokens but "
                    f"{len(result['token_ids'])} were streamed"
                )
        # Hardware evidence is part of the measurement, not an optional extra.
        telemetry = trial.get("gpu")
        if not isinstance(telemetry, dict) or any(
            telemetry.get(field) is None for field in REQUIRED_TELEMETRY
        ):
            raise ValueError(f"measured trial has no complete hardware telemetry: {trial['case']}")
        rows[key] = trial
    if not rows:
        raise ValueError("no measured trials")
    for case, declared in matrix.items():
        for repeat in range(declared.get("repeats", 0)):
            if (case, repeat) not in rows:
                raise ValueError(f"matrix row was not measured: {case}:{repeat}")
    hot = {key: trial for key, trial in rows.items() if key[0] == "hot_long"}
    if hot:
        if report.get("hot_prefix_tokens_reused", 0) <= 0:
            raise ValueError("hot cache workload has no observed prefix reuse")
        # Every measured repeat must reuse; a single hit cannot stand for the whole case.
        cold = sorted(
            key[1] for key, trial in hot.items() if trial.get("prefix_tokens_reused", 0) <= 0
        )
        if cold:
            raise ValueError(f"hot cache repeats without observed reuse: {cold}")
    return rows


def availability(values):
    """Report a metric's usable samples, and whether it can be judged at all."""
    usable = [value for value in values if value is not None]
    return usable


def compare(baseline, candidate, max_ratio=1.1, identical=False):
    for field in ("max_new_tokens", "temperature", "mtp_depth", "eos_tokens"):
        if field not in baseline or baseline[field] != candidate.get(field):
            raise ValueError(f"unaligned or missing workload field: {field}")
    for field in ("model", "inputs_sha256", "prefix_cache_enabled"):
        if field not in baseline or baseline[field] != candidate.get(field):
            raise ValueError(f"unaligned or missing workload field: {field}")
    require_release_identity(baseline)
    require_release_identity(candidate)
    align(baseline, candidate)
    old, new = measured(baseline), measured(candidate)
    if old.keys() != new.keys():
        raise ValueError("trial sets differ")
    token_mismatches = 0
    for key, trial in old.items():
        other = new[key]
        left = {r["slot"]: r for r in trial["results"]}
        right = {r["slot"]: r for r in other["results"]}
        if (
            len(left) != len(trial["results"])
            or len(right) != len(other["results"])
            or left.keys() != right.keys()
        ):
            raise ValueError(f"request slots differ: {key}")
        for slot, row in left.items():
            peer = right[slot]
            for item in (row, peer):
                visible = sum(token not in baseline["eos_tokens"] for token in item["token_ids"])
                if visible != item["output_tokens_excluding_eos"]:
                    raise ValueError("reported output count differs from token IDs")
            for field in ("input_tokens", "output_tokens_excluding_eos"):
                if row[field] != peer[field]:
                    raise ValueError(f"request work differs: {key}, {slot}, {field}")
            token_mismatches += row["token_ids"] != peer["token_ids"]
    metrics = []
    for case in sorted({key[0] for key in old}):
        groups = [[v for k, v in rows.items() if k[0] == case] for rows in (old, new)]
        if len(groups[0]) < 3:
            raise ValueError(f"at least three measured repeats required: {case}")
        for field in ("wall_seconds", "ttft_seconds", "tpot_seconds"):
            samples = [
                availability(
                    [t[field] for t in group]
                    if field == "wall_seconds"
                    else [r.get(field) for t in group for r in t["results"]]
                )
                for group in groups
            ]
            if not samples[0] or not samples[1]:
                # No request in the group exposed this metric; report it rather than fail.
                metrics.append(
                    {
                        "case": case,
                        "metric": field,
                        "available": False,
                        "note": "no measured request exposed this metric",
                    }
                )
                continue
            medians = [statistics.median(values) for values in samples]
            ratio = medians[1] / medians[0]
            metrics.append(
                {
                    "case": case,
                    "metric": field,
                    "available": True,
                    "samples": [len(values) for values in samples],
                    "baseline": medians[0],
                    "candidate": medians[1],
                    "ratio": ratio,
                    "passed": ratio <= max_ratio,
                }
            )
    performance = all(m.get("passed", True) for m in metrics)
    hardware = []
    for case in sorted({key[0] for key in old}):
        sides = [
            [trial["gpu"] for key, trial in side.items() if key[0] == case] for side in (old, new)
        ]
        hardware.append(
            {
                "case": case,
                "baseline_utilization_gpu_mean": statistics.fmean(
                    entry["utilization_gpu_mean"] for entry in sides[0]
                ),
                "candidate_utilization_gpu_mean": statistics.fmean(
                    entry["utilization_gpu_mean"] for entry in sides[1]
                ),
                "baseline_utilization_memory_mean": statistics.fmean(
                    entry["utilization_memory_mean"] for entry in sides[0]
                ),
                "candidate_utilization_memory_mean": statistics.fmean(
                    entry["utilization_memory_mean"] for entry in sides[1]
                ),
                "baseline_sm_clock_mean_mhz": statistics.fmean(
                    entry["sm_clock_mean_mhz"] for entry in sides[0]
                ),
                "candidate_sm_clock_mean_mhz": statistics.fmean(
                    entry["sm_clock_mean_mhz"] for entry in sides[1]
                ),
            }
        )
    numeric = {
        "required": identical,
        "identical": token_mismatches == 0,
        "token_mismatches": token_mismatches,
    }
    passed = performance and (not identical or numeric["identical"])
    return {
        "passed": passed,
        "max_latency_ratio": max_ratio,
        "token_mismatches": token_mismatches,
        "identical_tokens_required": identical,
        "performance": {"passed": performance, "metrics": metrics},
        "hardware": hardware,
        "numeric": numeric,
        "quality": {
            "available": False,
            "note": "task quality is verified by its own gate, not here",
        },
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-ratio", type=float, default=1.1)
    parser.add_argument("--require-identical-tokens", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.max_ratio) or args.max_ratio <= 0:
        parser.error("--max-ratio must be positive and finite")
    try:
        result = compare(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
            args.max_ratio,
            args.require_identical_tokens,
        )
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
