#!/usr/bin/env python3
"""Gate paired serving benchmark JSON reports; lower latency is better.

This checks workload alignment and measured latency, not model quality or
compute precision. Use --require-identical-tokens for numerical regressions.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def measured(report):
    if report.get("completed") is not True:
        raise ValueError("benchmark did not complete")
    rows = {}
    for trial in report["trials"]:
        if trial["warmup"]:
            continue
        key = (trial["case"], trial["repeat"])
        if key in rows or len(trial["results"]) != trial["concurrency"]:
            raise ValueError("duplicate trial or incomplete concurrency group")
        for latency in [trial["wall_seconds"]] + [
            result[field]
            for result in trial["results"]
            for field in ("ttft_seconds", "tpot_seconds")
        ]:
            if latency is None or not math.isfinite(latency) or latency <= 0:
                raise ValueError("missing or invalid latency")
        rows[key] = trial
    if not rows:
        raise ValueError("no measured trials")
    if any(key[0] == "hot_long" for key in rows) and report.get("hot_prefix_tokens_reused", 0) <= 0:
        raise ValueError("hot cache workload has no observed prefix reuse")
    return rows


def compare(baseline, candidate, max_ratio=1.1, identical=False):
    for field in (
        "model",
        "inputs_sha256",
        "max_new_tokens",
        "temperature",
        "mtp_depth",
        "prefix_cache_enabled",
        "eos_tokens",
    ):
        if field not in baseline or baseline[field] != candidate.get(field):
            raise ValueError(f"unaligned or missing workload field: {field}")
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
    results = []
    for case in sorted({key[0] for key in old}):
        groups = [[v for k, v in rows.items() if k[0] == case] for rows in (old, new)]
        if len(groups[0]) < 3:
            raise ValueError(f"at least three measured repeats required: {case}")
        for field in ("wall_seconds", "ttft_seconds", "tpot_seconds"):
            medians = [
                statistics.median(
                    [t[field] for t in group]
                    if field == "wall_seconds"
                    else [r[field] for t in group for r in t["results"]]
                )
                for group in groups
            ]
            ratio = medians[1] / medians[0]
            results.append(
                {
                    "case": case,
                    "metric": field,
                    "baseline": medians[0],
                    "candidate": medians[1],
                    "ratio": ratio,
                    "passed": ratio <= max_ratio,
                }
            )
    return {
        "passed": all(r["passed"] for r in results) and (not identical or token_mismatches == 0),
        "max_latency_ratio": max_ratio,
        "token_mismatches": token_mismatches,
        "identical_tokens_required": identical,
        "metrics": results,
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
