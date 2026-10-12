#!/usr/bin/env python3
"""Recompute bounded GSM8K-style task quality from complete, paired model outputs.

This checks answers independently of recorded scores. It does not certify kernel numerics,
model-wide capability, release identity, or serving performance.
"""

import argparse
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

FINAL_ANSWER = re.compile(r"(?m)^\s*####\s*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*(?:\$|%)?\s*$")
FINISH_REASONS = {"Eos", "Length", "stop", "length"}


def numeric_answer(text):
    matches = FINAL_ANSWER.findall(text)
    if not matches:
        return None
    try:
        # Compare Decimal values directly: normalize() can round long numbers to its context.
        value = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def log_binomial_cdf(k, n, probability):
    """Log P[X <= k] for binomial X; remains useful when ordinary probabilities underflow."""
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 0.0 if k == n else -math.inf
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(j + 1)
        - math.lgamma(n - j + 1)
        + j * math.log(probability)
        + (n - j) * math.log1p(-probability)
        for j in range(k + 1)
    ]
    largest = max(terms)
    return min(0.0, largest + math.log(sum(math.exp(t - largest) for t in terms)))


def regression_upper_bound(regressions, count, confidence):
    """One-sided Clopper-Pearson bound; ignoring gains bounds net accuracy loss conservatively."""
    if count < 1 or not 0 <= regressions <= count:
        raise ValueError("invalid paired task count")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if regressions == count:
        return 1.0
    if regressions == 0:
        return -math.expm1(math.log1p(-confidence) / count)
    lower, upper = regressions / count, 1.0
    target = math.log1p(-confidence)
    for _ in range(72):
        middle = (lower + upper) / 2
        if log_binomial_cdf(regressions, count, middle) > target:
            lower = middle
        else:
            upper = middle
    return upper


def validate_suite(suite):
    samples = suite["samples"]
    ids = [sample["id"] for sample in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError("suite needs nonempty, unique task IDs")
    for sample in samples:
        tokens = sample.get("tokens")
        if not tokens or any(
            type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in tokens
        ):
            raise ValueError("suite needs integer token inputs")
        if numeric_answer(sample["reference_answer"]) is None:
            raise ValueError("suite reference needs an explicit numeric final answer")
    acceptance = suite["acceptance"]
    margin = acceptance["noninferiority_margin_accuracy"]
    confidence = acceptance["confidence"]
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("invalid accuracy margin")
    regression_upper_bound(0, len(samples), confidence)
    return samples


def score(suite, report, suite_sha256):
    samples = validate_suite(suite)
    results = report.get("results", [])
    if report.get("completed") is not True or report.get("error"):
        raise ValueError("incomplete or failed task report")
    if report.get("suite_sha256") != suite_sha256:
        raise ValueError("task suite hash differs")
    if report.get("profile_id") != suite["profile_id"]:
        raise ValueError("quality profile differs")
    if [row["id"] for row in results] != [sample["id"] for sample in samples]:
        raise ValueError("missing, duplicate, or reordered task results")
    scored = []
    for sample, row in zip(samples, results, strict=True):
        if row.get("input_token_ids") != sample["tokens"]:
            raise ValueError("task input tokens differ")
        if row.get("variant") != sample["variant"] or row.get("row_id") != sample["row_id"]:
            raise ValueError("task metadata differs")
        if row.get("finish_reason") not in FINISH_REASONS or row.get("truncated") is not False:
            raise ValueError("task request did not finish successfully")
        if row.get("prefix_tokens_reused") != 0:
            raise ValueError("task request reused a prefix")
        finished = row.get("finished")
        if (
            isinstance(finished, dict)
            and (finished.get("measurement") or {}).get("successful") is not True
        ):
            raise ValueError("native task measurement was unsuccessful")
        expected = numeric_answer(sample["reference_answer"])
        predicted = numeric_answer(row["text"])
        correct = predicted is not None and predicted == expected
        if "correct" in row and row["correct"] is not correct:
            raise ValueError("recorded task score disagrees with raw text")
        scored.append(
            {
                "id": sample["id"],
                "variant": sample["variant"],
                "correct": correct,
                "predicted": str(predicted) if predicted is not None else None,
                "expected": str(expected),
                "format_missing": predicted is None,
                "length_limited": row["finish_reason"] in {"Length", "length"},
            }
        )
    return scored


def aggregate(rows):
    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "requests": count,
        "correct": correct,
        "accuracy": correct / count,
        "format_missing": sum(row["format_missing"] for row in rows),
        "length_limited": sum(row["length_limited"] for row in rows),
    }


def compare(suite, baseline, candidate, suite_sha256):
    left, right = [score(suite, report, suite_sha256) for report in (baseline, candidate)]
    for field in ("model", "hardware", "numeric_contract"):
        if not baseline.get(field) or baseline[field] != candidate.get(field):
            raise ValueError(f"quality identity differs or is missing: {field}")
    losses = [
        a["id"] for a, b in zip(left, right, strict=True) if a["correct"] and not b["correct"]
    ]
    gains = [a["id"] for a, b in zip(left, right, strict=True) if not a["correct"] and b["correct"]]
    acceptance = suite["acceptance"]
    upper = regression_upper_bound(len(losses), len(left), acceptance["confidence"])
    margin = acceptance["noninferiority_margin_accuracy"]
    variants = sorted({row["variant"] for row in left})
    return {
        "schema": 1,
        "scope": "fixed arithmetic task subset with declared generation budget",
        "baseline": aggregate(left),
        "candidate": aggregate(right),
        "strata": {
            variant: {
                "baseline": aggregate([r for r in left if r["variant"] == variant]),
                "candidate": aggregate([r for r in right if r["variant"] == variant]),
            }
            for variant in variants
        },
        "regressions": losses,
        "gains": gains,
        "accuracy_change": (len(gains) - len(losses)) / len(left),
        "confidence": acceptance["confidence"],
        "noninferiority_margin_accuracy": margin,
        "regression_rate_upper_bound": upper,
        "method": "one-sided Clopper-Pearson regression bound; gains cannot hide regressions",
        "statistical_assumption": (
            "Bernoulli task regression model; no general-capability guarantee"
        ),
        "quality_status": "pass" if upper <= margin else "not_demonstrated",
        "quality_accepted": upper <= margin,
        "performance_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new evidence path")
    paths = {"suite": args.suite, "baseline": args.baseline, "candidate": args.candidate}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    inputs = {name: json.loads(data) for name, data in raw.items()}
    result = compare(inputs["suite"], inputs["baseline"], inputs["candidate"], hashes["suite"])
    result["evidence"] = {
        name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=2))
    raise SystemExit(0 if result["quality_accepted"] else 1)


if __name__ == "__main__":
    main()
