"""Summarize complete suite reports; never average per-request token rates."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(report):
    rows = report["results"]
    expected = [row["id"] for row in report["dataset"]["samples"]]
    if not report.get("completed") or [row["id"] for row in rows] != expected:
        raise ValueError("Incomplete or reordered suite")
    seconds = report["wall_seconds"]
    eos = report["generation"]["sampling"]["eos_tokens"]
    # vLLM can remove its final stop token; exclude EOS from both numerators.
    output_tokens = sum(
        sum(
            token not in eos
            for token in row.get("token_ids", row.get("decode", {}).get("tokens", []))
        )
        for row in rows
    )
    input_tokens = sum(len(row["input_tokens"]) for row in rows)
    return {
        "framework": report["framework"],
        "mtp_depth": report["mtp_depth"],
        "requests": len(rows),
        "completed": True,
        "wall_seconds": seconds,
        "load_seconds": report["load_seconds"],
        "input_tokens": input_tokens,
        "output_tokens_excluding_eos": output_tokens,
        "requests_per_second": len(rows) / seconds,
        "output_tokens_per_second": output_tokens / seconds,
        "total_tokens_per_second": (input_tokens + output_tokens) / seconds,
        "length_limited": sum(row["finish_reason"] == "length" for row in rows),
        "metric": "resident model, sequential full-suite wall time, excludes load and warmup",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    base = reports[0]
    for report in reports[1:]:
        for key in ["dataset", "generation", "max_new_tokens", "concurrency", "model"]:
            if report[key] != base[key]:
                raise ValueError(f"Incompatible benchmark field: {key}")
        if [r["input_tokens"] for r in report["results"]] != [
            r["input_tokens"] for r in base["results"]
        ]:
            raise ValueError("Input token sequences differ")
    results = [
        dict(
            source=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **summarize(report),
        )
        for path, report in zip(args.reports, reports, strict=False)
    ]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
