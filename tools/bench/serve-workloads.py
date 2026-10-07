#!/usr/bin/env python3
"""Canonical serving workload matrix for cross-engine performance comparison.

The matrix mirrors the released qualification set: four prompt shapes per model
measured with one excluded warmup and three measured repeats. Prompts are built
from fixed text, then tokenized with the native CLI so every engine receives the
same token arrays. Nothing here depends on a local virtual environment.

Usage:
    python3 tools/bench/serve-workloads.py --native-binary BIN \
        --package /path/to/model --output artifacts/inputs.json
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

# Short/long prompts share one body so the long case is a strict superset.
BODY = (
    "Inference engines manage key value caches, schedule requests, and execute "
    "attention and matrix multiplication on GPUs. "
)
TAIL = "Explain the tradeoffs of batching and memory management in detail."

# case -> (body copies, concurrency)
CASES = {"short": (1, 1), "long": (24, 1), "batch4": (1, 4)}
HOT_CASE = "hot_long"
# The hot-prefix case is one long prompt measured once (warmup) and then three
# times; the repeats must hit the prefix cache populated by the warmup.
HOT_BODY_COPIES = 181


def content(case, repeat, slot):
    copies, _ = CASES[case]
    return f"Request {case}-{repeat}-{slot}. " + BODY * copies + TAIL


def hot_content():
    return "Hot prefix. " + BODY * HOT_BODY_COPIES + TAIL


def tokenize(binary, package, text):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump([{"role": "user", "content": text}], handle)
        path = handle.name
    try:
        raw = subprocess.check_output(
            [binary, "tokenize", "--package", package, "--messages", path]
        )
    finally:
        Path(path).unlink()
    return json.loads(raw)["tokens"]


def build(binary, package, repeats):
    rows = []
    for case, (_, concurrency) in CASES.items():
        for repeat in range(-1, repeats):
            rows.extend(
                {
                    "case": case,
                    "repeat": repeat,
                    "slot": slot,
                    "concurrency": concurrency,
                    "tokens": tokenize(binary, package, content(case, repeat, slot)),
                }
                for slot in range(concurrency)
            )
    hot = tokenize(binary, package, hot_content())
    rows.extend(
        {
            "case": HOT_CASE,
            "repeat": repeat,
            "slot": 0,
            "concurrency": 1,
            "tokens": hot,
        }
        for repeat in range(-1, repeats)
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binary", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three measured repeats are required by the serving gate")
    rows = build(args.native_binary, args.package, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    for case in [*CASES, HOT_CASE]:
        tokens = [len(r["tokens"]) for r in rows if r["case"] == case]
        print(f"{case}: {len(tokens)} requests, input tokens {sorted(set(tokens))}")


if __name__ == "__main__":
    main()
