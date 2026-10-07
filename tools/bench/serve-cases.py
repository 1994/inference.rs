#!/usr/bin/env python3
"""Filter a workload file down to selected cases for focused profiling runs."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--cases", nargs="+", required=True)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
rows = json.loads(args.inputs.read_text())
kept = [r for r in rows if r["case"] in args.cases and r["repeat"] < args.repeats]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(kept, indent=2))
for case in args.cases:
    subset = [len(r["tokens"]) for r in kept if r["case"] == case]
    print(f"{case}: {len(subset)} requests {sorted(set(subset))}")
