#!/usr/bin/env python3
"""Summarize opt-in CUDA task grants and graph replays without guessing graph widths."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_profile(path):
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    return rows, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def histogram(values):
    return {str(key): count for key, count in sorted(Counter(values).items())}


def summarize(step_rows, event_rows):
    steps = [row for row in step_rows if "tasks" in row]
    if not steps:
        raise ValueError("execution profile contains no steps")
    requests = defaultdict(list)
    grants = []
    mixed = []
    for step in steps:
        decodes = sum(task["kind"] == "decode" for task in step["tasks"])
        for task in step["tasks"]:
            if task["kind"] != "prefill":
                continue
            tokens = task["input_tokens"]
            grants.append(tokens)
            requests[json.dumps(task["request"], sort_keys=True)].append(tokens)
            mixed.append((tokens, decodes))
    if not grants:
        raise ValueError("execution profile contains no prefill tasks")

    groups = defaultdict(list)
    for row in event_rows:
        if row["graph"] in ("prefill", "prefill_last"):
            groups[(row["node_count"], row["graph"])].append(row)
    if not groups:
        raise ValueError("event profile contains no prefill replays")

    return {
        "scope": "All supplied records, including warmup; instrumented times are diagnostic.",
        "execution": {
            "steps": len(steps),
            "failed_steps": sum(not step["successful"] for step in steps),
            "pool_creation_failures": [row for row in step_rows if "pool_creation_failed" in row],
            "prefill_requests": len(requests),
            "prefill_tasks": len(grants),
            "prefill_tokens": sum(grants),
            "grant_tokens_histogram": histogram(grants),
            "tasks_per_prefill_request": histogram(len(v) for v in requests.values()),
            "tokens_per_prefill_request": histogram(sum(v) for v in requests.values()),
            "grant_with_decode_count": [
                {"grant_tokens": tokens, "decode_tasks": decodes, "count": count}
                for (tokens, decodes), count in sorted(Counter(mixed).items())
            ],
            "worker_prompt_width": histogram(
                step["worker_prompt_width"]
                for step in steps
                if step.get("worker_prompt_width") is not None
            ),
        },
        "prefill_graphs": [
            {
                "node_count": nodes,
                "graph": graph,
                "replays": len(rows),
                "tokens": sum(row["tokens"] for row in rows),
                "token_histogram": histogram(row["tokens"] for row in rows),
                "position_mod32": histogram(row["position"] % 32 for row in rows),
                "device_ms": sum(row["total_ms"] for row in rows),
            }
            for (nodes, graph), rows in sorted(groups.items())
        ],
        "interpretation": [
            "Node count groups structures; target/draft identification needs graph metadata.",
            "Token count is active rows, not captured graph width or measured memory traffic.",
            "Compare matching inputs, trial counts, binary, tuning and configuration readbacks.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        steps, step_source = read_profile(args.steps)
        events, event_source = read_profile(args.events)
        result = summarize(steps, events)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    result["sources"] = {"steps": step_source, "events": event_source}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
