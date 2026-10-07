#!/usr/bin/env python3
"""End-to-end MTP A/B benchmark on the production `infer run` path.

Measures decode throughput, TTFT and TPOT at several speculative depths and
concurrency levels, and checks that greedy token sequences stay identical
across depths. Run it through tools/bench/safe-run.sh so the job stays inside
the benchmark memory cgroup.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "/home/r/models/Qwen3.8-27B-NVFP4"


def encode(binary, model, messages):
    path = messages.with_suffix(".messages.json")
    path.write_text(
        json.dumps([{"role": "user", "content": messages.read_text()}], ensure_ascii=False)
    )
    payload = subprocess.check_output(
        [binary, "tokenize", "--package", model, "--messages", str(path)]
    )
    return json.loads(payload)["tokens"]


def run_once(binary, model, prompt, requests, max_new_tokens, depth, work):
    work.mkdir(parents=True, exist_ok=True)
    inputs = [
        {
            "id": index + 1,
            "model": 1,
            "input": {"Sequence": {"tokens": prompt}},
            "workload": {"Generate": {"max_new_tokens": max_new_tokens}},
            "sampling": {"temperature": 0.0, "eos_token": 248046},
        }
        for index in range(requests)
    ]
    requests_path = work / "requests.json"
    requests_path.write_text(json.dumps(inputs))
    config_path = work / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "max_requests": requests,
                "candidate_limit": requests,
                "max_num_seqs": requests,
                "max_request_units": requests,
                "max_num_batched_tokens": 256,
                "workspace_bytes": 268435456,
                "resource_timeout_us": 30000000,
            }
        )
    )
    events_path = work / "events.jsonl"
    command = [
        binary,
        "--backend",
        "cuda",
        "--num-speculative-tokens",
        str(depth),
        "run",
        "--package",
        model,
        "--config",
        str(config_path),
        "--gpu-memory-utilization",
        "0.95",
        "--requests",
        str(requests_path),
        "--events",
        str(events_path),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    wall = time.monotonic() - started
    (work / "stdout.log").write_text(completed.stdout)
    (work / "stderr.log").write_text(completed.stderr)
    record = {
        "tag": f"{work.name}",
        "depth": depth,
        "requests": requests,
        "max_new_tokens": max_new_tokens,
        "wall_s": wall,
    }
    if completed.returncode != 0:
        return record | {"error": completed.stderr.strip().splitlines()[-1][:400]}
    payload = json.loads(completed.stdout)
    record["rows"] = [
        {
            "id": result["request"],
            "tokens": result["output"]["Tokens"],
            "output_tokens": result["measurement"]["output_tokens"],
            "ttft_us": result["measurement"]["ttft_us"],
            "e2e_us": result["measurement"]["e2e_us"],
            "successful": result["measurement"]["successful"],
        }
        for result in payload["results"]
    ]
    steps = 0
    if events_path.exists():
        # Events are written as one JSON array; Submitted/Step counts decode steps.
        for event in json.loads(events_path.read_text()):
            if event.get("kind") == "Submitted" and event.get("object_kind") == "Step":
                steps += 1
    record["steps"] = steps
    return record


def summarize(record):
    if "error" in record:
        return f"{record['tag']}: ERROR {record['error']}"
    rows = record["rows"]
    decode_us = [row["e2e_us"] - row["ttft_us"] for row in rows]
    decode_tokens = [row["output_tokens"] - 1 for row in rows]
    span = max(decode_us) or 1
    tpot_ms = [
        (row["e2e_us"] - row["ttft_us"]) / max(1, row["output_tokens"] - 1) / 1000 for row in rows
    ]
    return (
        "{tag}: ttft_p50={ttft:.0f}ms tpot_p50={tpot:.2f}ms decode_tok/s={rate:.2f} "
        "e2e_p50={e2e:.0f}ms tokens/req={tokens} steps={steps}".format(
            tag=record["tag"],
            ttft=statistics.median(row["ttft_us"] for row in rows) / 1000,
            tpot=statistics.median(tpot_ms),
            rate=sum(decode_tokens) / (span / 1e6),
            e2e=statistics.median(row["e2e_us"] for row in rows) / 1000,
            tokens=rows[0]["output_tokens"],
            steps=record.get("steps", 0),
        )
    )


def greedy_equivalent(records, requests, depths):
    """Missing runs, failed requests and differences at any depth fail acceptance."""
    if len(records) != len(depths) or {r["depth"] for r in records} != set(depths):
        return False
    if any(
        "error" in record
        or len(record.get("rows", [])) != requests
        or not all(row["successful"] for row in record["rows"])
        for record in records
    ):
        return False
    values = [{row["id"]: row["tokens"] for row in record["rows"]} for record in records]
    return bool(values) and all(len(value) == requests and value == values[0] for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", type=Path, required=True, help="plain-text prompt file")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--depths", default="0,2")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--binary", default=str(REPO / "target/release/infer"))
    parser.add_argument("--work", type=Path, default=Path("/tmp/mtp-ab"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt = encode(args.binary, args.model, args.prompt)
    depths = [int(value) for value in args.depths.split(",")]
    records = []
    for rep in range(args.reps):
        for depth in depths:
            work = (
                args.work
                / f"{args.prompt.stem}-n{args.requests}-t{args.max_new_tokens}-rep{rep}-d{depth}"
            )
            print(f"running {work.name} ...", flush=True)
            record = run_once(
                args.binary, args.model, prompt, args.requests, args.max_new_tokens, depth, work
            )
            records.append(record)
            print("   ", summarize(record), flush=True)
    report = {
        "model": args.model,
        "prompt": str(args.prompt),
        "prompt_tokens": len(prompt),
        "requests": args.requests,
        "max_new_tokens": args.max_new_tokens,
        "depths": depths,
        "runs": records,
    }
    print("=== summary")
    for record in records:
        print(summarize(record))
    equivalent = True
    for rep in range(args.reps):
        group = records[rep * len(depths) : (rep + 1) * len(depths)]
        passed = greedy_equivalent(group, args.requests, depths)
        equivalent = equivalent and passed
        print(f"greedy-equivalent rep{rep}: {passed}")
    report["greedy_equivalent"] = equivalent
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote", args.output)
    if not equivalent:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
