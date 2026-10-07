"""Summarize only telemetry samples within the measured benchmark window."""

import argparse
import contextlib
import json
import math
import statistics
from pathlib import Path


def stats(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "mean": statistics.mean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "maximum": max(ordered),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    telemetry = [json.loads(line) for line in args.telemetry.read_text().splitlines()]
    if not report.get("completed") or telemetry[-1].get("returncode") != 0:
        raise ValueError("A complete successful monitored run is required")
    start = report["measured_start_unix"]
    end = start + report["wall_seconds"]
    samples = [
        row for row in telemetry if row["type"] == "sample" and start <= row["unix_seconds"] <= end
    ]
    if len(samples) < 2:
        raise ValueError("Insufficient measured-window telemetry")
    fields = telemetry[0]["gpu_fields"]
    metrics, reasons = {}, {}
    for row in samples:
        if row["gpu_returncode"]:
            raise ValueError("NVIDIA telemetry failed")
        for gpu in row["gpu"]:
            device = gpu[1].strip()
            for key, value in zip(fields[2:], gpu[2:], strict=False):
                if key == "clocks_event_reasons.active":
                    counts = reasons.setdefault(device, {})
                    counts[value.strip()] = counts.get(value.strip(), 0) + 1
                with contextlib.suppress(ValueError):
                    metrics.setdefault(f"gpu{device}.{key}", []).append(float(value))
    summary = {
        "framework": report["framework"],
        "mtp_depth": report["mtp_depth"],
        "samples": len(samples),
        "measured_start_unix": start,
        "measured_end_unix": end,
        "first_sample_unix": samples[0]["unix_seconds"],
        "last_sample_unix": samples[-1]["unix_seconds"],
        "gpu": {key: stats(values) for key, values in metrics.items()},
        "clock_reason_counts": reasons,
        "process_tree_cpu_percent_one_core": stats(
            [sum(row["cpu_percent_one_core"].values()) for row in samples]
        ),
        "process_tree_rss_bytes_sum": stats(
            [sum(p["rss_bytes"] for p in row["processes"].values()) for row in samples]
        ),
        "caveats": [
            "GPU metrics are device-wide and include desktop activity",
            "utilization.memory is memory-controller busy time, not percent of peak GB/s",
            "CPU 100% means one logical core; RSS sum can double-count shared pages",
            "1-second telemetry cannot establish kernel occupancy or Tensor Core utilization",
            "Profiler traces must be collected separately from throughput runs",
        ],
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
