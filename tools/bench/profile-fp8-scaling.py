#!/usr/bin/env python3
"""Attribute FP8 scale kernels using an IR consumer census and a saved Nsight trace.

Only identify a width when every FP8 projection at that width has a sole adjacent
Add consumer and the captured node count matches the census. Kernel time is an
upper bound on removable work, not a wall-time saving.
"""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(database, census, scale_columns):
    total = Counter()
    for row in census["counts"]:
        total[row["columns"]] += row["count"]
    eligible = Counter(
        row["columns"]
        for row in census["single_add_consumers"]
        if row["adjacent"] and row["last_use"] == row["add"]
    )
    widths = defaultdict(set)
    for columns in total:
        widths[(columns + scale_columns - 1) // scale_columns].add(columns)
    graphs = [
        row[0]
        for row in database.execute(
            "SELECT DISTINCT graphId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.shortName "
            "WHERE s.value=? AND graphId IS NOT NULL",
            ("chunk_scan_entry",),
        )
    ]
    identified, unresolved = [], []
    for graph in graphs:
        for rows, tiles, nodes, calls, seconds in database.execute(
            "SELECT gridX,gridY,COUNT(DISTINCT graphNodeId),COUNT(*),SUM(end-start)/1e9 "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName "
            "WHERE graphId=? AND s.value=? AND gridZ=1 GROUP BY gridX,gridY",
            (graph, "scale_entry"),
        ):
            entry = {
                "graph_id": graph,
                "captured_rows": rows,
                "column_tiles": tiles,
                "nodes": nodes,
                "calls": calls,
                "kernel_seconds": seconds,
            }
            candidates = widths[tiles]
            if len(candidates) != 1:
                entry["reason"] = "column width is not uniquely identified"
            else:
                columns = next(iter(candidates))
                entry["columns"] = columns
                if eligible[columns] != total[columns]:
                    entry["reason"] = "width has mixed consumers; time is not apportioned"
                elif nodes == 0 or nodes != eligible[columns] or calls % nodes:
                    entry["reason"] = "captured node/replay count does not match the census"
                else:
                    entry["replays"] = calls // nodes
                    identified.append(entry)
                    continue
            unresolved.append(entry)
    resources = []
    for name in ("norm_entry", "maximum_entry", "scale_entry", "binary_entry"):
        for threads, registers, local_bytes, seconds in database.execute(
            "SELECT blockX*blockY*blockZ,registersPerThread,localMemoryPerThread,"
            "SUM(end-start)/1e9 FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.shortName WHERE s.value=? "
            "GROUP BY blockX,blockY,blockZ,registersPerThread,localMemoryPerThread",
            (name,),
        ):
            resources.append(
                {
                    "kernel": name,
                    "threads": threads,
                    "registers_per_thread": registers,
                    "local_bytes_per_thread": local_bytes,
                    "kernel_seconds": seconds,
                }
            )
    return {
        "diagnostic_only": True,
        "scope": "Target prompt graphs identified by chunk_scan_entry; other graphs excluded.",
        "scale_tile_columns": scale_columns,
        "identified_scale_add": identified,
        "identified_kernel_seconds_upper_bound": sum(
            entry["kernel_seconds"] for entry in identified
        ),
        "unresolved_or_ineligible_scale_groups": unresolved,
        "cupti_reported_resources": resources,
        "limitations": [
            "Kernel sums exclude launch gaps and are not serving wall-time measurements.",
            "An eligible kernel's entire time is only an upper bound on fusion savings.",
            "Static IR matching does not prove capture-time buffer aliasing or numeric equality.",
            "Match capture/JIT specialization before comparing CUPTI and standalone resources.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("census", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale-tile-columns", type=int, default=256)
    args = parser.parse_args()
    if args.scale_tile_columns <= 0:
        parser.error("scale tile columns must be positive")
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        result = summarize(db, json.loads(args.census.read_text()), args.scale_tile_columns)
    result["sources"] = {
        key: {"path": str(path.resolve()), "sha256": sha256(path)}
        for key, path in (("database", args.database), ("census", args.census))
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
