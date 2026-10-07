"""Verify inherited cgroup limits without allocating memory or loading a model."""

import json
import os
import sys
from pathlib import Path


def limits():
    entry = next(
        line
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    group = Path("/sys/fs/cgroup") / entry[3:].lstrip("/")
    values = {
        name: (group / name).read_text().strip()
        for name in ["memory.max", "memory.high", "memory.swap.max", "memory.oom.group"]
    }
    if values["memory.max"] == "max" or int(values["memory.max"]) > 32 * 1024**3:
        raise RuntimeError("Missing benchmark memory limit; use tools/bench/safe-run.sh")
    if values["memory.swap.max"] != "0" or values["memory.oom.group"] != "1":
        raise RuntimeError("Benchmark requires zero swap and group OOM isolation")
    return group, values


if __name__ == "__main__":
    group, values = limits()
    if sys.argv[1:2] == ["--"] and len(sys.argv) > 2:
        os.execvp(sys.argv[2], sys.argv[2:])
    else:
        print(json.dumps({"cgroup": str(group), "limits": values}, indent=2))
