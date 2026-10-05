"""Timestamped NVIDIA/host telemetry. Attach to a PID or wrap a benchmark command."""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

FIELDS = "timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active"


def process_tree(root):
    entries = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            entries[int(path.parent.name)] = dict(
                ppid=int(fields[1]), ticks=int(fields[11]) + int(fields[12]),
                rss_bytes=int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                threads=int(fields[17]), state=fields[0],
            )
        except (OSError, ValueError, IndexError):
            continue
    selected = {root}
    while True:
        expanded = selected | {pid for pid, data in entries.items() if data["ppid"] in selected}
        if expanded == selected:
            break
        selected = expanded
    return {pid: entries[pid] for pid in selected if pid in entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if bool(args.pid) == bool(command) or args.interval < 0.1:
        parser.error("provide either --pid or a command; interval >= 0.1")
    # A shared lock prevents overlapping benchmark wrappers on this workspace GPU.
    with Path("artifacts/benchmark-gpu.lock").open("a") as lock, args.output.open("x") as output:
        fcntl.flock(lock, fcntl.LOCK_EX)
        process = subprocess.Popen(command) if command else None
        pid = process.pid if process else args.pid
        started = time.time()
        previous, previous_time = {}, time.monotonic()
        output.write(json.dumps(dict(type="metadata", pid=pid, command=command,
                                     started_unix=started, interval_seconds=args.interval,
                                     attached=process is None, gpu_fields=FIELDS.split(","),
                                     cpu_count=os.cpu_count(), clock_ticks=os.sysconf("SC_CLK_TCK"))) + "\n")
        while process.poll() is None if process else Path(f"/proc/{pid}").exists():
            sample_start = time.monotonic()
            gpu = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True)
            current = process_tree(pid)
            elapsed = sample_start - previous_time
            cpu = {key: 100 * (value["ticks"] - previous[key]["ticks"])
                   / os.sysconf("SC_CLK_TCK") / elapsed
                   for key, value in current.items() if key in previous and elapsed > 0}
            output.write(json.dumps(dict(type="sample", unix_seconds=time.time(),
                gpu=list(csv.reader(gpu.stdout.splitlines())), gpu_error=gpu.stderr,
                gpu_returncode=gpu.returncode, processes=current,
                cpu_percent_one_core=cpu, system_load=os.getloadavg(),
                meminfo=Path("/proc/meminfo").read_text())) + "\n")
            output.flush()
            previous, previous_time = current, sample_start
            if current.get(pid, {}).get("state") == "Z":
                break
            time.sleep(max(0, args.interval - (time.monotonic() - sample_start)))
        code = process.wait() if process else None
        output.write(json.dumps(dict(type="end", unix_seconds=time.time(), returncode=code)) + "\n")
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
