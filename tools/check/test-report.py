"""Run a test entry and report what it collected, ran, ignored and failed.

The baseline plan asks every test entry to report those four numbers, and a job that
only cross-compiles to say `compiled` rather than imply it ran anything. `cargo test`
prints them per binary, which is easy to miss in a long log and impossible to total by
eye; this wrapper streams the command unchanged, totals the lines, and can write the
summary as JSON for the job to keep.
"""

import argparse
import json
import re
import subprocess
import sys

RESULT = re.compile(
    r"^test result: (?P<result>\w+)\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored"
)
RUNNING = re.compile(r"^running (?P<collected>\d+) tests?$")


def summarize(entry, lines):
    """Totals for one entry, plus whether it ran anything at all."""
    binaries = 0
    collected = 0
    passed = 0
    failed = 0
    ignored = 0
    results = []
    for line in lines:
        running = RUNNING.match(line.strip())
        if running is not None:
            binaries += 1
            collected += int(running.group("collected"))
            continue
        result = RESULT.match(line.strip())
        if result is not None:
            passed += int(result.group("passed"))
            failed += int(result.group("failed"))
            ignored += int(result.group("ignored"))
            results.append(result.group("result"))
    ran = binaries > 0
    return {
        "entry": entry,
        "state": "ran" if ran else "compiled",
        "binaries": binaries,
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "result": "failed" if (failed or any(r != "ok" for r in results)) else "ok",
    }


def line(summary):
    return (
        f"{summary['entry']}: {summary['state']}, {summary['binaries']} binaries, "
        f"{summary['collected']} collected, {summary['passed']} passed, "
        f"{summary['failed']} failed, {summary['ignored']} ignored, {summary['result']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    # Only the separator this wrapper documents is removed; an inner one belongs to the command
    # (`cargo test ... -- --ignored`, for instance) and has to reach it unchanged.
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("no command given")

    captured = []
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for text in process.stdout:
        sys.stdout.write(text)
        captured.append(text)
    process.wait()

    summary = summarize(arguments.entry, captured)
    print(line(summary), flush=True)
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(summary, handle, indent=1)
    # A command that ran nothing is only acceptable when it was asked to compile.
    if process.returncode != 0 or summary["failed"] or summary["result"] != "ok":
        return process.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
