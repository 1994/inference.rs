#!/usr/bin/env python3
"""Run one engine over a prepared serving workload and record a gate-ready report.

This promotes the local qualification experiment into a reusable harness. It
accepts the engine, the model package and the executable as arguments, feeds
every engine the identical token arrays from a workload file produced by
``serve-workloads.py``, and writes a report that ``compare-results.py`` gates.

The harness starts exactly one server per invocation, measures the case matrix
with one excluded warmup plus measured repeats, and refuses to report success
when a reference source silently substituted a weaker configuration: a server
that exits, a stream that ends early, a warm request without observed prefix
reuse, or a native measurement reported as unsuccessful all fail the run.

Examples:
    python3 tools/bench/serve-compare.py --engine native \
        --model /home/r/models/Qwen3.8-27B-NVFP4 \
        --executable target/x86_64-unknown-linux-gnu/release/infer \
        --inputs artifacts/workloads/27b-inputs.json \
        --mtp 2 --output-dir artifacts/perf
    python3 tools/bench/serve-compare.py --engine vllm \
        --model /home/r/models/Qwen3.8-27B-NVFP4 \
        --executable artifacts/vllm-compare/bin/python \
        --inputs artifacts/workloads/27b-inputs.json \
        --mtp 2 --output-dir artifacts/perf --gpu-memory-utilization 0.85
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from functools import partial
from pathlib import Path

CASES = ["short", "long", "batch4", "hot_long"]
NATIVE_PREFIX_METRICS = ("prefix_tokens_reused",)
REFERENCE_PREFIX_METRICS = ("vllm:prefix_cache_hits_total", "sglang:prefix_cache_hits_total")


def http(url, payload=None, timeout=600):
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def eos_tokens(model):
    configuration = json.loads((Path(model) / "generation_config.json").read_text())
    value = configuration["eos_token_id"]
    return value if isinstance(value, list) else [value]


def native_command(args, port):
    command = [
        str(args.executable),
        str(args.model),
        "--listen",
        f"127.0.0.1:{port}",
        "--num-speculative-tokens",
        str(args.mtp),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.extra:
        command += args.extra
    return command


def vllm_command(args, port):
    command = [
        str(args.executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--generation-config",
        "vllm",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
    ]
    if args.prefix_cache:
        command.append("--enable-prefix-caching")
    if args.mtp:
        command += [
            "--speculative-config",
            json.dumps({"method": "mtp", "num_speculative_tokens": args.mtp}),
        ]
    if args.extra:
        command += args.extra
    return command


def sglang_command(args, port):
    command = [
        str(args.executable),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--context-length",
        str(args.max_model_len),
        "--max-running-requests",
        str(args.max_num_seqs),
        "--mem-fraction-static",
        str(args.gpu_memory_utilization),
    ]
    if args.mtp:
        command += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(args.mtp)]
    if args.extra:
        command += args.extra
    return command


def read_metrics(args, base):
    route = "/native/v1/observability" if args.engine == "native" else "/metrics"
    with http(base + route) as response:
        raw = response.read().decode()
    if args.engine == "native":
        return json.loads(raw)
    result = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"([^ {]+)(?:\{[^}]*\})?\s+([^ ]+)", line)
        if match and ("prefix_cache" in match[1] or "spec_decode" in match[1]):
            result[match[1]] = result.get(match[1], 0.0) + float(match[2])
    return result


def prefix_reuse(args, trial):
    names = NATIVE_PREFIX_METRICS if args.engine == "native" else REFERENCE_PREFIX_METRICS
    total = 0.0
    for name in names:
        before = trial["metrics_before"]
        after = trial["metrics_after"]
        if args.engine == "native":
            left = before.get("metrics", {}).get(name, 0.0)
            right = after.get("metrics", {}).get(name, 0.0)
        else:
            left = before.get(name, 0.0)
            right = after.get(name, 0.0)
        total += right - left
    return total


def stream_request(args, base, model, eos, counter, row):
    if args.engine == "native":
        payload = {
            "id": next(counter),
            "model": 1,
            "input": {"Sequence": {"tokens": row["tokens"]}},
            "workload": {"Generate": {"max_new_tokens": args.tokens}},
            "sampling": {"temperature": 0.0, "eos_tokens": eos},
        }
        route = "/native/v1/stream"
    else:
        payload = {
            "model": str(model),
            "prompt": row["tokens"],
            "max_tokens": args.tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "seed": 0,
            "stop_token_ids": eos,
            "stream": True,
            "return_token_ids": True,
        }
        route = "/v1/completions"
    started = time.perf_counter()
    tokens = []
    arrivals = []
    finished = None
    try:
        with http(base + route, payload) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                item = json.loads(data)
                if args.engine == "native":
                    if "Token" in item:
                        new = [item["Token"]["token"]]
                    elif "Finished" in item:
                        finished = item["Finished"]
                        new = []
                    else:
                        raise RuntimeError(f"unexpected native event: {item}")
                else:
                    if "error" in item:
                        raise RuntimeError(str(item))
                    choices = item.get("choices", [])
                    new = (choices[0].get("token_ids") or []) if choices else []
                    if choices and choices[0].get("finish_reason"):
                        finished = choices[0]["finish_reason"]
                if new:
                    tokens.extend(new)
                    arrivals.extend([time.perf_counter() - started] * len(new))
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
    elapsed = time.perf_counter() - started
    if not tokens or finished is None:
        raise RuntimeError(f"incomplete stream: {tokens}, {finished}")
    if args.engine == "native" and not finished["measurement"]["successful"]:
        raise RuntimeError(f"unsuccessful native measurement: {finished}")
    visible = [(token, at) for token, at in zip(tokens, arrivals, strict=True) if token not in eos]
    return {
        "case": row["case"],
        "repeat": row["repeat"],
        "slot": row["slot"],
        "input_tokens": len(row["tokens"]),
        "token_ids": tokens,
        "output_tokens_excluding_eos": len(visible),
        "wall_seconds": elapsed,
        "ttft_seconds": arrivals[0],
        "tpot_seconds": (
            (visible[-1][1] - visible[0][1]) / (len(visible) - 1) if len(visible) > 1 else None
        ),
        "arrival_seconds": arrivals,
        "finished": finished,
    }


def build_command(args, port):
    if args.engine == "native":
        return native_command(args, port)
    if args.engine == "vllm":
        return vllm_command(args, port)
    return sglang_command(args, port)


def wait_ready(args, process, base, log_path, timeout):
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            with http(base + "/health", timeout=2) as response:
                health = response.read()
                if args.engine != "native" or json.loads(health)["ready"]:
                    return time.perf_counter() - started
        except (OSError, ValueError):
            pass
        time.sleep(0.3)
    raise RuntimeError(f"startup timeout after {timeout}s; see {log_path}")


def measure(args, report, base, rows, save):
    counter = iter(range(1, 1_000_000))
    eos = report["eos_tokens"]
    for case in CASES:
        if not any(r["case"] == case for r in rows):
            continue
        for repeat in range(-1, args.repeats):
            group = [r for r in rows if r["case"] == case and r["repeat"] == repeat]
            before = read_metrics(args, base)
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as pool:
                results = list(
                    pool.map(partial(stream_request, args, base, args.model, eos, counter), group)
                )
            trial = {
                "case": case,
                "repeat": repeat,
                "warmup": repeat < 0,
                "concurrency": len(group),
                "start_unix": time.time(),
                "wall_seconds": time.perf_counter() - started,
                "results": results,
            }
            time.sleep(1.1)
            trial["metrics_before"] = before
            trial["metrics_after"] = read_metrics(args, base)
            report["trials"].append(trial)
            save()
            outputs = [r["output_tokens_excluding_eos"] for r in results]
            print(f"{case} {repeat} {trial['wall_seconds']:.3f}s {outputs}", flush=True)


def load_workload(path, repeats, max_input_tokens=0):
    """Read and validate a workload file, returning its rows.

    A partially filtered file must not silently measure an empty concurrency
    group, so every case that appears must carry its excluded warmup plus all
    measured repeats.
    """
    rows = json.loads(Path(path).read_text())
    for row in rows:
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise SystemExit(f"workload row has no tokens: {row.get('case')}/{row.get('slot')}")
        if max_input_tokens and len(tokens) > max_input_tokens:
            raise SystemExit(
                f"input of {len(tokens)} tokens exceeds --max-input-tokens {max_input_tokens}"
            )
    missing = [
        f"{case}:{repeat}"
        for case in CASES
        if any(r["case"] == case for r in rows)
        for repeat in range(-1, repeats)
        if not any(r["case"] == case and r["repeat"] == repeat for r in rows)
    ]
    if missing:
        raise SystemExit(f"workload is missing case/repeat rows: {', '.join(missing)}")
    return rows


def run(args):
    rows = load_workload(args.inputs, args.repeats, args.max_input_tokens)
    eos = eos_tokens(args.model)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    command = build_command(args, port)
    stem = args.output_dir / f"{args.model.name}-mtp{args.mtp}-{args.engine}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"{stem}.server.log")
    report = {
        "engine": args.engine,
        "engine_version": args.engine_version,
        "model": str(args.model),
        "model_package_sha256": args.package_sha256,
        "command": command,
        "max_new_tokens": args.tokens,
        "temperature": 0,
        "eos_tokens": eos,
        "mtp_depth": args.mtp,
        "prefix_cache_enabled": True,
        "binary_sha256": sha256_file(args.executable) if args.engine == "native" else None,
        "workload_file": str(args.inputs),
        "inputs_sha256": sha256_bytes(json.dumps(rows, sort_keys=True).encode()),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "trials": [],
        "completed": False,
    }

    def save():
        Path(f"{stem}.json").write_text(json.dumps(report, indent=2))

    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            report["startup_seconds"] = wait_ready(
                args, process, base, log_path, args.startup_timeout
            )
            print(f"ready after {report['startup_seconds']:.1f}s", flush=True)
            measure(args, report, base, rows, save)
            hot = [t for t in report["trials"] if t["case"] == "hot_long" and not t["warmup"]]
            reused = sum(prefix_reuse(args, t) for t in hot)
            report["hot_prefix_tokens_reused"] = reused
            if hot and reused <= 0:
                raise RuntimeError("prefix cache enabled but no actual hot-prefix reuse observed")
            report["completed"] = True
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            save()
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print(f"report {stem}.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["native", "vllm", "sglang"])
    parser.add_argument("--model", required=True, type=Path, help="model package directory")
    parser.add_argument(
        "--executable", required=True, type=Path, help="engine executable or interpreter"
    )
    parser.add_argument(
        "--inputs", required=True, type=Path, help="workload JSON from serve-workloads.py"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=64, help="generated tokens per request")
    parser.add_argument("--mtp", type=int, default=0, help="speculative/MTP depth")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-input-tokens", type=int, default=0, help="refuse longer inputs when non-zero"
    )
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--engine-version", default=None)
    parser.add_argument("--package-sha256", default=None)
    parser.add_argument("--prefix-cache", action="store_true", default=True)
    parser.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra engine flags")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("the serving gate requires at least three measured repeats")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be within (0, 1)")
    try:
        run(args)
    except Exception as error:  # top-level reporting boundary
        print(f"benchmark failed: {error!r}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
