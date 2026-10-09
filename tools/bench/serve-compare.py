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
import tomllib
import urllib.error
import urllib.request
from functools import partial
from pathlib import Path

CASES = ["short", "long", "batch4", "hot_long"]
NATIVE_PREFIX_METRICS = ("prefix_tokens_reused",)
REFERENCE_PREFIX_METRICS = ("vllm:prefix_cache_hits_total", "sglang:prefix_cache_hits_total")
REPO_ROOT = Path(__file__).resolve().parents[2]
# Model artifacts whose content decides the input identity. Weight shards are recorded by name
# and size instead of hashed, because a 27B package is tens of gigabytes.
MODEL_ARTIFACTS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "chat_template.jinja",
)
# Section-name prefixes that only exist in a debug or explicitly unstripped image. Scanning the
# bytes for these names is not enough: std's backtrace symbolizer mentions them in read-only data
# even when the section table has none, so the section table itself is read.
DEBUG_PREFIXES = ("__debug_", ".debug_")


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


def load_checklist_module():
    """Load the checklist helper, which is a hyphenated script and not importable by name."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "experiment_checklist", Path(__file__).with_name("experiment-checklist.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def eos_tokens(model):
    configuration = json.loads((Path(model) / "generation_config.json").read_text())
    value = configuration["eos_token_id"]
    return value if isinstance(value, list) else [value]


def git_identity():
    """Full source revision and whether the measured tree is dirty."""
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def release_profile():
    """The workspace release profile, so the recorded build facts are the real ones."""
    try:
        manifest = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return manifest.get("profile", {}).get("release", {})


def elf_section_names(data):
    """Section names of an ELF image, or None when the bytes are not ELF."""
    if data[:4] != b"\x7fELF":
        return None
    is64, little = data[4] == 2, data[5] == 1
    order = "little" if little else "big"

    def read(offset, size):
        return int.from_bytes(data[offset : offset + size], order)

    if is64:
        table_offset, entry_size, count, strings = (
            read(0x28, 8),
            read(0x3A, 2),
            read(0x3C, 2),
            read(0x3E, 2),
        )
        name_at, offset_at, size_at = 0, 0x18, 0x20
        wide = 8
    else:
        table_offset, entry_size, count, strings = (
            read(0x20, 4),
            read(0x2E, 2),
            read(0x30, 2),
            read(0x32, 2),
        )
        name_at, offset_at, size_at = 0, 0x10, 0x14
        wide = 4
    if not count or strings >= count:
        return []

    def entry(index):
        base = table_offset + index * entry_size
        return read(base + name_at, 4), read(base + offset_at, wide), read(base + size_at, wide)

    _, strings_offset, strings_size = entry(strings)
    names = []
    for index in range(count):
        start, _, _ = entry(index)
        section = data[strings_offset : strings_offset + strings_size]
        end = section.find(b"\0", start)
        names.append(section[start : end if end != -1 else len(section)].decode("ascii", "replace"))
    return names


def macho_section_names(data):
    """Section names of a Mach-O image, or None when the bytes are not Mach-O."""
    magic = int.from_bytes(data[:4], "little")
    if magic not in (0xFEEDFACF, 0xFEEDFACE):
        return None
    is64 = magic == 0xFEEDFACF
    commands = int.from_bytes(data[0x10:0x14], "little")
    offset = 0x20 if is64 else 0x1C
    names = []
    for _ in range(commands):
        command = int.from_bytes(data[offset : offset + 4], "little")
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if command == 0x19 and is64:
            count, section, stride = (
                int.from_bytes(data[offset + 64 : offset + 68], "little"),
                offset + 72,
                80,
            )
        elif command == 0x1 and not is64:
            count, section, stride = (
                int.from_bytes(data[offset + 48 : offset + 52], "little"),
                offset + 56,
                68,
            )
        else:
            count = 0
        for _ in range(count):
            names.append(data[section : section + 16].rstrip(b"\0").decode("ascii", "replace"))
            section += stride
        offset += size
    return names


def binary_identity(path):
    """Observable build facts for an engine executable.

    A debug or explicitly unstripped build carries debug sections in its section table, so the
    table is read rather than searched for names. An image whose format cannot be read is
    reported as unverifiable, and the gate refuses to call that a release build.
    """
    data = Path(path).read_bytes()
    names = elf_section_names(data)
    image = "elf"
    if names is None:
        names = macho_section_names(data)
        image = "macho"
    if names is None:
        return {
            "path": str(path),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "image": "unknown",
            "debug_sections": [],
            "release_like": None,
        }
    debug_sections = sorted(name for name in names if name.startswith(DEBUG_PREFIXES))
    return {
        "path": str(path),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "image": image,
        "debug_sections": debug_sections,
        "release_like": not debug_sections,
    }


def hardware_identity():
    """Device identity and driver, so a different GPU cannot be paired silently."""
    fields = ["uuid", "name", "driver_version"]
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"devices": []}
    return {
        "devices": [
            dict(zip(fields, [part.strip() for part in line.split(",")], strict=True))
            for line in raw.splitlines()
            if line.strip()
        ]
    }


def model_identity(model):
    """Fingerprint the artifacts that decide which model is really being measured."""
    files = {}
    for name in MODEL_ARTIFACTS:
        path = Path(model) / name
        if path.exists():
            files[name] = sha256_file(path)
    shards = sorted(Path(model).glob("*.safetensors"))
    return {
        "path": str(model),
        "files": files,
        "shards": {path.name: path.stat().st_size for path in shards},
    }


def runtime_readback(args, base):
    """Read the configuration the native server actually applied."""
    if args.engine != "native":
        return {"source": "command line", "readback": None}
    try:
        with http(base + "/native/v1/runtime", timeout=30) as response:
            inspection = json.loads(response.read())
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        return {"source": "unavailable", "error": repr(error)}
    return {
        "source": "native runtime inspection",
        "readback": {
            key: inspection.get(key)
            for key in ("model_name", "lengths", "scheduler", "execution_profile", "kv_cache")
        },
    }


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
        # The reference engine is given these; leaving native on its own defaults would compare
        # two different service constraints. The per-step token budget is deliberately not forced
        # here: vLLM treats it as a cap while native derives its own chunk and only floors it, so
        # both sides record their effective value instead of pretending one flag means one thing.
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
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
    command.append("--enable-prefix-caching" if args.prefix_cache else "--no-enable-prefix-caching")
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


def _seconds(microseconds):
    """Convert an optional microsecond reading to seconds."""
    return None if microseconds is None else microseconds / 1_000_000


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
    events = []
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
                    at = time.perf_counter() - started
                    tokens.extend(new)
                    arrivals.extend([at] * len(new))
                    events.append({"at_seconds": at, "tokens": len(new)})
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
    elapsed = time.perf_counter() - started
    if not tokens or finished is None:
        raise RuntimeError(f"incomplete stream: {tokens}, {finished}")
    if args.engine == "native" and not finished["measurement"]["successful"]:
        raise RuntimeError(f"unsuccessful native measurement: {finished}")
    visible = [(token, at) for token, at in zip(tokens, arrivals, strict=True) if token not in eos]
    # A single visible token has no inter-token interval, so TPOT is unavailable rather than
    # invalid; the gate keeps the request and reports the metric as not measurable.
    tpot = (visible[-1][1] - visible[0][1]) / (len(visible) - 1) if len(visible) > 1 else None
    finish_reason = finished.get("reason") if isinstance(finished, dict) else finished
    # The server's own view of the same request, recorded next to the client's so client latency
    # and engine phases are never conflated.
    measurement = (finished or {}).get("measurement") if isinstance(finished, dict) else None
    server = (
        {
            "ttft_seconds": _seconds(measurement.get("ttft_us")),
            "e2e_seconds": _seconds(measurement.get("e2e_us")),
            "max_tpot_seconds": _seconds(measurement.get("max_tpot_us")),
            "output_tokens": measurement.get("output_tokens"),
        }
        if measurement
        else None
    )
    return {
        "case": row["case"],
        "repeat": row["repeat"],
        "slot": row["slot"],
        "input_tokens": len(row["tokens"]),
        "token_ids": tokens,
        "output_tokens_excluding_eos": len(visible),
        # Three distinct endpoints, named separately so they cannot be mixed up: the request
        # wall, the last visible token, and (per trial) the group makespan.
        "wall_seconds": elapsed,
        "request_wall_seconds": elapsed,
        "time_to_last_token_seconds": visible[-1][1] if visible else None,
        "ttft_seconds": arrivals[0],
        "tpot_seconds": tpot,
        "tpot_available": tpot is not None,
        "arrival_seconds": arrivals,
        "stream_events": events,
        "finish_reason": finish_reason,
        "truncated": finish_reason is None,
        "server_measurement": server,
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
        declared = [r for r in rows if r["case"] == case]
        if not declared:
            continue
        for repeat in range(-1, args.repeats):
            group = [r for r in declared if r["repeat"] == repeat]
            # The workload declares how many requests a case runs together; measuring fewer
            # would silently turn batch4 into four single-request runs.
            expected = declared[0]["concurrency"]
            if len(group) != expected or len({r["slot"] for r in group}) != len(group):
                raise RuntimeError(
                    f"{case} repeat {repeat} has {len(group)} requests, expected {expected}"
                )
            before = read_metrics(args, base)
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as pool:
                results = list(
                    pool.map(partial(stream_request, args, base, args.model, eos, counter), group)
                )
            makespan = time.perf_counter() - started
            trial = {
                "case": case,
                "repeat": repeat,
                "warmup": repeat < 0,
                "concurrency": len(group),
                "declared_concurrency": expected,
                "start_unix": time.time(),
                "wall_seconds": makespan,
                "group_makespan_seconds": makespan,
                "results": results,
            }
            time.sleep(1.1)
            trial["metrics_before"] = before
            trial["metrics_after"] = read_metrics(args, base)
            # Reuse is recorded per trial, so one hit cannot stand in for the whole hot case.
            trial["prefix_tokens_reused"] = prefix_reuse(args, trial)
            report["trials"].append(trial)
            save()
            outputs = [r["output_tokens_excluding_eos"] for r in results]
            reused = trial["prefix_tokens_reused"]
            print(f"{case} {repeat} {makespan:.3f}s reuse={reused:.0f} {outputs}", flush=True)


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
        if not isinstance(row.get("concurrency"), int) or row["concurrency"] < 1:
            raise SystemExit(
                "workload row declares no positive concurrency: "
                f"{row.get('case')}/{row.get('slot')}"
            )
        if max_input_tokens and len(tokens) > max_input_tokens:
            raise SystemExit(
                f"input of {len(tokens)} tokens exceeds --max-input-tokens {max_input_tokens}"
            )
    for case in CASES:
        declared = {r["concurrency"] for r in rows if r["case"] == case}
        if not declared:
            continue
        if len(declared) > 1:
            raise SystemExit(f"{case} declares inconsistent concurrency: {sorted(declared)}")
        expected = declared.pop()
        for repeat in range(-1, repeats):
            count = sum(1 for r in rows if r["case"] == case and r["repeat"] == repeat)
            if count != expected:
                raise SystemExit(
                    f"{case} repeat {repeat} carries {count} slots, expected {expected}"
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


def workload_matrix(rows, repeats):
    """The case/slot/repeat matrix the workload declares, so the gate can require all of it."""
    return {
        case: {
            "concurrency": next(r["concurrency"] for r in rows if r["case"] == case),
            "repeats": repeats,
        }
        for case in CASES
        if any(r["case"] == case for r in rows)
    }


def require_cache_switch(args):
    """Refuse a requested cache state the engine cannot actually apply."""
    if args.engine == "native" and not args.prefix_cache:
        raise RuntimeError(
            "the native engine cannot disable its prefix cache, so --no-prefix-cache cannot take "
            "effect; a cold profile is unavailable for this engine version"
        )


def run(args):
    require_cache_switch(args)
    rows = load_workload(args.inputs, args.repeats, args.max_input_tokens)
    eos = eos_tokens(args.model)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    command = build_command(args, port)
    name = f"{args.model.name}-mtp{args.mtp}-{args.engine}"
    run_id = args.run_id or None
    stem = args.output_dir / (name if run_id is None else f"{name}-{run_id}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"{stem}.server.log")
    report_path = Path(f"{stem}.json")
    # Evidence must not be replaced by a later run of the same name.
    if report_path.exists():
        raise RuntimeError(
            f"report {report_path} already exists; pass a unique --run-id or a fresh --output-dir"
        )
    build = (
        binary_identity(args.executable)
        if args.engine == "native"
        else {
            "path": str(args.executable),
            "sha256": sha256_file(args.executable),
            "release_like": None,
        }
    )
    profile = release_profile()
    identity = {
        "run_id": run_id,
        "profile_id": args.profile_id,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "source": git_identity(),
        "release_profile": profile,
        "build": build,
        "model": model_identity(args.model),
        "hand_reported_package_sha256": args.package_sha256,
        "hardware": hardware_identity(),
        "limits": (
            {
                "source": "declared on the command line, verified after startup",
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            }
            if args.engine != "native"
            else {
                "source": "pending native runtime inspection",
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
            }
        ),
        "cache": {
            "requested": bool(args.prefix_cache),
            "effective": bool(args.prefix_cache),
        },
    }
    # A release identity is verifiable for native through the binary's own sections; a debug or
    # otherwise unverifiable build must not be recorded as if it were a release.
    identity["verified"] = args.engine != "native" or build["release_like"] is True
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
        "prefix_cache_enabled": bool(args.prefix_cache),
        "binary_sha256": build.get("sha256"),
        "workload_file": str(args.inputs),
        "inputs_sha256": sha256_bytes(json.dumps(rows, sort_keys=True).encode()),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "matrix": workload_matrix(rows, args.repeats),
        "identity": identity,
        "trials": [],
        "completed": False,
    }

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            report["startup_seconds"] = wait_ready(
                args, process, base, log_path, args.startup_timeout
            )
            print(f"ready after {report['startup_seconds']:.1f}s", flush=True)
            report["config_readback"] = runtime_readback(args, base)
            if args.engine == "native":
                readback = (report["config_readback"] or {}).get("readback") or {}
                lengths = readback.get("lengths") or {}
                # The CLI values are not the effective configuration; the readback is.
                report["identity"]["limits"] = {
                    "source": "native runtime inspection",
                    "effective_max_model_len": lengths.get("total"),
                    "effective_input_cap": lengths.get("input_cap"),
                    "effective_output_cap": lengths.get("output_cap"),
                    "effective_prefill_width": (readback.get("execution_profile") or {}).get(
                        "prefill_width"
                    ),
                    "declared": report["identity"]["limits"],
                }
                if lengths.get("total") != args.max_model_len:
                    raise RuntimeError(
                        "native effective context "
                        f"{lengths.get('total')} differs from the requested {args.max_model_len}"
                    )
            if args.checklist is not None:
                # Declared conditions are checked against what the run actually did, including
                # the effective limits just read back, not against the command line.
                module = load_checklist_module()
                checklist = module.load(args.checklist)
                mismatches = module.check_run(checklist, report)
                report["checklist"] = {
                    "path": str(args.checklist),
                    "sha256": sha256_file(args.checklist),
                    "profile_id": checklist["profile_id"],
                    "compliant": not mismatches,
                    "mismatches": mismatches,
                }
                save()
                if mismatches:
                    raise RuntimeError(
                        "run does not match the experiment checklist: " + "; ".join(mismatches)
                    )
            measure(args, report, base, rows, save)
            hot = [t for t in report["trials"] if t["case"] == "hot_long" and not t["warmup"]]
            reused = sum(t["prefix_tokens_reused"] for t in hot)
            report["hot_prefix_tokens_reused"] = reused
            if hot and reused <= 0:
                raise RuntimeError("prefix cache enabled but no actual hot-prefix reuse observed")
            # Every measured repeat must show reuse; one hit cannot speak for the whole case.
            cold = [t["repeat"] for t in hot if t["prefix_tokens_reused"] <= 0]
            if cold:
                raise RuntimeError(
                    f"hot_long repeats {cold} reused no prefix even though the cache is enabled"
                )
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
    parser.add_argument(
        "--run-id", default=None, help="unique evidence id; reports never overwrite"
    )
    parser.add_argument("--profile-id", default=None, help="experiment profile this run belongs to")
    parser.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="experiment profile this run must match; a mismatch fails the run",
    )
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
