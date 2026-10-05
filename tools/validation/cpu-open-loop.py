"""Bounded open-loop HTTP load with offered-load latency and independent Metal golden checks."""

import argparse
import hashlib
import json
import math
import platform
import resource
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def http(url, path, payload=None):
    request = Request(
        url + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def percentiles(values):
    ordered = sorted(values)
    return (
        {
            f"p{percentile}_us": round(
                ordered[math.ceil(len(ordered) * percentile / 100) - 1] * 1e6
            )
            for percentile in (50, 95, 99)
        }
        if ordered
        else {}
    )


def load(url, golden, rate, count, first_id):
    permits = threading.BoundedSemaphore(64)
    rows, controls, futures, monitor_errors = [], [], [], []
    stop = threading.Event()
    tokens = golden["prefixes"][-1]["tokens"]
    start = time.perf_counter()

    def request(index, offered, dispatched):
        try:
            status, body = http(
                url,
                "/native/v1/requests",
                {
                    "id": first_id + index,
                    "model": 1,
                    "input": {"Sequence": {"tokens": tokens}},
                    "workload": {"Generate": {"max_new_tokens": 5}},
                },
            )
            if status == 200:
                assert body["output"]["Tokens"] == golden["greedy_tokens"], body
            return {
                "status": status,
                "latency": time.perf_counter() - offered,
                "dispatch_delay": dispatched - offered,
            }
        finally:
            permits.release()

    def monitor():
        try:
            while not stop.is_set():
                before = time.perf_counter()
                status, _ = http(url, "/native/v1/runtime")
                assert status == 200
                controls.append(time.perf_counter() - before)
                stop.wait(0.01)
        except Exception as error:
            monitor_errors.append(error)

    watcher = threading.Thread(target=monitor)
    watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=64) as pool:
            for index in range(count):
                offered = start + index / rate
                delay = offered - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                dispatched = time.perf_counter()
                if permits.acquire(blocking=False):
                    futures.append(pool.submit(request, index, offered, dispatched))
                else:
                    rows.append(
                        {"status": "client_capacity", "dispatch_delay": dispatched - offered}
                    )
            rows.extend(future.result(timeout=20) for future in futures)
    finally:
        stop.set()
        watcher.join(timeout=20)
    if watcher.is_alive() or monitor_errors:
        raise RuntimeError(f"Control monitor failed: {monitor_errors}")
    assert controls, "No control latency samples collected"
    elapsed = time.perf_counter() - start
    deadline = time.monotonic() + 5
    while True:
        _, final = http(url, "/native/v1/runtime")
        if not final["active_requests"] and not final["resource_release_pending"]:
            break
        if time.monotonic() >= deadline:
            raise AssertionError({"undrained_runtime": final})
        time.sleep(0.001)
    assert final["state"]["allocated_pages"] == 0
    assert final["kv_cache"]["active_blocks"] == 0
    statuses = {}
    for row in rows:
        key = str(row["status"])
        statuses[key] = statuses.get(key, 0) + 1
    successful = statuses.get("200", 0)
    assert sum(statuses.values()) == count
    assert set(statuses) <= {"200", "429", "client_capacity"}, statuses
    return {
        "offered_rps": rate,
        "offered": count,
        "max_client_inflight": 64,
        "statuses": statuses,
        "elapsed_s": elapsed,
        "successful_rps": successful / elapsed,
        "offered_e2e": percentiles([row["latency"] for row in rows if row["status"] == 200]),
        "dispatch_delay": percentiles([row["dispatch_delay"] for row in rows]),
        "control_latency": percentiles(controls),
        "final_runtime": final,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/infer")
    parser.add_argument("--rates", nargs="+", type=int, default=[50, 200, 1000])
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/cpu-completion/open-loop.json"
    )
    args = parser.parse_args()
    if min(args.rates) <= 0 or not 1 <= args.requests <= 10000:
        parser.error("positive rates and 1..10000 requests required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    golden = json.loads((ROOT / "examples/qwen-hybrid-tiny/golden.json").read_text())
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with args.output.with_suffix(".server.log").open("w") as log:
        server = subprocess.Popen(
            [
                str(args.binary.resolve()),
                "--backend",
                "metal",
                "serve",
                "--package",
                "examples/qwen-hybrid-tiny",
                "--listen",
                address,
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    if http(f"http://{address}", "/health")[1]["ready"]:
                        break
                except OSError:
                    pass
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Metal service failed to become ready")
                time.sleep(0.05)
            results = [
                load(f"http://{address}", golden, rate, args.requests, 10000 + i * 10000)
                for i, rate in enumerate(args.rates)
            ]
            _, observation = http(f"http://{address}", "/native/v1/observability")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    report = {
        "backend": "metal",
        "model": "qwen-hybrid-tiny",
        "golden_parity": True,
        "scope": "Local tiny-model CPU pipeline; not target-model production throughput",
        "environment": {"os": platform.platform(), "architecture": platform.machine()},
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "server_cpu_s": cpu,
        "server_average_cpu_cores": cpu / (time.perf_counter() - started),
        "server_max_rss_platform_units": after.ru_maxrss,
        "server_max_rss_bytes": after.ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "loads": results,
        "observation": observation,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "golden_parity": True,
                "loads": [
                    {key: value for key, value in result.items() if key != "final_runtime"}
                    for result in results
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
