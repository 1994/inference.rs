"""Protected real-model CLI/HTTP acceptance; invoke inside safe-run + hardware-monitor."""

import argparse
import json
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error


def timed_request(url, payload):
    started = time.perf_counter()
    response = request(url, payload)
    return response, (time.perf_counter() - started) * 1000


def check_run(args, config):
    messages = args.output.with_suffix(".messages.json")
    messages.write_text(json.dumps([{"role": "user", "content": "只输出17乘23的结果。"}]))
    encoded = json.loads(
        subprocess.check_output(
            [args.binary, "tokenize", "--package", args.model, "--messages", str(messages)]
        )
    )
    inputs = args.output.with_suffix(".requests.json")
    inputs.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "model": 1,
                    "input": {"Sequence": {"tokens": encoded["tokens"]}},
                    "workload": {"Generate": {"max_new_tokens": 16}},
                    "sampling": {"temperature": 0.0, "eos_token": 248046},
                }
            ]
        )
    )
    command = [
        args.binary,
        "--backend",
        "cuda",
        "run",
        "--package",
        args.model,
        "--config",
        str(config),
        "--gpu-memory-utilization",
        "0.95",
        "--requests",
        str(inputs),
    ]
    result = json.loads(subprocess.check_output(command))
    assert result["results"][0]["output"]["Tokens"] == [18, 24, 16, 248046], result["results"]
    assert result["results"][0]["measurement"]["e2e_us"] > 10000, result["results"]
    args.output.write_text(
        json.dumps(
            {
                "passed": True,
                "command": command,
                "results": result["results"],
                "scope": "native CUDA CLI run actual-model acceptance",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"passed": True, "output": str(args.output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-only", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", default="target/release/infer")
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite acceptance evidence")
    prefix = args.output.with_suffix("")
    config = prefix.with_suffix(".config.json")
    config.write_text(
        json.dumps(
            {
                "max_requests": 4,
                "candidate_limit": 4,
                "max_num_seqs": 2,
                "max_request_units": 4,
                "max_num_batched_tokens": 128,
                "workspace_bytes": 268435456,
                "resource_timeout_us": 30000000,
            }
        )
    )
    if args.cli_only:
        check_run(args, config)
        return
    binary = [
        args.binary,
        "--backend",
        "cuda",
        "--num-speculative-tokens",
        str(args.num_speculative_tokens),
    ]
    doctor = json.loads(subprocess.check_output([*binary, "doctor"]))
    assert doctor["backends"]["cuda"]["available"] is True, doctor
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        *binary,
        "serve",
        "--package",
        args.model,
        "--config",
        str(config),
        "--gpu-memory-utilization",
        "0.95",
        "--listen",
        f"127.0.0.1:{port}",
    ]
    log = prefix.with_suffix(".server.log")
    responses = []
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited: {log.read_text()}")
                try:
                    models = request(base + "/v1/models")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"server readiness timed out: {log.read_text()}"
                        ) from None
                    time.sleep(0.2)
            model_id = models["data"][0]["id"]
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "只输出17乘23的结果。"}],
                "max_tokens": 16,
                "temperature": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "enable_thinking": False,
            }
            wall_ms = []
            for _ in range(4):
                response, elapsed_ms = timed_request(base + "/v1/chat/completions", payload)
                responses.append(response)
                wall_ms.append(elapsed_ms)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(request, base + "/v1/chat/completions", payload) for _ in range(2)
                ]
                responses.extend(f.result() for f in futures)
            for response in responses:
                assert response["choices"][0]["message"]["content"].strip() == "391", response
            # Omitted sampling fields follow the package resolver; only exercise completion here.
            defaults = request(
                base + "/v1/chat/completions",
                {
                    "model": model_id,
                    "messages": payload["messages"],
                    "max_tokens": 1,
                },
            )
            assert defaults["usage"]["completion_tokens"] == 1, defaults
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("server failed to drain on SIGINT") from None
        if process.returncode != 0:
            raise RuntimeError(f"server exit {process.returncode}: {log.read_text()}")
    result = {
        "passed": True,
        "model": args.model,
        "command": command,
        "doctor": doctor,
        "responses": responses,
        "default_sampling_response": defaults,
        "sequential_request_wall_ms": wall_ms,
        "checks": [
            "CUDA CLI selection",
            "HTTP model discovery",
            "greedy override",
            "two concurrent HTTP requests",
            "omitted sampling fields",
            "graceful shutdown",
        ],
        "scope": (
            "functional acceptance; synchronous CUDA provider, "
            "not GPU continuous batching or performance baseline"
        ),
        "mtp_depth": args.num_speculative_tokens,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output), "requests": 7}))


if __name__ == "__main__":
    main()
