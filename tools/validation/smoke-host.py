"""CLI/Agent/HTTP acceptance using saved weights and an explicit backend."""

import argparse
import copy
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, default=Path("target/release/infer"))
parser.add_argument("--output", type=Path, default=Path("artifacts/host-acceptance"))
parser.add_argument("--server-url")
parser.add_argument("--qwen-text-package", type=Path)
parser.add_argument("--backend", choices=["test-cpu", "metal", "auto"], default="metal")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
binary = str(args.binary.resolve())
package = "examples/qwen-hybrid-tiny"
golden = json.loads(Path(package, "golden.json").read_text())
requests = json.loads(Path("examples/requests.json").read_text())
summary = {}


def save(name, value):
    Path(args.output, name + ".json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    return value


def run(name, *command):
    completed = subprocess.run(
        [binary, "--backend", args.backend, *map(str, command)], capture_output=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr.decode()
    return save(name, json.loads(completed.stdout))


inspect = run("package", "inspect-package", "--package", package)
assert inspect["manifest"]["bindings"] and inspect["program"]["nodes"]
verified = run(
    "verify",
    "verify",
    "--package",
    package,
    "--golden",
    Path(package, "golden.json"),
    "--atol",
    "0.000002",
    "--rtol",
    "0.00002",
)
assert verified["passed"] and verified["incremental_tokens_executed"] == 6
grouped = run(
    "verify-grouped",
    "verify",
    "--package",
    "examples/qwen-hybrid-grouped",
    "--golden",
    "examples/qwen-hybrid-grouped/golden.json",
    "--atol",
    "0.000002",
    "--rtol",
    "0.00002",
)
assert grouped["passed"]
journal, snapshot = Path(args.output, "journal.json"), Path(args.output, "snapshot.json")
executed = run(
    "run",
    "run",
    "--package",
    package,
    "--requests",
    "examples/requests.json",
    "--config",
    "examples/runtime.json",
    "--journal",
    journal,
    "--snapshot",
    snapshot,
    "--op-trace",
    Path(args.output, "ops.json"),
)
assert len(executed["results"]) == 4 and all(
    r["measurement"]["successful"] for r in executed["results"]
)
assert executed["execution"]["state"]["sequences"] == 0
assert any("CostFeedback" in a for a in json.loads(journal.read_text()))
captured_run = json.loads(snapshot.read_text())
assert captured_run["schema_version"] == 6
assert any(d["step"] and d["step"]["role"] == "Mixed" for d in captured_run["decisions"])
for name, option, path in [
    ("journal-replay", "--journal", journal),
    ("snapshot-replay", "--snapshot", snapshot),
]:
    restored = run(
        name, "replay", "--package", package, option, path, "--config", "examples/runtime.json"
    )
    if args.backend == "test-cpu":
        assert restored["results"] == executed["results"]
    else:

        def semantic(results):
            return [{k: r[k] for k in ["request", "output", "reason"]} for r in results]

        assert semantic(restored["results"]) == semantic(executed["results"])
run(
    "profile-command",
    "profile",
    "--package",
    package,
    "--journal",
    journal,
    "--config",
    "examples/runtime.json",
    "--output",
    Path(args.output, "profile.json"),
)
profile = json.loads(Path(args.output, "profile.json").read_text())
assert profile["traceEvents"] and all(e["ph"] == "X" for e in profile["traceEvents"])
baseline = run(
    "benchmark",
    "benchmark",
    "--package",
    package,
    "--requests",
    8,
    "--input-tokens",
    6,
    "--output-tokens",
    5,
    "--output",
    Path(args.output, "baseline.json"),
)
assert baseline["successful"] == 8
comparison = run(
    "compare",
    "compare",
    "--baseline",
    Path(args.output, "baseline.json"),
    "--candidate",
    Path(args.output, "baseline.json"),
    "--correctness-passed",
)
assert comparison["accepted"]
encoded = run("tokenize", "tokenize", "--package", package, "--text", "hello world")
assert encoded["tokens"] == [1, 2]
summary["cli"] = {
    "four_workloads": True,
    "journal_snapshot_parity": True,
    "golden_passed": True,
    "execution_profile_events": len(profile["traceEvents"]),
    "backend": args.backend,
    "benchmark_successful": baseline["successful"],
}

agent = subprocess.Popen(
    [binary, "--backend", args.backend, "agent", "--package", package, "--probe-memory-mib", "4"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)
transcript = []


def rpc(method, params=None):
    message = {"jsonrpc": "2.0", "id": len(transcript) + 1, "method": method}
    if params is not None:
        message["params"] = params
    agent.stdin.write(json.dumps(message) + "\n")
    agent.stdin.flush()
    response = json.loads(agent.stdout.readline())
    assert "error" not in response, response
    transcript.append({"request": message, "response": response})
    return response["result"]


try:
    discovery = rpc("agent.discover")
    assert discovery["protocol_version"] == "1.0"
    assert {c["method"] for c in discovery["commands"]} >= {
        "experiment.run",
        "runtime.graph",
        "accuracy.verify",
        "benchmark.run",
        "kernel.profile",
        "observability.inspect",
        "observability.events",
    }
    assert rpc("model.bindings")
    capabilities = rpc("executor.inspect")["capabilities"]
    assert capabilities["backend"]["kind"] in ["test_cpu", "metal"]
    assert "nvidia" not in capabilities
    cache = rpc("kv.inspect")
    if capabilities["backend"]["kind"] == "metal":
        assert cache["pool"]["total_blocks"] > 0
        assert cache["pool"]["active_blocks"] == 0
    assert all(k["source"] for k in rpc("kernel.sources"))
    request = {
        "id": 1,
        "model": 1,
        "input": {"Sequence": {"tokens": golden["prefixes"][-1]["tokens"]}},
        "workload": {"Generate": {"max_new_tokens": 5}},
    }
    rpc("runtime.submit", request)
    rpc("runtime.tick", {"now_us": 0})
    capture_clock = 1
    captured = rpc("snapshot.capture", {"now_us": capture_clock})
    while captured["pending"]:
        time.sleep(0.001)
        capture_clock += 1
        assert capture_clock < 10000
        captured = rpc("snapshot.capture", {"now_us": capture_clock})
    assert not captured["pending"] and captured["snapshot"]["execution_state"]
    assert captured["snapshot"]["schema_version"] == 6
    save("mid-generation-snapshot", captured["snapshot"])
    probes = rpc("executor.probes")
    assert probes and {p["layer"] for p in probes} == {0, 1}
    rpc("snapshot.replay", captured["snapshot"])
    for now in range(capture_clock + 1, capture_clock + 10000):
        if rpc("runtime.inspect")["active_requests"] == 0:
            break
        rpc("runtime.tick", {"now_us": now})
        time.sleep(0.001)
    query = rpc("runtime.query")
    scheduling = rpc("scheduler.inspect")
    assert scheduling["cost_model"]["observations"] > 0
    selected_backend = rpc("runtime.inspect")["backend_kind"]
    assert scheduling["cost_model"]["last_source"] == (
        "MetalGpu" if selected_backend == "metal" else "CpuWall"
    )
    explanation = rpc("scheduler.explain", {"request_id": 1})
    assert explanation["admission"]["rejection"] is None and explanation["last_selection"]
    assert query["requests"][0]["completed"]["output"]["Tokens"] == golden["greedy_tokens"]
    state = rpc("executor.inspect")["state"]
    assert state["sequences"] == state["reserved_bytes"] == state["allocated_bytes"] == 0
    assert rpc("executor.trace")
    assert rpc("profile")["traceEvents"]
    assert rpc(
        "verify.numeric",
        {"reference": [1.0, 2.0], "candidate": [1.0, 2.0], "atol": 0.000002, "rtol": 0.00002},
    )["passed"]
    report = rpc(
        "benchmark.measure", {"requests": requests, "ttft_slo_us": 1000000, "tpot_slo_us": 100000}
    )["report"]
    assert report["successful"] == 4
    assert rpc(
        "experiment.compare",
        {
            "baseline": report,
            "candidate": report,
            "correctness_passed": True,
            "max_p99_regression": 0.05,
        },
    )["accepted"]
    graph = rpc("runtime.graph", {"request_id": 1})
    assert {edge["relation"] for edge in graph["edges"]} >= {
        "scheduled_by",
        "executed_by",
        "contains",
        "lowered_to",
        "implemented_by",
    }
    sources = rpc("kernel.sources")
    kernel_profile = rpc("kernel.profile", {"kernel_id": sources[0]["kernel"]})
    assert kernel_profile["source"]
    assert kernel_profile["timing_scope"] == (
        "cpu_encoding" if selected_backend == "metal" else "cpu_wall"
    )
    before_experiment = rpc("runtime.inspect")
    experiment = rpc(
        "experiment.run",
        {
            "baseline": {},
            "candidate": {"token_budget": 1, "max_batch": 1},
            "workload": {"requests": [request], "ttft_slo_us": 1000000, "tpot_slo_us": 100000},
            "constraints": {
                "accuracy": {
                    "atol": 0.000002,
                    "rtol": 0.00002,
                    "reference_outputs": {"1": {"Tokens": golden["greedy_tokens"]}},
                },
                "max_p99_regression": 100.0,
            },
        },
    )
    assert experiment["baseline"]["successful"] == experiment["candidate"]["successful"] == 1
    assert len(experiment["correctness"]) == 3
    assert all(evidence["passed"] for evidence in experiment["correctness"])
    assert isinstance(experiment["verdict"]["accepted"], bool)
    assert rpc("runtime.inspect") == before_experiment
    assert rpc("experiment.inspect", {"experiment_id": experiment["id"]}) == experiment
    assert rpc("runtime.diagnose")["invariants_passed"]
    save("experiment", experiment)
    observation = rpc("observability.inspect")
    assert observation["metrics"]["successful_requests"] == 1
    assert (
        observation["metrics"]["gpu_execution" if selected_backend == "metal" else "cpu_execution"][
            "count"
        ]
        > 0
    )
    assert observation["resources"]["logical_pages"] == 0
    observed = rpc("observability.events", {"request_id": 1})
    assert observed["events"] and not observed["cursor_gap"]
    assert rpc("observability.timeline")["traceEvents"]
    assert "infer_request_e2e_seconds_bucket" in rpc("observability.metrics")["text"]
    save("observability", observation)
    summary["agent"] = {
        "physical_checkpoint_parity": True,
        "layer_probes": len(probes),
        "quality_methods": True,
        "typed_command_discovery": True,
        "semantic_source_graph": True,
        "isolated_experiment_golden": True,
        "live_metrics_and_event_history": True,
        "calls": len(transcript),
        "calibration": scheduling["cost_model"],
    }
finally:
    agent.stdin.close()
    agent.wait(timeout=10)
save("agent", transcript)

if args.qwen_text_package:
    text_golden = json.loads(Path("examples/qwen3.8-27b/text-golden.json").read_text())
    for index, case in enumerate(text_golden["cases"]):
        messages = Path(args.output, f"messages-{index}.json")
        messages.write_text(json.dumps(case["messages"], ensure_ascii=False))
        command = ["tokenize", "--package", args.qwen_text_package, "--messages", messages]
        if case["enable_thinking"]:
            command.append("--enable-thinking")
        actual = run(f"qwen-text-{index}", *command)
        assert actual["rendered"] == case["rendered"] and actual["tokens"] == case["tokens"]
    summary["qwen_text"] = {"official_cases": 2, "rendered_and_token_parity": True}

if args.server_url:

    def http(path, payload=None, extra_headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            args.server_url + path,
            data=data,
            headers={
                **({"Content-Type": "application/json"} if data else {}),
                **(extra_headers or {}),
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                return response.status, response.read().decode()
        except HTTPError as error:
            return error.code, error.read().decode()

    health = json.loads(http("/health")[1])
    assert health["ready"] and health["weight_backed_dataflow"]

    def generate(id):
        payload = copy.deepcopy(request)
        payload["id"] = id
        status, body = http(
            "/native/v1/requests",
            payload,
            {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
        )
        result = json.loads(body)
        assert status == 200 and result["output"]["Tokens"] == golden["greedy_tokens"]
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        save("http-concurrent", list(pool.map(generate, range(101, 109))))
    stream_request = copy.deepcopy(request)
    stream_request["id"] = 201
    status, body = http("/native/v1/stream", stream_request)
    assert status == 200 and body.count("event: token") == 5 and body.count("event: finished") == 1
    Path(args.output, "stream.txt").write_text(body)
    payload = {
        "id": 301,
        "model": 1,
        "prompt": "hello world system assistant token8 token13",
        "workload": {"Generate": {"max_new_tokens": 5}},
    }
    status, body = http("/native/v1/text", payload)
    plain = save("http-text", json.loads(body))
    assert status == 200 and plain["result"]["output"]["Tokens"] == golden["greedy_tokens"]
    assert plain["text"] == "token25 system system system system"
    assert http("/native/v1/text", payload)[0] == 409
    chat = {
        "id": 302,
        "model": 1,
        "messages": [{"role": "user", "content": "hello world"}],
        "workload": {"Generate": {"max_new_tokens": 5}},
    }
    status, body = http("/native/v1/text", chat)
    assert status == 200 and save("http-chat", json.loads(body))["text"]
    embed = copy.deepcopy(requests[1])
    text_embed = {
        "id": 303,
        "model": 1,
        "prompt": "hello world system user assistant",
        "workload": embed["workload"],
    }
    status, body = http("/native/v1/text", text_embed)
    assert status == 200 and len(json.loads(body)["result"]["output"]["Embedding"]) == 4
    for id, item in [(305, requests[2]), (306, requests[3])]:
        item = copy.deepcopy(item)
        item["id"] = id
        status, body = http("/native/v1/requests", item)
        assert status == 200 and json.loads(body)["measurement"]["successful"]
    malformed = copy.deepcopy(payload)
    malformed["unknown"] = True
    assert http("/native/v1/text", malformed)[0] == 422
    malformed = copy.deepcopy(payload)
    malformed["messages"] = chat["messages"]
    assert http("/native/v1/text", malformed)[0] == 400
    inspection = save("http-inspection", json.loads(http("/native/v1/runtime")[1]))
    assert inspection["active_requests"] == inspection["completed_requests"] == 0
    assert inspection["inflight_step"] is None and inspection["state"]["allocated_pages"] == 0
    observation = save("http-observability", json.loads(http("/native/v1/observability")[1]))
    assert observation["metrics"]["successful_requests"] == 14
    assert (
        observation["metrics"][
            "gpu_execution" if health["backend_kind"] == "metal" else "cpu_execution"
        ]["count"]
        > 0
    )
    assert observation["trace_coverage"]["complete_requests"] == 14
    assert observation["resources"]["active_kv_blocks"] == 0
    metrics = http("/metrics")[1]
    assert (
        "infer_event_ring_dropped_total" in metrics
        and "infer_request_ttft_seconds_bucket" in metrics
    )
    Path(args.output, "metrics.prom").write_text(metrics)
    events = save("http-events", json.loads(http("/native/v1/events?request=101&limit=4096")[1]))
    assert any(event["event"]["kind"] == "Finished" for event in events["events"])
    otlp = save("http-otlp", json.loads(http("/native/v1/traces/otlp")[1]))
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert sum(span["traceId"] == "4bf92f3577b34da6a3ce929d0e0e4736" for span in spans) == 8
    assert any(
        d["code"] == "admission_rejected" for d in json.loads(http("/native/v1/diagnostics")[1])
    )
    summary["http"] = {
        "concurrent": 8,
        "sse_tokens": 5,
        "text_chat_embedding_rank_decision": True,
        "schema_and_duplicate_errors": True,
        "logical_state_leaks": 0,
        "prometheus_w3c_otlp_events": True,
    }
save("summary", summary)
print(json.dumps(summary, ensure_ascii=False))
