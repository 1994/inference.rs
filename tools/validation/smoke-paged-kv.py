"""Release acceptance for a bounded real Metal KV pool, with Host output parity."""

import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, default=Path("target/release/infer"))
parser.add_argument(
    "--test-binary",
    type=Path,
    default=Path("target/debug/infer"),
    help="CPU oracle built with test-backends",
)
parser.add_argument("--output", type=Path, default=Path("artifacts/paged-kv-pressure"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
binary = str(args.binary.resolve())
requests = [
    {
        "id": 1,
        "model": 1,
        "input": {"Sequence": {"tokens": [1, 2, 3, 5, 8, 13]}},
        "workload": {"Generate": {"max_new_tokens": 5}},
    },
    {
        "id": 2,
        "model": 1,
        "input": {"Sequence": {"tokens": [2, 3, 4, 6, 9, 14]}},
        "workload": {"Generate": {"max_new_tokens": 5}},
    },
]


class Agent:
    def __init__(self, backend, blocks=None):
        command = [
            str(args.test_binary.resolve()) if backend == "test-cpu" else binary,
            "--backend",
            backend,
            "--kv-page-tokens",
            "2",
        ]
        if blocks is not None:
            command += ["--kv-cache-blocks", str(blocks)]
        command += ["agent", "--package", "examples/qwen-hybrid-tiny"]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.transcript = []
        self.now = 0

    def rpc(self, method, params=None):
        message = {"jsonrpc": "2.0", "id": len(self.transcript) + 1, "method": method}
        if params is not None:
            message["params"] = params
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        assert line, f"Agent exited: {self.process.poll()}"
        response = json.loads(line)
        assert "error" not in response, response
        self.transcript.append({"request": message, "response": response})
        return response["result"]

    def tick(self):
        self.rpc("runtime.tick", {"now_us": self.now})
        self.now += 1
        time.sleep(0.001)

    def finish(self):
        deadline = time.monotonic() + 30
        while self.rpc("runtime.inspect")["active_requests"]:
            assert time.monotonic() < deadline, "bounded page pool stalled"
            self.tick()
        result = self.rpc("runtime.query")["requests"]
        assert all(r["completed"]["measurement"]["successful"] for r in result)
        return {r["request"]["id"]: r["completed"]["output"] for r in result}

    def close(self, name):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        assert self.process.returncode == 0
        Path(args.output, name + ".json").write_text(
            json.dumps(self.transcript, ensure_ascii=False, indent=2) + "\n"
        )


host = Agent("test-cpu")
try:
    for request in requests:
        host.rpc("runtime.submit", request)
    expected = host.finish()
finally:
    host.close("host")

metal = Agent("metal", 5)
try:
    capabilities = metal.rpc("executor.inspect")["capabilities"]
    assert capabilities["backend"]["kind"] == "metal" and "nvidia" not in capabilities
    assert capabilities["backend"]["capabilities"]["simd_width"] > 0
    for request in requests:
        metal.rpc("runtime.submit", request)
    cache = metal.rpc("kv.inspect")
    assert cache["pool"]["total_blocks"] == 5
    assert cache["pool"]["active_blocks"] == 0
    assert cache["logical_state"]["allocated_pages"] == 0
    deadline = time.monotonic() + 30
    while metal.rpc("kv.inspect")["preemptions"] == 0:
        assert time.monotonic() < deadline, "expected page-pressure preemption"
        metal.tick()
    checkpoint = metal.rpc("snapshot.capture", {"now_us": metal.now})
    while checkpoint["pending"]:
        assert time.monotonic() < deadline
        time.sleep(0.001)
        metal.now += 1
        checkpoint = metal.rpc("snapshot.capture", {"now_us": metal.now})
    assert checkpoint["snapshot"]["schema_version"] == 6
    assert checkpoint["snapshot"]["preemption_focus"] is not None
    metal.rpc("snapshot.replay", checkpoint["snapshot"])
    metal.now += 1
    actual = metal.finish()
    assert actual == expected
    cache = metal.rpc("cache.inspect")
    assert cache["preemptions"] > 0
    assert cache["pool"]["available_blocks"] == 5
    assert cache["pool"]["active_blocks"] == 0
    assert cache["logical_state"]["allocated_pages"] == 0
    summary = {
        "backend": "metal",
        "blocks": 5,
        "page_tokens": 2,
        "output_parity": True,
        "preemptions": cache["preemptions"],
        "checkpoint_during_preemption": True,
        "final_pool": cache["pool"],
    }
    Path(args.output, "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
finally:
    metal.close("metal")
