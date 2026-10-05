"""Run native Metal acceptance with a temporary HTTP server and guaranteed cleanup."""

import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/gates-metal"


def wait_ready(server, url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("Metal server exited; see artifacts/gates-metal/server.log")
        try:
            with urlopen(url + "/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(0.1)
    raise TimeoutError("Metal server did not become ready within 30 seconds")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"127.0.0.1:{port}"
    with (OUTPUT / "server.log").open("w") as log:
        server = subprocess.Popen(
            [
                str(ROOT / "target/release/infer"),
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
            wait_ready(server, f"http://{address}")
            subprocess.run(
                [
                    sys.executable,
                    "tools/validation/smoke-host.py",
                    "--binary",
                    "target/release/infer",
                    "--backend",
                    "metal",
                    "--server-url",
                    f"http://{address}",
                    "--output",
                    str(OUTPUT),
                ],
                cwd=ROOT,
                check=True,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    subprocess.run(
        [
            sys.executable,
            "tools/validation/smoke-paged-kv.py",
            "--binary",
            "target/release/infer",
            "--test-binary",
            "target/debug/infer",
            "--output",
            "artifacts/gates-metal-pages",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
