"""Verify the fixed SM120 FP16 AOT identity, host ABI and FP16 instruction family."""
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert manifest["schema"] == 1 and manifest["host_ABI"] == 1
    assert manifest["runtime_python"] is False
    assert manifest["host_platform"] == "linux-x86_64"
    assert manifest["cutlass_dsl"] == "4.8.0"
    assert manifest["profile"].startswith("fa4-f16-qkv")
    assert manifest["upstream_commit"] == provenance["upstream_commit"]
    assert provenance["schema"] == "fp16-fa4-artifact-v1"
    # The recorded compile phase never opened a GPU; hardware evidence is the separate probe.
    assert provenance["GPU_execution"] is False
    assert provenance["compile_result"]["dtype"] == "Float16"
    features = provenance["compile_result"]["PTX_features"]
    assert features["f16_mma"] is True and features["bf16_mma"] is False
    assert features["warp_MMA"] is True and features["WGMMA"] is False
    assert provenance["profile"] == "fp16-inputs-f32-accumulator-fp16-output"
    assert provenance["source_tree_sha256"] == (
        "06cc6745698d3e51dfe7c63d09ee302205f7fedd486be13b2ef254d05a4a3fb0"
    )
    assert provenance["production_bf16_modified"] is False
    entry = manifest["entries"][0]
    assert len(manifest["entries"]) == 1 and entry["name"] == "sm120"
    assert entry["architecture"] == "sm_120"
    assert entry["profile"] == "fp16-inputs-f32-accumulator-fp16-output"
    assert entry["dtype"] == "Float16"
    assert entry["shape"] == {
        "query_heads": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "lanes_multiple": 64,
        "context_dynamic": True,
    }
    for file, digest in entry["files"].items():
        assert sha(ROOT / file) == digest, file
    assert entry["task_quality_accepted"] is False
    assert entry["on_hardware_validated"] is True
    assert entry["attention_ptx_sha256"] == (
        "05c0786195978fb76dbf8ee4b0ca1bcbaec04c2c6ff81b543549619ab9ceea8d"
    )
    assert entry["attention_cubin_sha256"] == (
        "fbf42b66627618f0f87e93cc1da0c6f7f5c7deada8d2267b5bf3a035e60b5fab"
    )
    ptx = (ROOT / "bridge.ptx").read_text()
    assert re.search(r"\.target\s+sm_90\b", ptx)
    assert ".entry prepare_fa4_f16(" in ptx and ".entry finish_fa4_f16(" in ptx
    header = ptx.split(".entry prepare_fa4_f16(", 1)[1].split(")", 1)[0]
    assert header.count(".param") == 13
    assert header.count(".u64") == 9
    assert header.count(".u32") == 2
    assert header.count(".f32") == 2
    image = (ROOT / "sm120/attention.so").read_bytes()
    assert image[:6] == b"\x7fELF\x02\x01"
    assert struct.unpack_from("<H", image, 18)[0] == 62
    for symbol in [b"infer_fa4_initialize", b"infer_fa4_forward"]:
        assert symbol + b"\0" in image
    dynamic = subprocess.check_output(
        ["readelf", "-d", str(ROOT / "sm120/attention.so")],
        text=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    deps = set(re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", dynamic))
    assert {"libcudart.so.13", "libcuda.so.1"} <= deps
    assert deps <= set(manifest["runtime_dependencies"])
    assert not any("python" in dep.lower() or "cutedsl" in dep for dep in deps)
    print(
        "FA4 FP16 SM120: bridge/PTX identity, FP16 MMA family, host ABI and CUDA "
        "runtime dependencies verified; task quality pending"
    )


if __name__ == "__main__":
    main()
