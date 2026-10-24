"""Compile one pinned FA4 SM120 artifact with plain Float16 operands.

Minimal adaptation of crates/backend/cuda/kernels/fa4-attention/compile.py:
the only semantic changes are the operand/output dtype (Float16 instead of
BFloat16), the profile string, the host prefix, and extra PTX dtype checks.
The upstream source, tile/tiling, SDK and architecture are unchanged.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source-tree", type=Path, required=True)
ap.add_argument("--sdk-site", type=Path, required=True)
ap.add_argument("--quack", type=Path, required=True)
ap.add_argument("--tvm-ffi", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
arch = "sm_120"
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    CUTE_DSL_ARCH=arch,
    CUTE_DSL_KEEP="ptx,cubin",
    CUTE_DSL_DUMP_DIR=str(out),
    CUDA_CACHE_PATH=str(out / "cuda-cache"),
)
sys.path[:0] = [
    str(args.source_tree.resolve()),
    str(args.sdk_site.resolve()),
    str(args.sdk_site.resolve() / "nvidia_cutlass_dsl/dsl_packages"),
    str(args.quack.resolve()),
    str(args.tvm_ffi.resolve()),
]
assert importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
rec = {
    "completed": False,
    "GPU_execution": False,
    "started_unix": time.time(),
    "architecture": arch,
    "candidate": "sm120-f16",
    "dtype": "Float16",
    "profile": "fp16-inputs-f32-accumulator-fp16-output",
    "source_official_commit": "e9d8944c855489e03c66262d07ba4c81d9d8dd31",
    "on_hardware_validated": False,
    "quality_accepted": False,
}
try:
    import cutlass
    import cutlass.cute as cute
    import quack
    import tvm_ffi
    from cuda.bindings import driver as cuda
    from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120

    rec["quack_version"] = getattr(quack, "__version__", None)
    rec["tvm_ffi_path"] = str(Path(tvm_ffi.__file__).resolve())
    rec["sdk_site"] = str(args.sdk_site.resolve())
    rec["quack_path"] = str(args.quack.resolve())
    rec["tvm_ffi_site"] = str(args.tvm_ffi.resolve())

    FN = FlashAttentionForwardSm120(
        cutlass.Float16,
        256,
        256,
        6,
        is_causal=True,
        pack_gqa=True,
        tile_m=64,
        tile_n=64,
        num_stages=1,
        num_threads=128,
        Q_in_regs=False,
    )
    config = {
        "tile": [64, 64],
        "threads": 128,
        "cluster": [1, 1, 1],
        "pack_gqa": True,
        "features": ["warp-MMA", "cp.async"],
    }

    @cute.jit
    def entry(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        usedq: cute.Tensor,
        usedk: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        FN(q, k, v, o, None, scale, mSeqUsedQ=usedq, mSeqUsedK=usedk, stream=stream)

    shape = (1, cute.sym_int32(divisibility=64), 24, 256)

    def make(shape, dtype=cutlass.Float16, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=align,
        )

    q = make(shape)
    k = make((1, cute.sym_int32(divisibility=32), 4, 256))
    v = make((1, cute.sym_int32(divisibility=32), 4, 256))
    o = make(shape)
    usedq = make((1,), cutlass.Int32, 4)
    usedk = make((1,), cutlass.Int32, 4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    compiled = cute.compile(
        entry,
        q,
        k,
        v,
        o,
        usedq,
        usedk,
        cutlass.Float32(0.0625),
        stream,
        options="--gpu-arch " + arch + " --keep-ptx --keep-cubin",
    )
    rec["config"] = config
    for name, content in [
        ("attention.ptx", compiled.__ptx__),
        ("attention.cubin", compiled.__cubin__),
    ]:
        if isinstance(content, str) and name.endswith("cubin") and Path(content).exists():
            content = Path(content).read_bytes()
        assert content is not None, name
        if isinstance(content, str):
            content = content.encode()
        (out / name).write_bytes(content)
        rec[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    ptx = (out / "attention.ptx").read_text()
    rec["kernel_symbols"] = list(compiled.kernel_info.keys())
    patterns = {
        "warp_MMA": r"mma\.sync",
        "f16_mma": r"mma\.sync[^\n]*\.f16\.f16",
        "bf16_mma": r"mma\.sync[^\n]*\.bf16\.bf16",
        "WGMMA": r"wgmma\.mma_async",
        "TMA": r"cp\.async\.bulk\.tensor",
        "tcgen05": r"tcgen05\.mma",
        "TMEM": r"tcgen05\.(alloc|ld|st)",
        "two_CTA": r"cta_group::2",
    }
    rec["PTX_features"] = {name: bool(re.search(regex, ptx)) for name, regex in patterns.items()}
    compiled.export_to_c(str(out), "attention_host", function_prefix="infer_fa4_f16_sm120")
    rec["sdk_version"] = importlib.metadata.version("nvidia-cutlass-dsl")
    assert rec["sdk_version"] == "4.8.0"
    rec["completed"] = True
except BaseException as e:
    rec.update(error=repr(e), traceback=traceback.format_exc())
    raise
finally:
    rec["finished_unix"] = time.time()
    with (out / "result.json").open("x") as f:
        json.dump(rec, f, indent=2)
    print(
        json.dumps({k: v for k, v in rec.items() if k not in ["traceback", "kernel_symbols"]}),
        flush=True,
    )
