"""Recover the FP16 AOT bundle from pinned sources without machine-local scripts."""

import json
import shutil
import tempfile
from pathlib import Path


def rebuild_f16(entry, args, directory, out, run, check_hash, require):
    require(
        args.upstream is not None and args.sdk_site is not None,
        "FP16 FA4 requires --upstream and --sdk-site",
    )
    require(
        args.quack is not None and args.tvm_ffi is not None,
        "FP16 FA4 requires --quack and --tvm-ffi package directories",
    )
    provenance = json.loads((directory / "provenance.json").read_text())
    require((args.quack / "quack_kernels-0.6.5.dist-info").is_dir(), "Expected quack 0.6.5")
    require((args.tvm_ffi / "apache_tvm_ffi-0.1.11.dist-info").is_dir(), "Expected tvm-ffi 0.1.11")
    core = out / "sm120"
    with tempfile.TemporaryDirectory(prefix="infer-fa4-f16-") as temporary:
        tree = Path(temporary)
        for name, digest in provenance["source_files"].items():
            dest = tree / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name == "flash_attn/__init__.py":
                dest.write_text("# AOT compilation namespace for verified CuTe sources.\n")
            else:
                check_hash(args.upstream / name, digest)
                shutil.copyfile(args.upstream / name, dest)
            check_hash(dest, digest)
        run(
            [
                args.python,
                Path(__file__).with_name("compile_f16.py"),
                "--source-tree",
                tree,
                "--sdk-site",
                args.sdk_site.resolve(),
                "--quack",
                args.quack.resolve(),
                "--tvm-ffi",
                args.tvm_ffi.resolve(),
                "--output",
                core,
            ],
            directory,
            out / "compile.log",
        )
    # Exact comparison covers instructions, host argument layout and generated object code.
    for name in ("attention.ptx", "attention.cubin", "attention_host.h", "attention_host.o"):
        check_hash(core / name, provenance["artifact_hashes"][name])
    template = directory.parent / "fa4-attention/host.cpp.in"
    (core / "host.cpp").write_text(template.read_text().replace("PREFIX", "infer_fa4_f16_sm120"))
    check_hash(core / "host.cpp", provenance["artifact_hashes"]["host.cpp"])
    cuda = args.cuda_root.resolve()
    run(
        [
            args.cxx,
            "-shared",
            "-fPIC",
            "-O2",
            "-fvisibility=hidden",
            "-I" + str(cuda / "include"),
            core / "host.cpp",
            core / "attention_host.o",
            "-L" + str(cuda / "lib64"),
            "-L" + str(cuda / "lib64/stubs"),
            "-lcudart",
            "-lcuda",
            "-Wl,-z,defs",
            "-Wl,-z,relro,-z,now",
            "-Wl,-rpath,$ORIGIN",
            "-o",
            core / "attention.so",
        ],
        directory,
        out / "link.log",
    )
    check_hash(core / "attention.so", provenance["artifact_hashes"]["attention.so"])
    generated = out / entry["ptx"]
    run(
        [cuda / "bin/nvcc", *entry["arguments"], directory / entry["source"], "-o", generated],
        directory,
        out / "bridge.log",
    )
    return generated
