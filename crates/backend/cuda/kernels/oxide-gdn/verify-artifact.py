#!/usr/bin/env python3
"""Verify cargo oxide build --arch sm_120 --no-fmad against the embedded artifact.

The pinned SDK puts StableCrateId in module-scope shared symbols. Path and Git
dependencies produce different crate identities. This verifier accepts only a
consistent renaming of those symbols in this single-kernel module: every other
PTX byte, including declarations, instructions and arguments, must match.
"""
from pathlib import Path
import argparse, hashlib, json, re, tomllib

ROOT = Path(__file__).resolve().parent
PATTERN = re.compile(r"__shared_mem_([a-f0-9]{16})_([0-9]+)")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    matches = PATTERN.findall(text)
    if len({namespace for namespace, _ in matches}) != 1:
        raise ValueError("Expected one shared-memory namespace")
    declared = re.findall(r"^\.visible \.shared \.align 4 \.b8 "
                          r"__shared_mem_[a-f0-9]{16}_([0-9]+)\[([0-9]+)\];$", text, re.M)
    if sorted(declared) != [("0", "1024"), ("1", "32"), ("2", "512")]:
        raise ValueError("Shared allocation layout differs")
    if {index for _, index in matches} != {"0", "1", "2"}:
        raise ValueError("Unexpected shared symbol reference")
    if text.count(".visible .entry gdn_f32_128(") != 1:
        raise ValueError("Expected one GDN entry")
    return PATTERN.sub(lambda match: "__shared_mem_CANONICAL_" + match.group(2), text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=ROOT / "oxide_gdn_simt.ptx")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "provenance.json").read_text())
    source = (ROOT / "src/main.rs").read_bytes()
    frozen = (ROOT / "gdn-sm120.ptx").read_bytes()
    generated = args.generated.read_bytes()
    if digest(source) != provenance["source_sha256"] or digest(frozen) != provenance["ptx_sha256"]:
        raise ValueError("Embedded source/artifact differs from provenance")
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    for name in ("cuda-core", "cuda-device", "cuda-host"):
        packages = [entry for entry in lock["package"] if entry["name"] == name]
        if len(packages) != 1 or not packages[0].get("source", "").endswith("#" + provenance["cuda_rust_commit"]):
            raise ValueError("SDK dependency revision is not pinned: " + name)
    canonical = normalize(frozen.decode())
    if canonical != normalize(generated.decode()):
        raise ValueError("Generated PTX changes exceed shared symbol renaming")
    print(json.dumps({
        "source_sha256": digest(source), "embedded_ptx_sha256": digest(frozen),
        "generated_ptx_sha256": digest(generated), "ptx_byte_identical": frozen == generated,
        "shared_symbol_equivalent": True, "normalized_ptx_sha256": digest(canonical.encode()),
        "cuda_rust_commit": provenance["cuda_rust_commit"],
    }))


if __name__ == "__main__":
    main()
