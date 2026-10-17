#!/usr/bin/env python3
"""Verify the combined vector key/consumer CUDA artifact and its pinned base chain.

The combined kernel is a bit-exact composition of two accepted variants on the pinned
`vector-attention` source, so the CPU-side identity chain is:
`vector-attention/attention-sm120.*` -> `v-combined.{cu,ptx}` with the exact nvcc contract.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SYMBOL = "attention_fp8_256_vec_k_combined"
CONTRACT = [
    "--ptx",
    "-arch=compute_120",
    "-O3",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path)
    args = parser.parse_args()
    provenance = json.loads((ROOT / "provenance.json").read_text())
    source = ROOT / "v-combined.cu"
    embedded = ROOT / "v-combined.ptx"
    base = ROOT.parent / "vector-attention"
    if digest(source) != provenance["source_sha256"]:
        raise ValueError("CUDA source differs from provenance")
    if digest(embedded) != provenance["ptx_sha256"]:
        raise ValueError("Embedded PTX differs from provenance")
    if provenance["arguments"] != CONTRACT:
        raise ValueError("Compile arguments differ from the pinned contract")
    base_source = base / "attention-sm120.cu"
    base_ptx = base / "attention-sm120.ptx"
    if digest(base_source) != provenance["base_source_sha256"]:
        raise ValueError("Base vector CUDA source differs from provenance")
    if digest(base_ptx) != provenance["canonical_attention_ptx_sha256"]:
        raise ValueError("Canonical vector PTX reference differs from provenance")
    if provenance["base_ptx_sha256"] != provenance["canonical_attention_ptx_sha256"]:
        raise ValueError("Base and canonical PTX identities diverged")
    text = embedded.read_text()
    if ".target sm_120" not in text or text.count(f".visible .entry {SYMBOL}(") != 1:
        raise ValueError("Unexpected target or entry")
    if source.read_text().count(f"{SYMBOL}(") != 1:
        raise ValueError("Unexpected device source entry")
    if args.generated is not None and args.generated.read_bytes() != embedded.read_bytes():
        raise ValueError("Regenerated PTX differs from the embedded artifact")
    print(json.dumps({
        "source_sha256": digest(source),
        "embedded_ptx_sha256": digest(embedded),
        "canonical_vector_ptx_sha256": digest(base_ptx),
        "regenerated_byte_identical": args.generated is not None,
        "compiler": provenance["compiler"],
    }))


if __name__ == "__main__":
    main()
