#!/usr/bin/env python3
"""Verify the embedded CUDA C++ device artifact and optional regenerated PTX."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path)
    args = parser.parse_args()
    provenance = json.loads((ROOT / "provenance.json").read_text())
    source = ROOT / "attention-sm120.cu"
    embedded = ROOT / "attention-sm120.ptx"
    if digest(source) != provenance["source_sha256"]:
        raise ValueError("CUDA source differs from provenance")
    if digest(embedded) != provenance["ptx_sha256"]:
        raise ValueError("Embedded PTX differs from provenance")
    canonical = ROOT.parent / "oxide-attention" / "attention-sm120.ptx"
    if digest(canonical) != provenance["canonical_attention_ptx_sha256"]:
        raise ValueError("Canonical PTX reference differs from provenance")
    text = embedded.read_text()
    if ".target sm_120" not in text or text.count(".visible .entry attention_fp8_256(") != 1:
        raise ValueError("Unexpected target or entry")
    if args.generated is not None and args.generated.read_bytes() != embedded.read_bytes():
        raise ValueError("Regenerated PTX differs from the embedded artifact")
    print(json.dumps({
        "source_sha256": digest(source),
        "embedded_ptx_sha256": digest(embedded),
        "regenerated_byte_identical": args.generated is not None,
        "compiler": provenance["compiler"],
    }))


if __name__ == "__main__":
    main()
