"""Extract validated cuTile 0.4 cache payloads and inspect actual SASS instructions."""

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuobjdump", default="/opt/cuda/bin/cuobjdump")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(args.cache.rglob("*.cubin")):
        data = source.read_bytes()
        if (
            len(data) < 96
            or data[:12] != b"CUTILECUBIN\0"
            or struct.unpack_from("<H", data, 12)[0] != 1
        ):
            raise ValueError(f"Unrecognized cuTile cache format: {source}")
        gpu_length, tool_length = struct.unpack_from("<HH", data, 80)
        offset = 96 + gpu_length + tool_length
        payload = data[offset:]
        if (
            len(payload) != struct.unpack_from("<Q", data, 88)[0]
            or hashlib.sha256(payload).digest() != data[16:48]
        ):
            raise ValueError(f"Invalid cache payload: {source}")
        target = data[96 : 96 + gpu_length].decode()
        cubin = args.output / f"{source.stem}.cubin"
        cubin.write_bytes(payload)
        process = subprocess.run(
            [args.cuobjdump, "--dump-sass", str(cubin)], capture_output=True, text=True, check=True
        )
        (args.output / f"{source.stem}.sass").write_text(process.stdout)
        opcodes = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9_.]+)", process.stdout)
        counts = {name: opcodes.count(name) for name in sorted(set(opcodes))}
        records.append(
            {
                "source": str(source),
                "target": target,
                "opt_level": data[84],
                "flags": data[85],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "opcode_counts": counts,
                "functions": re.findall(r"Function\s*:\s*(\S+)", process.stdout),
                "mma_opcodes": [name for name in counts if "MMA" in name],
            }
        )
    if not records:
        raise ValueError("No cuTile cubin cache files found")
    (args.output / "index.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    "target": r["target"],
                    "functions": r["functions"],
                    "mma_opcodes": r["mma_opcodes"],
                    "opt_level": r["opt_level"],
                }
                for r in records
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
