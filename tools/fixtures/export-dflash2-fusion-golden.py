"""Cold-path DFlash2 fusion golden: the draft's feature projection on recorded inputs.

`combine_hidden_states` in vLLM's `qwen3_dflash.py` shows what the released draft expects: the five
target taps concatenated in order, projected by `fc` to the draft's hidden size, with no bias and no
normalisation inside the projection. This exporter records that projection's output for one
deterministic input so the draft graph has a bit-level target to compare against, instead of only
the shape contract.

The weights are not in the repository, so the golden records the model identity it came from:
the configuration digest registered with the fixture, the tensor name and shape, torch's version and
the arithmetic used. Re-run with `--model <path to Qwen3.8-27B-DFlash2>`.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

# One deterministic input row: tap `t`, channel `c` maps to a bounded, non-repeating value, so the
# golden does not depend on a random seed that a future reader cannot replay.
TAP_SCALE = 5_120
MODULUS = 251
OFFSET = 125
DIVISOR = 128


def deterministic_input(rows: int, channels: int) -> torch.Tensor:
    taps = torch.arange(rows, dtype=torch.int64).view(rows, 1) * TAP_SCALE
    channels_index = torch.arange(channels, dtype=torch.int64).view(1, channels)
    values = (taps + channels_index) % MODULUS - OFFSET
    return values.to(torch.float32) / DIVISOR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("examples/qwen3.8-27b-dflash2"))
    args = parser.parse_args()

    config_path = args.fixture / "config.json"
    config = json.loads(config_path.read_text())
    taps = config["dflash_config"]["target_layer_ids"]
    hidden = config["hidden_size"]

    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as weights:
        fc = weights.get_tensor("fc.weight")
        if tuple(fc.shape) != (hidden, hidden * len(taps)):
            raise ValueError(f"unexpected fc shape {tuple(fc.shape)}")
        fused = torch.nn.functional.linear(
            deterministic_input(len(taps), hidden).reshape(1, -1), fc.to(torch.float32)
        )

    golden = {
        "model": {
            "repository": "z-lab/Qwen3.8-27B-DFlash2",
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "tensor": "fc.weight",
            "shape": list(fc.shape),
            # The safetensors name, which is the vocabulary the crate and the package use.
            "dtype": {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}.get(
                str(fc.dtype).split(".")[-1], str(fc.dtype).split(".")[-1]
            ),
        },
        "target_layer_ids": taps,
        "formula": {
            "input": (
                f"tap t, channel c -> ((t * {TAP_SCALE} + c) % {MODULUS} - {OFFSET}) / {DIVISOR}"
            ),
            "projection": "fused = fc @ concat(taps) in float32, no bias and no normalisation",
        },
        "tolerance": {
            "absolute": 2e-2,
            "relative": 2e-2,
            "reason": "bf16 weights, f32 accumulation",
        },
        "generator": {
            "script": "tools/fixtures/export-dflash2-fusion-golden.py",
            "torch": torch.__version__,
        },
        "fused": [float(value) for value in fused.reshape(-1)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(golden) + "\n")
    print(f"wrote {args.output} ({len(golden['fused'])} values)")


if __name__ == "__main__":
    main()
