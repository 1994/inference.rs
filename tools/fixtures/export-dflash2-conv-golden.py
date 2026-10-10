"""Cold-path DFlash2 grouped-convolution golden.

The draft's double-tap dynamic convolution runs twice per layer: `prepare` folds the
attention input with side 0 and returns side 1 for `finish`. The reference path in
vLLM's `qwen3_dflash2.py` settles the shapes the device kernel alone leaves open -
`base_kernel` is `[side, tap, channel]` and `kernel_projection` produces
`[row, side, tap, group]` - so this exporter can state the arithmetic exactly:

    coefficients = kernel_projection(hidden).reshape(rows, 2, taps, groups)
    blocks = hidden.reshape(rows, groups, group_size)
    weight = base_kernel[side].reshape(1, taps, groups, group_size) \
        + coefficients[:, side, :, :, None]
    output = weight[:, 0] * blocks
    for tap in 1..taps:
        output += weight[:, tap] * shift_by(blocks, tap) * (row % block_size >= tap)

The weights are not in the repository, so the golden records the model identity and
the rounding: values are rounded to six significant digits, far inside the recorded
tolerance. Re-run with `--model <path to Qwen3.8-27B-DFlash2>`.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

MODULUS = 251
OFFSET = 125
DIVISOR = 128
LAYER = 0
SIDE = 1
SIGNIFICANT_DIGITS = 6


def deterministic_hidden(rows: int, channels: int) -> torch.Tensor:
    row_index = torch.arange(rows, dtype=torch.int64).view(rows, 1) * channels
    channel_index = torch.arange(channels, dtype=torch.int64).view(1, channels)
    values = (row_index + channel_index) % MODULUS - OFFSET
    return values.to(torch.float32) / DIVISOR


def grouped_conv(hidden, coefficients, base, block_size, num_groups, group_size, taps):
    rows = hidden.shape[0]
    blocks = hidden.reshape(rows, num_groups, group_size)
    weight = base.reshape(1, taps, num_groups, group_size) + coefficients.unsqueeze(-1)
    output = weight[:, 0] * blocks
    position = torch.arange(rows, dtype=torch.int64) % block_size
    for tap in range(1, taps):
        shifted = torch.cat((torch.zeros_like(blocks[:tap]), blocks[:-tap]), dim=0)
        output = output + weight[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    # The reference flattens the group dimension back into the channel dimension.
    return output.reshape(rows, num_groups * group_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("examples/qwen3.8-27b-dflash2"))
    args = parser.parse_args()

    config_path = args.fixture / "config.json"
    config = json.loads(config_path.read_text())
    hidden_size = config["hidden_size"]
    draft = config["dflash_config"]
    block_size = draft["block_size"]
    group_size = draft["conv_group_size"]
    taps = draft["conv_kernel_size"]
    groups = hidden_size // group_size

    prefix = f"layers.{LAYER}.attention_conv."
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as weights:
        projection = weights.get_tensor(prefix + "kernel_projection.weight")
        base = weights.get_tensor(prefix + "base_kernel")
        expected_projection = (2 * taps * groups, hidden_size)
        expected_base = (2, taps, hidden_size)
        if tuple(projection.shape) != expected_projection:
            raise ValueError(f"unexpected kernel_projection shape {tuple(projection.shape)}")
        if tuple(base.shape) != expected_base:
            raise ValueError(f"unexpected base_kernel shape {tuple(base.shape)}")

        hidden = deterministic_hidden(block_size, hidden_size)
        coefficients = torch.nn.functional.linear(hidden, projection.to(torch.float32))
        coefficients = coefficients.reshape(block_size, 2, taps, groups)[:, SIDE]
        output = grouped_conv(
            hidden, coefficients, base[SIDE].to(torch.float32), block_size, groups, group_size, taps
        )

    rounded = [
        [float(f"{value:.{SIGNIFICANT_DIGITS}g}") for value in row] for row in output.tolist()
    ]
    golden = {
        "model": {
            "repository": "z-lab/Qwen3.8-27B-DFlash2",
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "tensors": {
                prefix + "kernel_projection.weight": list(projection.shape),
                prefix + "base_kernel": list(base.shape),
            },
            "dtype": "BF16",
        },
        "layer": LAYER,
        "side": SIDE,
        "side_role": "finish",
        "block_size": block_size,
        "group_size": group_size,
        "taps": taps,
        "rows": block_size,
        "formula": {
            "input": (
                f"hidden[r, c] = ((r * {hidden_size} + c) % {MODULUS} - {OFFSET}) / {DIVISOR}"
            ),
            "conv": (
                "coefficients = kernel_projection @ hidden reshaped [row, side, tap, group]; "
                "weight = base_kernel[side] + coefficients over the group; "
                "output = weight[0] * block; then for each later tap, "
                "output += weight[tap] * block[row - tap] where row % block_size >= tap"
            ),
            "prepare": "side 0 runs before attention and returns side 1 for finish",
        },
        "rounding": {"significant_digits": SIGNIFICANT_DIGITS, "inside_tolerance": True},
        "tolerance": {
            "absolute": 2e-2,
            "relative": 2e-2,
            "reason": "bf16 weights, f32 accumulation",
        },
        "generator": {
            "script": "tools/fixtures/export-dflash2-conv-golden.py",
            "torch": torch.__version__,
        },
        "output": rounded,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(golden) + "\n")
    print(f"wrote {args.output} ({len(rounded)}x{len(rounded[0])} values)")


if __name__ == "__main__":
    main()
