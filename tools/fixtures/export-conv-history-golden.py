"""Cold-path golden for the causal-conv history step, kept before its CPU executor goes.

The convolution's state is the trailing window of its own inputs, and R1 commits it by gathering
that window from the base history and the accepted inputs rather than by recomputing outputs. Both
the step and the gather lived only in the CPU host executor that E1 deletes, so this records them:
the weights are laid out per channel, the sum accumulates in f64, the output is rounded
back to f32 through a SiLU, and the history rotates one input per step.

    python3 tools/fixtures/export-conv-history-golden.py \
        --output examples/recurrent-conv/golden.json
"""

import argparse
import json
import math
import struct
from pathlib import Path

CHANNELS = 4
KERNEL = 3
STEPS = 3
MODULUS = 19
OFFSET = 9
DIVISOR = 13
HISTORY = KERNEL - 1


def f32(value: float) -> float:
    """The state and the inputs are f32, so the golden rounds where the reference rounds."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def declared_inputs(step: int) -> tuple[list[float], list[float]]:
    """The activation row and the per-channel weights, from one deterministic formula."""
    activation = [
        f32(((step * CHANNELS + index) % MODULUS - OFFSET) / DIVISOR) for index in range(CHANNELS)
    ]
    weights = [
        f32(((step * CHANNELS * KERNEL + index) % MODULUS - OFFSET) / DIVISOR)
        for index in range(CHANNELS * KERNEL)
    ]
    return activation, weights


def step(history: list[float], activation: list[float], weights: list[float]):
    """One convolution step and its history rotation, in the reference's order."""
    updated = list(history)
    out = [0.0] * CHANNELS
    for channel in range(CHANNELS):
        row = weights[channel * KERNEL : (channel + 1) * KERNEL]
        past = updated[channel * HISTORY : (channel + 1) * HISTORY]
        total = (
            # The trailing weight belongs to the current input, so the window pairs with the
            # leading ones.
            sum(x * w for x, w in zip(past, row[:-1], strict=True))
            + activation[channel] * row[KERNEL - 1]
        )
        out[channel] = f32(silu(total))
        if past:
            past = [*past[1:], activation[channel]]
            updated[channel * HISTORY : (channel + 1) * HISTORY] = past
    return updated, out


def gathered(base: list[float], accepted: list[list[float]]) -> list[float]:
    """R1's commit: take the trailing window from the base history and the accepted inputs."""
    per_channel = []
    for channel in range(CHANNELS):
        window = list(base[channel * HISTORY : (channel + 1) * HISTORY])
        for activation in accepted:
            window = [*window, activation[channel]][-HISTORY:] if HISTORY else []
        per_channel.extend(window)
    return per_channel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    inputs = [declared_inputs(step) for step in range(STEPS)]
    base = [0.0] * (CHANNELS * HISTORY)

    history = list(base)
    histories = []
    outputs = []
    for activation, weights in inputs:
        history, out = step(history, activation, weights)
        histories.append(history)
        outputs.append(out)

    accepted = [activation for activation, _ in inputs]
    replay_matches = history == gathered(base, accepted)

    golden = {
        "op": "causal convolution history step",
        "geometry": {"channels": CHANNELS, "kernel": KERNEL, "history": CHANNELS * HISTORY},
        "formula": {
            "weights": "inputs[1] holds one row of kernel weights per channel",
            "sum": (
                "past . weights[..kernel-1] + activation * weights[kernel-1], accumulated in f64"
            ),
            "output": "out[channel] = silu(sum) rounded to f32",
            "history": "the channel's window rotates left and appends the activation",
            "precision": "f32 inputs and state, f64 accumulation, f32 output",
        },
        "declared_inputs": {
            "formula": (
                f"activation[channel], step -> "
                f"((step * {CHANNELS} + channel) % {MODULUS} - {OFFSET}) / {DIVISOR}; "
                "weights use the same rule over the flattened row"
            ),
            "rows": [
                {"activation": activation, "weights": weights} for activation, weights in inputs
            ],
        },
        "base_history": base,
        "histories": histories,
        "outputs": outputs,
        "replay_property": {
            "claim": (
                "gathering the trailing window from the base history and the accepted inputs "
                "reproduces the history the steps produced"
            ),
            "holds": replay_matches,
        },
        "tolerance": {
            "absolute": 1e-6,
            "relative": 1e-5,
            "reason": "the reference accumulates in f64 and rounds the output once",
        },
        "generator": {"script": "tools/fixtures/export-conv-history-golden.py"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(golden) + "\n")
    print(
        f"wrote {args.output}: {len(histories)} steps, history {len(base)}, "
        f"gather holds: {replay_matches}"
    )


if __name__ == "__main__":
    main()
