"""Cold-path golden for the gated-delta state step, kept before its CPU executor goes.

The recurrent state update the host executor implements is the reference the CUDA kernel is
compared against, and E1 deletes that executor. This exporter re-implements the step from the
formula and records it - the normalization of q and k, the sigmoid gate, the softplus log-decay,
the rank-one update, and the readout - so the numbers survive the code they came from.

It also records the property R1 is built on: replaying the recorded inputs from the base state to
the accepted length reproduces the per-step states, which is what lets a commit roll back by
replaying an accepted prefix instead of snapshotting every candidate.

    python3 tools/fixtures/export-delta-state-golden.py \
        --output examples/recurrent-delta/golden.json
"""

import argparse
import json
import math
import struct
from pathlib import Path

# A small geometry with two key heads shared by four value heads, which is the grouping the step
# has to get right.
KEY_HEADS = 2
VALUE_HEADS = 4
KEY_DIM = 4
VALUE_DIM = 4
STEPS = 3
NORM_EPSILON = 1e-6
MODULUS = 23
OFFSET = 11
DIVISOR = 17


def f32(value: float) -> float:
    """The state is stored as f32 between steps, so the replay has to round the same way."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def softplus(value: float) -> float:
    # log1p(exp(x)) is stable for the small values this golden uses.
    return math.log1p(math.exp(value))


def declared_inputs(step: int) -> list[list[float]]:
    """The five input rows the step reads, from one deterministic formula."""
    key_size = KEY_HEADS * KEY_DIM
    value_size = VALUE_HEADS * VALUE_DIM
    widths = (
        2 * key_size + value_size,
        VALUE_HEADS,
        VALUE_HEADS,
        VALUE_HEADS,
        VALUE_HEADS,
    )
    return [
        [((step * width + index) % MODULUS - OFFSET) / DIVISOR for index in range(width)]
        for width in widths
    ]


def step(state: list[float], inputs: list[list[float]]) -> list[float]:
    """One gated-delta update and its readout, in the order the reference applies them."""
    key_size = KEY_HEADS * KEY_DIM
    value_size = VALUE_HEADS * VALUE_DIM
    recurrent = list(state)
    out = [0.0] * value_size
    for head in range(VALUE_HEADS):
        key_head = head // (VALUE_HEADS // KEY_HEADS)
        q = inputs[0][key_head * KEY_DIM : (key_head + 1) * KEY_DIM]
        k = inputs[0][key_size + key_head * KEY_DIM : key_size + (key_head + 1) * KEY_DIM]
        v = inputs[0][2 * key_size + head * VALUE_DIM : 2 * key_size + (head + 1) * VALUE_DIM]
        qscale = math.sqrt(sum(value * value for value in q) + NORM_EPSILON) * math.sqrt(KEY_DIM)
        kscale = math.sqrt(sum(value * value for value in k) + NORM_EPSILON)
        q = [value / qscale for value in q]
        k = [value / kscale for value in k]
        beta = sigmoid(inputs[1][head])
        log_decay = -math.exp(inputs[3][head]) * softplus(inputs[2][head] + inputs[4][head])
        decay = math.exp(log_decay)
        base = head * KEY_DIM * VALUE_DIM
        for index in range(KEY_DIM * VALUE_DIM):
            recurrent[base + index] = f32(recurrent[base + index] * decay)
        for d in range(VALUE_DIM):
            predicted = sum(recurrent[base + i * VALUE_DIM + d] * k[i] for i in range(KEY_DIM))
            delta = (v[d] - predicted) * beta
            for i in range(KEY_DIM):
                recurrent[base + i * VALUE_DIM + d] = f32(
                    recurrent[base + i * VALUE_DIM + d] + f32(k[i] * delta)
                )
            out[head * VALUE_DIM + d] = f32(
                sum(recurrent[base + i * VALUE_DIM + d] * q[i] for i in range(KEY_DIM))
            )
    return recurrent, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    base = [0.0] * (VALUE_HEADS * KEY_DIM * VALUE_DIM)
    inputs = [declared_inputs(step) for step in range(STEPS)]

    states = []
    outputs = []
    state = list(base)
    for rows in inputs:
        state, out = step(state, rows)
        states.append(state)
        outputs.append(out)

    # R1's property: replaying the prefix from the base state reproduces the states, so a commit
    # can drop snapshots and replay instead.
    replayed = list(base)
    for rows in inputs:
        replayed, _ = step(replayed, rows)
    replay_matches = replayed == states[-1]

    golden = {
        "op": "gated-delta state step",
        "geometry": {
            "key_heads": KEY_HEADS,
            "value_heads": VALUE_HEADS,
            "key_dim": KEY_DIM,
            "value_dim": VALUE_DIM,
            "norm_epsilon": NORM_EPSILON,
        },
        "formula": {
            "inputs": (
                "row 0 is [q | k | v]; rows 1..4 are the per-value-head gate, softplus input and "
                "log-decay inputs"
            ),
            "normalise": "q /= sqrt(sum(q^2) + eps) * sqrt(key_dim); k /= sqrt(sum(k^2) + eps)",
            "decay": "state *= exp(-exp(row3) * softplus(row2 + row4))",
            "update": "delta = (v - k @ state) * sigmoid(row1); state += outer(k, delta)",
            "readout": "out = q @ state",
            "precision": (
                "f64 accumulation with the state rounded to f32 after each step, which is what the "
                "reference does"
            ),
        },
        "declared_inputs": {
            "formula": (
                f"row[width], step -> ((step * width + index) % {MODULUS} - {OFFSET}) / {DIVISOR}"
            ),
            "rows": inputs,
        },
        "base_state": base,
        "states": states,
        "outputs": outputs,
        "replay_property": {
            "claim": "replaying the recorded inputs from the base state reproduces the last state",
            "holds": replay_matches,
        },
        "tolerance": {
            "absolute": 1e-5,
            "relative": 1e-4,
            "reason": "the reference accumulates in f64 and keeps the state in f32",
        },
        "generator": {"script": "tools/fixtures/export-delta-state-golden.py"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(golden) + "\n")
    print(
        f"wrote {args.output}: {len(states)} steps, {len(base)} state values, "
        f"replay holds: {replay_matches}"
    )


if __name__ == "__main__":
    main()
