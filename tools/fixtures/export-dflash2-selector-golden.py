"""Cold-path DFlash2 selector golden: the draft's candidate scoring on recorded inputs.

vLLM's `_score_edges` shows what the released checkpoint computes: the draft hidden
state is projected to the selector rank by `hidden_projection` (no bias), each
candidate contributes its unary logit, and the low-rank term is a bilinear form
between the predecessor's codebook row (scaled by the projected hidden) and the
successor's row. This exporter records the score matrix for a declared set of
candidates, including a deliberate tie in the unary logits, so a draft
implementation has something bit-level to compare against and the tie-break stays
visible.

The weights are not in the repository, so the golden records the model identity: the
configuration digest registered with the fixture, the tensor names and shapes,
torch's version and the arithmetic. Re-run with
`--model <path to Qwen3.8-27B-DFlash2>`.
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
# Candidate ids and unary logits are generated from `selector_top_k`, and the first two logits are
# equal so the tie-break is a recorded property rather than a footnote.
CANDIDATE_STRIDE = 7
CANDIDATE_BASE = 11
ANCHOR_TOKEN_ID = 7
TIE_PAIR = (1.0, 1.0)


def candidate_ids(top_k: int) -> list[int]:
    return [CANDIDATE_BASE + CANDIDATE_STRIDE * index for index in range(top_k)]


def unary_logits(top_k: int) -> list[float]:
    values = [
        TIE_PAIR[0] if index == 0 else TIE_PAIR[1] if index == 1 else -0.5 + 0.25 * index
        for index in range(top_k)
    ]
    return values


def deterministic_hidden(channels: int) -> torch.Tensor:
    index = torch.arange(channels, dtype=torch.int64)
    return ((index % MODULUS) - OFFSET).to(torch.float32) / DIVISOR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("examples/qwen3.8-27b-dflash2"))
    args = parser.parse_args()

    config_path = args.fixture / "config.json"
    config = json.loads(config_path.read_text())
    hidden_size = config["hidden_size"]
    rank = config["dflash_config"]["selector_rank"]
    top_k = config["dflash_config"]["selector_top_k"]
    ids = candidate_ids(top_k)
    logits = unary_logits(top_k)
    candidates = torch.tensor(ids, dtype=torch.int64)

    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as weights:
        projection = weights.get_tensor("candidate_selector.hidden_projection.weight")
        predecessor_rows = weights.get_slice("candidate_selector.predecessor_codebook")
        successor_rows = weights.get_slice("candidate_selector.successor_codebook")
        if tuple(projection.shape) != (rank, hidden_size):
            raise ValueError(f"unexpected hidden_projection shape {tuple(projection.shape)}")

        hidden = torch.nn.functional.linear(
            deterministic_hidden(hidden_size).reshape(1, -1), projection.to(torch.float32)
        )[0]
        # The predecessor of the first candidate is the anchor, and each later candidate follows the
        # one before it, exactly as the reference builds the shifted list.
        predecessor_ids = torch.cat(
            (torch.tensor([ANCHOR_TOKEN_ID], dtype=torch.int64), candidates[:-1])
        )
        predecessors = predecessor_rows[predecessor_ids].to(torch.float32)
        successors = successor_rows[candidates].to(torch.float32)
        unary = torch.tensor(logits, dtype=torch.float32)
        scores = unary[None, :] + (predecessors * hidden[None, :]) @ successors.T

    golden = {
        "model": {
            "repository": "z-lab/Qwen3.8-27B-DFlash2",
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "tensors": {
                "candidate_selector.hidden_projection.weight": list(projection.shape),
                "candidate_selector.predecessor_codebook": list(predecessor_rows.get_shape()),
                "candidate_selector.successor_codebook": list(successor_rows.get_shape()),
            },
            "dtype": "BF16",
        },
        "rank": rank,
        "top_k": top_k,
        "formula": {
            "hidden": (
                f"hidden_states[c] = ((c % {MODULUS}) - {OFFSET}) / {DIVISOR}; "
                "hidden = hidden_projection @ hidden_states in float32, no bias"
            ),
            "score": (
                "score[p, c] = unary_logits[c] + sum_r "
                "predecessor_codebook[predecessor[p], r] * hidden[r] * successor_codebook[c, r]"
            ),
            "predecessors": "predecessor[0] = anchor, predecessor[p] = candidate[p - 1]",
        },
        "inputs": {
            "anchor_token_id": ANCHOR_TOKEN_ID,
            "candidate_ids": ids,
            "unary_logits": logits,
            "tie": ("unary_logits[0] == unary_logits[1]; the scores keep the tie visible"),
        },
        "tolerance": {
            "absolute": 2e-2,
            "relative": 2e-2,
            "reason": "bf16 weights, f32 accumulation",
        },
        "generator": {
            "script": "tools/fixtures/export-dflash2-selector-golden.py",
            "torch": torch.__version__,
        },
        "scores": [[float(value) for value in row] for row in scores],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(golden) + "\n")
    print(f"wrote {args.output} ({scores.shape[0]}x{scores.shape[1]} scores)")


if __name__ == "__main__":
    main()
