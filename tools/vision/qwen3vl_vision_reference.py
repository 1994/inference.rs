#!/usr/bin/env python3
"""Qwen3.5-VL vision-tower reference and golden generator.

Runs the official `transformers` `Qwen3VLVisionModel` on the real package weights and
writes golden tensors for the Rust/cuTile implementation to match. The golden is an
`artifacts/` product, not a repository asset: generate it on the machine that owns the
package.

Run:

    uv run --with "transformers>=5,<6" --with torch --with safetensors \
        python3 tools/vision/qwen3vl_vision_reference.py \
        --package /home/r/models/Qwen3.8-27B-NVFP4 \
        --out artifacts/vision/qwen3vl-vision-golden.safetensors

Semantics ported by the Rust implementation (authoritative source: the installed
`transformers/models/qwen3_vl/modeling_qwen3_vl.py` plus `transformers/vision_utils.py`):

1. Patch order is spatial-merge-block order: patch `within` decodes as
   `(block_row, block_col, in_row, in_col)` with `row = block_row*m + in_row`,
   `col = block_col*m + in_col`, `m = spatial_merge_size`. The merger therefore only
   has to `view(-1, m*m*hidden)`.
2. `patch_embed` is `Conv3d(3, hidden, kernel=(temporal, patch, patch), stride=same)`
   over a flattened `[C, temporal, patch, patch]` patch vector.
3. `pos_embed` is a learned `(num_grid_per_side^2, hidden)` table bilinearly
   interpolated to each grid with `align_corners=True`, `padding="border"`, evaluated as
   four gathered taps with outer-product weights.
4. Block: `x += attn(norm1(x))`, `x += mlp(norm2(x))`; LayerNorm with bias, eps=1e-6;
   MLP is `gelu_pytorch_tanh`; QKV has bias; attention is packed per image
   (`cu_seqlens`), non-causal, `scaling = head_dim**-0.5`.
5. RoPE is axial 2D over the full head: `inv_freq` covers `head_dim//2` with stride 2,
   `cos/sin = cat([h, w], -1)` then `cat([hw, hw], -1)`, and `rotate_half` splits at
   `head_dim//2`.
6. `merger`: LayerNorm (eps=1e-6) over `hidden`, then `view(-1, m*m*hidden)`,
   `linear_fc1`, **exact erf GELU** (`nn.GELU()`), `linear_fc2` to `out_hidden_size`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3VLVisionConfig, Qwen3VLVisionModel
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

# Cases: (grid_thw, seed). Grids are patch-grid sizes; both axes must be multiples of
# the merge size. Three shapes exercise raster/merge ordering, non-square grids and the
# temporal dimension.
CASES = [
    ((1, 4, 4), 0),
    ((1, 4, 6), 1),
    ((2, 4, 4), 2),
    # Longer grids cross the 32-token attention block boundary, which the first three do not.
    ((1, 8, 8), 3),
    ((1, 16, 16), 4),
]


def visual_state_dict(package: pathlib.Path) -> dict[str, torch.Tensor]:
    """Vision tensors of a package, sharded or single-file."""
    index = package / "model.safetensors.index.json"
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        shards = sorted(
            {shard for name, shard in weight_map.items() if name.startswith("model.visual.")}
        )
    else:
        shards = ["model.safetensors"]
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(package / shard, framework="pt") as handle:
            for name in list(handle.keys()):
                if name.startswith("model.visual."):
                    state[name.removeprefix("model.visual.")] = handle.get_tensor(name)
    return state


def recorder(captured: dict[str, torch.Tensor], key: str):
    """Forward hook that stores a module's output under `key`."""

    def hook(_module, _inputs, output):
        captured[key] = output.detach().clone()

    return hook


def input_recorder(captured: dict[str, torch.Tensor], key: str):
    """Forward-pre hook that stores a module's first input under `key`."""

    def hook(_module, inputs):
        captured[key] = inputs[0].detach().clone()

    return hook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--trace-blocks", action="store_true")
    args = parser.parse_args()

    config = json.loads((args.package / "config.json").read_text())
    vision = Qwen3VLVisionConfig(**config["vision_config"])
    model = Qwen3VLVisionModel(vision)
    state = visual_state_dict(args.package)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys, result.missing_keys
    assert not result.unexpected_keys, result.unexpected_keys
    # F32 is the default accuracy oracle; BF16 is a separate rounding diagnostic.
    # Disable TF32 for both projections and Conv3d so CUDA F32 does not silently
    # introduce another operand precision into that comparison.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.to(device=args.device, dtype=getattr(torch, args.dtype)).eval()

    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {
        "source": "transformers Qwen3VLVisionModel",
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "tf32": False,
        "device": args.device,
        "vision_config": {
            key: getattr(vision, key)
            for key in (
                "depth",
                "hidden_size",
                "intermediate_size",
                "num_heads",
                "in_channels",
                "patch_size",
                "spatial_merge_size",
                "temporal_patch_size",
                "out_hidden_size",
                "num_position_embeddings",
                "hidden_act",
            )
        },
        "rope_parameters": vision.rope_parameters,
        "cases": [],
    }

    for index, (grid, seed) in enumerate(CASES):
        t, h, w = grid
        patches = t * h * w
        width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        generator = torch.Generator().manual_seed(seed)
        pixel_values = torch.randn(patches, width, generator=generator) * 0.5
        captured: dict[str, torch.Tensor] = {}
        prefix = f"c{index}"
        handles = [
            model.patch_embed.register_forward_hook(recorder(captured, f"{prefix}/patch_embed")),
            # The rotary module's input is the hidden state *after* the position add.
            model.rotary_pos_emb.register_forward_pre_hook(
                input_recorder(captured, f"{prefix}/post_pos")
            ),
            model.blocks[0].register_forward_hook(recorder(captured, f"{prefix}/block0")),
            model.merger.register_forward_hook(recorder(captured, f"{prefix}/pooler")),
        ]
        if args.trace_blocks:
            for layer, block in enumerate(model.blocks):
                handles.append(
                    block.register_forward_hook(recorder(captured, f"{prefix}/blocks/{layer}"))
                )
        # Block-0 internals for every case, so each op can be validated on its own.
        block = model.blocks[0]
        handles.extend(
            [
                block.norm1.register_forward_hook(recorder(captured, f"{prefix}/block0_norm1")),
                block.attn.qkv.register_forward_hook(recorder(captured, f"{prefix}/block0_qkv")),
                block.attn.proj.register_forward_pre_hook(
                    input_recorder(captured, f"{prefix}/block0_attn")
                ),
                block.attn.proj.register_forward_hook(
                    recorder(captured, f"{prefix}/block0_attn_proj")
                ),
                block.norm2.register_forward_hook(recorder(captured, f"{prefix}/block0_norm2")),
                block.mlp.linear_fc1.register_forward_hook(
                    recorder(captured, f"{prefix}/block0_fc1")
                ),
                block.mlp.linear_fc2.register_forward_hook(
                    recorder(captured, f"{prefix}/block0_fc2")
                ),
            ]
        )
        grid_thw = torch.tensor([grid], dtype=torch.long)
        with torch.no_grad():
            output = model(pixel_values.to(args.device), grid_thw=grid_thw.to(args.device))
        for handle in handles:
            handle.remove()

        tensors[f"c{index}/pixel_values"] = pixel_values
        tensors[f"c{index}/patch_embed"] = captured[f"c{index}/patch_embed"]
        tensors[f"c{index}/post_pos"] = captured[f"c{index}/post_pos"]
        tensors[f"c{index}/block0"] = captured[f"c{index}/block0"]
        tensors[f"c{index}/last_hidden_state"] = output.last_hidden_state.detach()
        tensors[f"c{index}/pooler_output"] = output.pooler_output.detach()
        tensors.update(captured)
        tensors[f"c{index}/grid_thw"] = grid_thw

        if index == 0:
            indices, weights = get_vision_interpolation_indices_and_weights(
                grid_thw,
                num_grid_per_side=model.num_grid_per_side,
                mode="bilinear",
                align_corners=True,
                spatial_merge_size=vision.spatial_merge_size,
            )
            position_ids = get_vision_position_ids(grid_thw, vision.spatial_merge_size)
            cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, vision)
            tensors["c0/interp_indices"] = indices
            tensors["c0/interp_weights"] = weights
            tensors["c0/position_ids"] = position_ids
            tensors["c0/cu_seqlens"] = cu_seqlens
            meta["cu_seqlens"] = cu_seqlens.tolist()
            meta["max_seqlen"] = max_seqlen

        meta["cases"].append(
            {
                "index": index,
                "grid_thw": list(grid),
                "seed": seed,
                "patches": patches,
                "patch_elements": width,
                "merged_tokens": patches // (vision.spatial_merge_size**2),
            }
        )
        merged = tuple(output.pooler_output.shape)
        print(f"case {index}: grid={grid} patches={patches} pooler={merged}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: (value.float() if value.is_floating_point() else value).cpu().contiguous()
            for name, value in tensors.items()
        },
        args.out,
    )
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes) and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
