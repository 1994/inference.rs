#!/usr/bin/env python3
"""Reference image preprocessing for the vision tower.

Runs the official `Qwen3VLProcessor` image processor on one deterministic synthetic image and
writes the resulting `pixel_values`, `image_grid_thw` and source bytes as a golden for the Rust
implementation to match. The golden is an `artifacts/` product, not a repository asset.

Run:

    uv run --with "transformers>=5,<6" --with torch --with torchvision --with numpy \
        --with safetensors python3 tools/vision/qwen3vl_image_reference.py \
        --package /home/r/models/Qwen3.8-27B-NVFP4 \
        --out artifacts/vision/qwen3vl-image-golden.safetensors
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor

# Deterministic source image: a smooth gradient plus blocks, so resampling differences show up.
HEIGHT = 60
WIDTH = 100
SEED = 7


def synthetic_image() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    rows = np.linspace(0.0, 1.0, HEIGHT)[:, None]
    columns = np.linspace(0.0, 1.0, WIDTH)[None, :]
    base = np.stack(
        [rows * np.ones_like(columns), columns * np.ones_like(rows), (rows + columns) / 2], -1
    )
    blocks = (rng.random((HEIGHT, WIDTH, 3)) > 0.85) * 0.4
    return np.clip((base + blocks) * 255.0, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.package)
    image = synthetic_image()
    features = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = features["pixel_values"].to(torch.float32)
    grid = features["image_grid_thw"]
    image_processor = processor.image_processor
    tensors = {
        "pixel_values": pixel_values,
        "image_grid_thw": grid.to(torch.int64),
        "image": torch.from_numpy(np.array([HEIGHT, WIDTH], dtype=np.int64)),
        "image_pixels": torch.from_numpy(image.reshape(-1).copy()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    meta = {
        "source": type(image_processor).__name__,
        "resample": int(image_processor.resample),
        "patch_size": image_processor.patch_size,
        "merge_size": image_processor.merge_size,
        "temporal_patch_size": image_processor.temporal_patch_size,
        "size": dict(image_processor.size),
        "image_mean": list(image_processor.image_mean),
        "image_std": list(image_processor.image_std),
        "grid_thw": grid.tolist(),
        "pixel_values_shape": list(pixel_values.shape),
        "image": [HEIGHT, WIDTH],
        "seed": SEED,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out} {tuple(pixel_values.shape)} grid={grid.tolist()} {meta['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
