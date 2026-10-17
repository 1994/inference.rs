#!/usr/bin/env python3
"""Golden fixture for the Rust video preprocessing path.

Runs the official `Qwen3VLVideoProcessor` on a deterministic clip and stores the source frames,
the resulting `pixel_values_videos` and `video_grid_thw`. The Rust test compares its own sampling,
resize, temporal padding and patch layout against this file. Deepstack is irrelevant here: this
covers preprocessing only.

Run:

    uv run --with "transformers>=5,<6" --with torch --with torchvision --with numpy \
        --with safetensors python3 tools/vision/qwen3vl_video_golden.py \
        --package /home/r/models/qwen3vl-2b \
        --out artifacts/vision/qwen3vl-2b-video-golden.safetensors
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor

# Frames are generated procedurally so the fixture carries its own source bytes.
FRAMES = 4
HEIGHT = 60
WIDTH = 100
FPS = 30.0


def synthetic_video() -> np.ndarray:
    frames = np.zeros((FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    for frame in range(FRAMES):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                frames[frame, y, x] = (
                    (y * 255 // HEIGHT + frame * 40) % 256,
                    (x * 255 // WIDTH + frame * 20) % 256,
                    128 - frame * 12,
                )
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.package)
    frames = synthetic_video()
    # The clip is already sampled: compare one preprocessing pass, not sampler timing.
    inputs = processor(
        videos=[frames],
        do_sample_frames=False,
        # Pin the uncapped behaviour the Rust processor implements; transformers 5.22 flips it.
        cap_pixels_per_frame=False,
        return_tensors="pt",
    )
    pixels = inputs["pixel_values_videos"].to(torch.float32).cpu()
    grid = inputs["video_grid_thw"].to(torch.int64).cpu()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "pixel_values": pixels,
            "grid_thw": grid,
            "frames": torch.from_numpy(frames.reshape(-1).copy()),
            "shape": torch.tensor([FRAMES, HEIGHT, WIDTH], dtype=torch.int64),
            "fps": torch.tensor([FPS], dtype=torch.float32),
        },
        str(args.out),
    )
    print({"out": str(args.out), "pixels": list(pixels.shape), "grid": grid.tolist()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
