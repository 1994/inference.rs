#!/usr/bin/env python3
"""Create attention fixtures and measure PyTorch SDPA with the same inputs and CUDA graph scope.

Follow with the ignored Rust attention_benchmark_against_sdpa test, setting
INFER_ATTENTION_BENCH to the output safetensors path. F32 is the accuracy reference;
BF16 SDPA is also timed but has a different precision contract.
"""

import argparse
import functools
import json
import pathlib
import statistics

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def rotated(x, cosine, sine):
    low, high = x.chunk(2, dim=-1)
    return x * cosine + torch.cat((-high, low), dim=-1) * sine


def graph_time(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    for _ in range(100):
        graph.replay()
    samples = []
    for _ in range(60):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}, output


def packed(value, head):
    # Match the resident half-head padding, including the 72 -> 2*64 case.
    half = 1 << (head // 2 - 1).bit_length()
    value = value.permute(1, 0, 2).reshape(value.shape[1], value.shape[0], 2, head // 2)
    return F.pad(value, (0, half - head // 2)).cpu().contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tensors, results = {}, []
    for head in (64, 72):
        for tokens in (128, 512, 2048):
            index = len(results)
            torch.manual_seed(index)
            q, k, v = [torch.randn(16, tokens, head, device="cuda") * 0.5 for _ in range(3)]
            angles = (
                torch.arange(tokens, device="cuda")[:, None]
                * torch.arange(head // 2, device="cuda")[None, :]
                * 0.017
            )
            cosine, sine = [torch.cat((f(angles), f(angles)), -1) for f in (torch.cos, torch.sin)]
            cosine, sine = cosine[None], sine[None]

            def execute(dtype, q=q, k=k, v=v, cosine=cosine, sine=sine):
                return F.scaled_dot_product_attention(
                    rotated(q, cosine, sine).to(dtype).unsqueeze(0),
                    rotated(k, cosine, sine).to(dtype).unsqueeze(0),
                    v.to(dtype).unsqueeze(0),
                ).squeeze(0)

            f32, output = graph_time(functools.partial(execute, torch.float32))
            bf16, _ = graph_time(functools.partial(execute, torch.bfloat16))
            prefix = f"c{index}"
            tensors.update(
                {
                    f"{prefix}/{name}": packed(value, head)
                    for name, value in [("q", q), ("k", k), ("v", v), ("output", output)]
                }
            )
            half = 1 << (head // 2 - 1).bit_length()
            for name, value in [("cos", cosine), ("sin", sine)]:
                tensors[f"{prefix}/{name}"] = (
                    F.pad(value.reshape(tokens, 2, head // 2), (0, half - head // 2))
                    .cpu()
                    .contiguous()
                )
            tensors[f"{prefix}/head_dim"] = torch.zeros(head)
            results.append(
                {
                    "tokens": tokens,
                    "heads": 16,
                    "head_dim": head,
                    "sdpa_f32": f32,
                    "sdpa_bf16": bf16,
                }
            )
            print(tokens, head, f32["median_ms"], bf16["median_ms"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "tf32": False,
                "scope": "warm CUDA graph: RoPE + SDPA, plus BF16 conversion in BF16 mode",
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
