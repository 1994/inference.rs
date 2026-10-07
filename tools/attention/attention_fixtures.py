#!/usr/bin/env python3
"""Model-independent SDPA fixtures with an explicit F64 oracle and RoPE regressions."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def packed(value, head):
    half = 1 << (head // 2 - 1).bit_length()
    value = value.permute(1, 0, 2).reshape(value.shape[1], value.shape[0], 2, head // 2)
    return F.pad(value, (0, half - head // 2)).cpu().contiguous()


def rotated(x, cosine, sine):
    low, high = x.chunk(2, dim=-1)
    return x * cosine + torch.cat((-high, low), dim=-1) * sine


def case(name, q, k, h, kh, d, mask="none", frame=None, rope=False, left=0, right=0):
    return {
        "id": name,
        "tokens": q,
        "kv_tokens": k,
        "heads": h,
        "kv_heads": kh,
        "head_dim": d,
        "frame_tokens": frame or k,
        "mask": mask,
        "query_start": k - q,
        "left": left,
        "right": right,
        "rope": rope,
        "elements": q * h * d,
    }


def fixtures():
    for head in (64, 72, 128):
        for tokens in (1, 31, 33, 65, 128, 512, 2048):
            heads = 2 if tokens < 128 else 16
            yield case(f"n{tokens}-d{head}", tokens, tokens, heads, heads, head)
    yield case("frames-tail", 65, 65, 2, 2, 72, "segments", 16, True)
    yield case("frames-long", 512, 512, 16, 16, 72, "segments", 128, True)
    yield case("cancellation", 32, 32, 2, 2, 64)
    yield case("cross-attention", 7, 33, 4, 4, 64)
    yield case("gqa-causal", 33, 65, 8, 2, 64, "causal")
    yield case("mqa-decode", 1, 257, 8, 1, 128, "causal")
    yield case("causal-prefill", 65, 65, 4, 4, 72, "causal")
    yield case("causal-empty-rows", 65, 33, 4, 4, 64, "causal")
    yield case("window-gqa", 33, 65, 4, 2, 64, "window", left=16, right=3)
    yield case("small-head", 33, 65, 4, 2, 32)


def make_case(index, shape):
    name, nq, nk = shape["id"], shape["tokens"], shape["kv_tokens"]
    h, kh, d = shape["heads"], shape["kv_heads"], shape["head_dim"]
    torch.manual_seed(index + 107)
    q = torch.randn(h, nq, d, device="cuda") * 0.5
    k, v = [torch.randn(kh, nk, d, device="cuda") * 0.5 for _ in range(2)]
    angles = (
        torch.arange(nq, device="cuda")[:, None]
        * torch.arange(d // 2, device="cuda")[None, :]
        * 0.017
    )
    cos, sin = [torch.cat((f(angles), f(angles)), -1)[None] for f in (torch.cos, torch.sin)]
    if name == "cancellation":
        q.zero_()
        k.zero_()
        v[:, ::2] = 1 + 1 / 4096
        v[:, 1::2] = -1
    qr, kr = q.double(), k.double()
    if shape["rope"]:
        qr, kr = [rotated(x, cos.double(), sin.double()) for x in (qr, kr)]
    kr = kr.repeat_interleave(h // kh, dim=0)
    vr = v.double().repeat_interleave(h // kh, dim=0)
    scores = (qr @ kr.transpose(-1, -2)) / d**0.5
    qi, ki = torch.arange(nq, device="cuda")[:, None], torch.arange(nk, device="cuda")[None, :]
    if shape["mask"] == "segments":
        allowed = qi // shape["frame_tokens"] == ki // shape["frame_tokens"]
    elif shape["mask"] == "causal":
        allowed = ki <= qi + shape["query_start"]
    elif shape["mask"] == "window":
        center = qi + shape["query_start"]
        allowed = (ki >= center - shape["left"]) & (ki <= center + shape["right"])
    else:
        allowed = torch.ones((nq, nk), device="cuda", dtype=torch.bool)
    scores.masked_fill_(~allowed, -torch.inf)
    probabilities = scores.softmax(-1).nan_to_num(nan=0.0)
    expected = (probabilities @ vr).float()
    values = {
        f"{name}/{key}": packed(x, d)
        for key, x in (("q", q), ("k", k), ("v", v), ("output", expected))
    }
    for key, x in (("cos", cos), ("sin", sin)):
        values[f"{name}/{key}"] = packed(x, d).squeeze(1)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tensors, cases = {}, list(fixtures())
    for index, shape in enumerate(cases):
        tensors.update(make_case(index, shape))
        print(shape["id"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    manifest = {
        "schema": 1,
        "contract": "sdpa-f32-v1",
        "scope": "case-defined-sdpa-pipeline-f32-output-cuda-graph",
        "fixture_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "oracle": "F64 matmul/mask/softmax/matmul; optional RoPE; empty rows zero; output F32",
        "cases": cases,
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
