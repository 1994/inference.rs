# Qwen hybrid host golden package

This small, untrained test package is exported by the official Hugging Face
Qwen3_5ForCausalLM implementation with seed 20261004, F32 CPU eager attention,
Transformers 5.18.0 and Torch 2.14.1. It contains a linear-attention layer and a
full-attention layer with GQA, QK normalization, partial RoPE and output gating.
Norm, decay and gate parameters are deliberately nontrivial.

model.safetensors contains actual generated weights. golden.json contains
independent full-sequence layer outputs, prefix logits/hidden states and cached greedy decode
tokens. These are numerical references for this architecture and configuration,
not Qwen3.8-27B quality or performance evidence.

The tiny WordLevel tokenizer is a serving fixture; the real Qwen3.8 tokenizer
is tested separately against pinned official assets.

Reproduce from the repository root:

    uv venv artifacts/golden-env --python 3.12
    uv pip install --python artifacts/golden-env/bin/python -r tools/fixtures/requirements.txt
    artifacts/golden-env/bin/python tools/fixtures/export-qwen-golden.py --output examples/qwen-hybrid-tiny

The Python dependencies are golden-export tools. The Rust executable reads the
saved package without Python or Torch.
