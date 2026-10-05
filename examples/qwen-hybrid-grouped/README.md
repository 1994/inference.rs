# Grouped hybrid golden package

An independent, untrained official Transformers Qwen3_5ForCausalLM fixture with
hidden size 16, GQA 4 query / 2 KV heads, head dimension 8, partial RoPE 0.25,
and linear-attention 2 key / 6 value heads with dimension 4. The linear value
to key head ratio 3 exercises the same grouping ratio as Qwen3.8-27B.

Weights, prefix layer/hidden/logit references, cached greedy trajectory and
separate readout references are generated independently of the Rust executor.
Versions and seed match qwen-hybrid-tiny. This is numerical architecture
coverage, not evidence for trained model quality or GPU performance.

    artifacts/golden-env/bin/python tools/fixtures/export-qwen-golden.py \
      --grouped --output examples/qwen-hybrid-grouped
    target/release/infer verify --package examples/qwen-hybrid-grouped \
      --golden examples/qwen-hybrid-grouped/golden.json --atol 0.000002 --rtol 0.00002
