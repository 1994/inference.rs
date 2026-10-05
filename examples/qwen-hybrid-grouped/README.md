# qwen-hybrid-grouped：分组比例样例包

这是一个独立导出的未训练官方 `Qwen3_5ForCausalLM` 夹具：hidden size 16、GQA 4 个 query / 2 个 KV head、head dimension 8、partial RoPE 0.25，linear-attention 为 2 个 key head / 6 个 value head、维度 4。linear value 与 key 的 head 比例 3 复现了 Qwen3.8-27B 的分组比例。

权重、prefix layer/hidden/logits 参考、cached greedy 轨迹和单独的 readout 参考都由官方实现独立生成，不依赖 Rust 执行器；版本与 seed 与 [qwen-hybrid-tiny](../qwen-hybrid-tiny/README.md) 一致。它提供的是数值架构覆盖，不代表训练后模型质量或 GPU 性能。

## 重新导出与验证

```sh
artifacts/golden-env/bin/python tools/fixtures/export-qwen-golden.py \
  --grouped --output examples/qwen-hybrid-grouped

target/release/infer verify --package examples/qwen-hybrid-grouped \
  --golden examples/qwen-hybrid-grouped/golden.json --atol 0.000002 --rtol 0.00002
```

导出环境见 [tiny 样例说明](../qwen-hybrid-tiny/README.md)。
