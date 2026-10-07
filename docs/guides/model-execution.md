# 模型加载与执行

## 加载链

```text
ModelPackage::open → WeightLoadPlan → load_weights / WeightTarget
    → backend assemble / compile → Engine admission / schedule
    → prefill chunks / incremental decode → workload postprocess
```

公共加载器校验 config、shard index、Safetensors header、参数绑定与 shape，并生成描述源/目标精度、驻留和暂存预算的 `WeightLoadPlan`。`WeightTarget` 拥有设备分配与上传同步；公共加载器不持有具体设备 handle。

```sh
target/release/infer inspect-package --package /path/to/hf-package \
  --gpu-memory-utilization 0.9
target/release/infer inspect-package --package /path/to/hf-package --weights-f32
```

`inspect-package` 的驻留预检只覆盖权重，完整预算以 backend preflight / admission 为准。

## 精度与内存

- 源格式支持 F32 / BF16 / F16；Metal 保持原始权重字节，在 shader 中转换为 F32。
- activation、累加、KV 与 recurrent 状态使用 F32。
- 默认上传 chunk 为 4 MiB；F32 展开路径同时计算源与转换后 chunk，不会先整包展开。
- payload hash 与 chunk 大小无关。不可变权重可共享，可变资源每次重建。
- 加载前校验权重、chunk scratch、cache/probe reserve、单 buffer 与设备索引限制；请求私有状态和页池在 admission 阶段另行报价。
- I/O、非有限值、分配或上传失败时释放已创建的 handle。异步 DMA adapter 自带 staging 预算与完成确认。

## 包格式

模型包至少包含 `config.json` 和 `model.safetensors`，或索引文件及其全部 shards。`tokenizer.json` 与 `tokenizer_config.json` / `chat_template.jinja` 启用文本接口；`--text` 与 `--messages` 互斥，模板支持 `enable_thinking`，结果携带输入与模板 fingerprint。

可选的 `readouts.safetensors` 启用 projection：

| Tensor | Shape |
|---|---|
| `embedding.weight` | `[embedding_dim, hidden_size]` |
| `rank.weight` / `rank.bias` | `[1, hidden_size]` / `[1]` |
| `decision.weight` / `decision.bias` | `[decision_channels, hidden_size]` / `[decision_channels]` |

Decision 选项通过 HeadChannel 绑定投影通道。额外 embedding / rank / decision projection 由 CPU workload postprocess 执行，设备只需提供对应 hidden。

[Qwen3.8-27B 示例](../../examples/qwen3.8-27b) 只有 config 与索引，没有完整 shards；`inspect-model` 会检查元数据并保持 `execution_supported=false`。索引记录 BF16 权重约 55.6 GB，超过 32 GiB，在 5090 上执行需要量化方案与独立验收。

## Prefill 与 decode

- 编译图给出单 token activation shape，Metal 将其扩展为连续 chunk rows，并按 lifetime slot 复用 scratch。
- stateless 节点一次处理全部 rows；attention 先写 KV，再按绝对位置读取并应用因果掩码；conv / delta 在独占 channel/head 内顺序推进 rows。
- 算子覆盖 embedding、RMSNorm / QK norm、split、partial RoPE、GQA / attention gate、causal conv、GatedDeltaNet、gated norm、SwiGLU、残差与 LM head。
- decode 的矩阵行由 SIMD group 归约，prefill 使用 4×4 shared tile，两者保持相同累加语义。
- 节点与不可变 buffer 在加载期预绑定。LM head 只计算最终输出或 prefix 边界最后一行；Generate 只保留一行 hidden，Full readout 保留历史并单独报价。
- prefix 启用时，chunk 在页边界截断，KV 与 recurrent snapshot 必须对齐。
- 每个 sequence 独立状态；当前实现将同一请求的步骤编码在同一 command buffer 内，共享一套 scratch，compute depth = 1。
- 提交前校验全部 token / frontier / 页需求；失败时回滚页表与 lease，copy pin 保留到 GPU fence。事务细节见 [KV Manager](../architecture/kv-manager.md)。

```sh
target/release/infer --backend metal --max-num-batched-tokens 32 --upload-staging-mib 4 \
  run --package examples/qwen-hybrid-tiny --requests examples/requests.json
```

`profile` 区分权重加载、chunk/scratch、GPU command 与 CPU op encoding 时间。数值验证范围见[验证记录](../validation/index.md)，生产 kernel 与完整模型缺口见[实现状态](../design/status.md)。

### Memory-mapped weight loading

Safetensors shards use read-only memory maps. Opening a package validates the
bounded headers; payload pages are read on demand. Native chunk uploads borrow
mapped bytes directly, and CUDA quantized loading uses borrowed tensor views
before conversion into the device upload buffer. The owned `read_bytes` and
`QuantizedPackage::read` APIs remain available for callers that need a copy.
Staging budgets still conservatively count the input bytes as well as conversion
buffers; mapped pages also contribute to host memory usage when touched.

Keep weight files immutable while a package is open. Publish replacements by
renaming new files, never by overwriting or truncating mapped files in place.
Mapping removes explicit read buffers and copies, but does not eliminate disk
I/O, numeric validation, format conversion, hashing, or host-to-device transfers.
Measure full model readiness with cold and warm filesystem caches separately.
