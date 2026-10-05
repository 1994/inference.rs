# 模型加载与 forward

## 加载链与预算

```text
QwenPackage::open → WeightLoadPlan → load_weights / WeightTarget
    → backend assemble / compile → Engine admission / schedule
    → prefill chunks / incremental decode → workload postprocess
```

[公共加载器](../../crates/model/package/src/loader.rs)校验 config、shard index、Safetensors headers、参数绑定与 shape；WeightLoadPlan 描述源/目标精度、驻留和暂存预算。WeightTarget 拥有设备分配与上传同步，公共 loader 不携带具体设备 handle。

源格式 F32/BF16/F16；Metal 保持原始权重字节，shader 转 F32，activation/累加/KV/recurrent 为 F32。默认上传 chunk 为 4 MiB；F32 展开路径同时计源与转换 chunk，不先整包展开。payload hash 与 chunk 大小无关，fresh 共享不可变权重、重建可变资源。

加载前校验 weights、chunk scratch、cache/probe reserve、单 buffer 与设备索引限制；请求私有状态和页池在 admission 另报价。I/O、非有限值、分配或上传失败释放已有 handles。upload 返回前必须消费借用切片，异步 DMA adapter 自持有预算内 staging 与完成确认。

```sh
target/release/infer inspect-package --package /path/to/hf-package \
  --device-memory-mib 32768
target/release/infer inspect-package --package /path/to/hf-package --weights-f32
```

inspect-package 的驻留预检只覆盖权重，完整预算以 backend preflight/admission 为准。

## 包格式与输入

包至少包含 config.json、model.safetensors，或索引及其全部 shards。tokenizer.json、tokenizer_config.json / chat_template.jinja 启用文本；prompt/messages 互斥，模板支持 enable_thinking，结果携带输入/模板 fingerprint。

可选 readouts.safetensors 启用 projection：

| Tensor | Shape |
|---|---|
| embedding.weight | [embedding_dim, hidden_size] |
| rank.weight / rank.bias | [1, hidden_size] / [1] |
| decision.weight / decision.bias | [decision_channels, hidden_size] / [decision_channels] |

Decision 选项使用 HeadChannel 绑定投影通道；当前额外 embedding/rank/decision projection 由 CPU workload postprocess 执行，设备提供所需 hidden。

[Qwen3.8-27B 示例](../../examples/qwen3.8-27b)仅有 config/index，没有完整 shards；inspect-model 检查元数据并保持 execution_supported=false。索引记录 BF16 权重 55,562,855,904 bytes，超过 32 GiB，5090 执行需要量化方案与独立验收。

## Prefill 与 decode

编译图给出单 token activation shape，Metal 扩展成连续 chunk rows，按 lifetime slot 复用 scratch。stateless 节点处理全部 rows，attention 先写 KV 再按绝对位置读取并应用因果掩码；conv/delta 在独占 channel/head 内顺序推进 rows。

算子覆盖 embedding、RMSNorm/QK norm、split、partial RoPE、GQA/attention gate、causal conv、GatedDeltaNet、gated norm、SwiGLU、残差与 LM head。decode 的矩阵行由 SIMD group 归约，prefill 使用 4×4 shared tile，并保留累加语义。

节点和不可变 buffer 在加载期预绑定。LM head 只计算最终输出或 prefix 边界最后一行；Generate hidden 仅保留一行，Full readout 保留历史并报价。prefix 启用时 chunk 在页边界截断，KV 与 recurrent snapshot 必须对齐。

每 sequence 独立状态；当前请求在同一 command buffer 内依次编码，共享一套 scratch，compute depth=1。提交前校验全部 token/frontier/页需求，失败回滚页表与 lease，copy pin 留到 GPU fence。具体事务见[KV Manager](../architecture/kv-manager.md)。

```sh
target/release/infer --backend metal --prefill-chunk-tokens 32 --upload-staging-mib 4 \
  run --package examples/qwen-hybrid-tiny --requests examples/requests.json
```

profile 区分 weight load、chunk/scratch、GPU command 与 CPU op encoding。独立 golden 和 chunk/prefix/checkpoint 验证见[验证记录](../validation/index.md)，生产 kernels 与完整模型缺口见[状态表](../design/status.md)。
