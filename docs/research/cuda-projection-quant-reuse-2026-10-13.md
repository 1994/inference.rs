# CUDA 相邻共享输入投影量化复用（2026-10-13）

在已提交的 SwiGLU gate/up 复用基础上，把复用扩展到读取相同输入的连续 Linear。
此修订保留原量化 kernel、GEMM、F32 状态、FP8 KV、缩放顺序与硬件调优选择。
默认注意力仍为 SIMT。当前完成整模型逐位验证，任务质量与正式服务性能仍待验收；
没有把减少 kernel 数量写成端到端提速。

## 适用条件

每个投影必须读取同一非 constant 输入，量化契约一致，输入及输出非空，
没有 state、输入输出别名或重复输出；输出宽度可以不同。
FP8 per-token 量化只依赖输入与 K，不依赖输出 N；NVFP4 scale bits 仍须相同。
block-scaled FP8、不同输入/契约、任何插入的其他节点仍使用原路径。

连续三个及更多投影也可复用。每个后续许可只在上一投影记录成功后生效，
运行时继续核对实际输入地址、完整形状/步长、M/K、编码和 codes/scales 地址。
跳过中间消费者、重复消费者、任何缓冲区变化、未知 view、失败或新 capture 均撤销许可。
每张图仍保留第一次量化，每次 replay 都处理新输入，不复用上次 replay 的数据。
规划器一次建立 tensor 尺寸索引；只有使用整张量的 Flat capture 才建立该规划。
pooled 与完整矩阵 prefill 仍走自己的 capture 入口；逐 lane 路径保留原量化。

Qwen3.8-27B-NVFP4 当前图中 256 个 Linear 节点满足条件：64 对 MLP、
48 对 GDN qkv/z、16 对注意力 k/v。每份图可省 128 次量化，上一版省 64 次。
注意力 q 后的 Split 中断连续投影，因此 q 的量化仍保留。
三消费者规则通过独立许可状态测试，但本模型的上述真实分组均为两消费者。

## 验证与身份

CUDA release 严格 clippy、布局、策略检查通过；默认 CUDA 测试 68 通过、79 ignored。
同一二进制/权重中开启与关闭全部复用，width=64/256、实际 511/51/64/255/256/1
长度、非零 base、tail/reset、forced/greedy decode、verification 部分提交与回滚、
四路 pooled 的 inactive/reactivate/rebind，完整 hidden/词表 logits/F32 state/FP8 KV 逐位相同。
每个 width 的四张 capture 合计由 2804 次原量化变为 2292 次量化 + 512 次复用；
上一版为 2548 次量化 + 256 次复用。

最终逐位验证使用原冻结缓存 SHA256：
`62b6532c8cb44b70709f1931fca41cb8d91da22d846b83cc7b11c1a93c4b3c27`。
首次检查误写为未识别的 `INFER_CUDA_TUNING_CACHE`，实际使用已有默认缓存，
保留为诊断；修正为 `INFER_CUDA_TUNING_TABLE` 后使用同一冻结二进制重新完整检查通过。
没有用那次诊断替代冻结缓存验收。

开发、构建与 GPU 验证均在 45；所有 heavy run 经 safe-run，32 GiB host memory、
swap=0、30 分钟限时，不终止其他 GPU 使用者。
证据：`/home/r/benchmarks/inference.rs/quant-projection-runs-2026-10-13/`。
`compiled-source-v1.json`、`source-v1.patch`、`unit-v1.log`、
`model-bitwise-v1.log`、`frozen-model-policy-v1.json`、`model-bitwise-frozen-v1.log`、
`cli-build-v1.json` 保留源码、构建与完整验证身份。

测试二进制 SHA256：
`fa1cdcfdf8481f1edd4138623aecaf0b263a2d256591424b952b675e02fe072f`。
候选 CLI SHA256：
`8154e417363297ece0d095e9929685b8a37b29c2a6459198d608fe3a817af32a`。
后续任务质量使用同一冻结 256 题、原评分器及额外严格 token 流断言；
性能须在 CI 和相应数值/质量门槛通过后配对测量。
