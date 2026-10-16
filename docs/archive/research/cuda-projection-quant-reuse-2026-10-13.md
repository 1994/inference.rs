# CUDA 相邻共享输入投影量化复用（2026-10-13）

在已提交的 SwiGLU gate/up 复用基础上，把复用扩展到读取相同输入的连续 Linear。
此修订保留原量化 kernel、GEMM、F32 状态、FP8 KV、缩放顺序与硬件调优选择。
默认注意力仍为 SIMT。当前完成整模型逐位验证和原 256 题质量评分；服务 token 一致性与性能仍待验收；
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

## 原任务质量与对照修正

提交 `8f9cce72c7142e2822de7b21e7e29c7dc09d609f` 的
[10 项代码 CI](https://github.com/1994/inference.rs/actions/runs/38384675222) 全部通过，
[文档 CI](https://github.com/1994/inference.rs/actions/runs/38384675261) 通过。

首次完整配对 v1 错误地为两版设置 `INFER_CUDA_TILED_ATTENTION=0`。
旧版 4fafeb7 以变量存在作为启用条件，因此实际运行了旧三项补偿 tiled 注意力；
候选只接受 1/true，实际运行 SIMT。该配对 **路径不一致，版本对照无效**。
v1 的 242→245/256、2 题退步、232 条不同 token 流和原门槛失败全部保留，
不能据此接受或拒绝本次复用质量、不能推断耗时差距。
`quality-v1-invalid-comparison.json` 记录原因和修正边界。

修正后的 v2 在两边隔离环境中都移除该变量，只修正错误环境，不改变
二进制、原缓存、题目、评分器、生成预算或并发；重新完整运行全部 256 题。
结果 **244→245/256**，逐题退步 0、提升 1；原 95% 单侧退步率上界
1.1634% 低于原 2% 门槛，`quality_accepted=true`。
zero-shot 为 118→119/128，eight-shot 均为 126/128；两版均有 7 条达到长度限制。
全部请求、完整 token 流和原评分在 `quality-v2/`。

46 条完整 token 流不同，额外严格断言失败。
原质量评分通过与服务流严格一致性失败分别保留，未删除断言或改评分。
尚未据此授予全服务性能验收。

## 同一二进制的服务波动定位

同一候选二进制、同一原缓存与原前 16 个输入、output=512、并发 4，
预声明重复两次；只打开现有 host execution profile，没有增加 GPU 观察核。
两次均为 14/16，但 8 条完整 token 流不同。该小集合是功能诊断，不替代 256 题质量验收，
也不用于性能计时。证据 `same-binary-profile-16-v1/` 保留全部输出与两份 JSONL。

这 16 条请求在两次中均各用一次完整 prefill，chunk 切分相同。
8 条不同流恰好对应第一次 decode 多走或少走一次私有路径；另外 8 条均直接进入 pooled
且流一致。不同 slot 编号本身未导致这些其余请求改变。
`grouping-correlation-v1.json` 记录各请求首个不同 token、私有步和首次 slot 位置。

源码确认小 BF16 投影在私有路径使用 F32 GEMV、pooled 使用两个 BF16 分量的补偿 MMA，
其浮点评估不同。上述相关性说明应隔离这条路径，但还不能独自证明所有波动的原因。
后续私有小投影实验默认关闭，先比较完整 hidden、词表、F32 state 和 FP8 KV，
然后按原独立参考、完整任务质量与成本门槛处理。
