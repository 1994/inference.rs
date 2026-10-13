# CUDA 相邻 SwiGLU 量化复用（2026-10-13）

本轮完成 main 上相邻 gate/up 投影的输入量化复用。基线为
`08969ed2a931abe6fea9b985cdfb5be6671a4bf2`，开发、编译、GPU 验证和推送均在
`r@192.168.1.45`。这是减少重复工作的改动，尚未记录性能收益。

## 实现与边界

规划器只识别相邻的 Linear(gate)、Linear(up)、Silu、Multiply 数据流。
两次 Linear 必须读取同一逻辑输入，且量化契约一致。NVFP4 输入全局缩放逐位匹配；
per-token FP8 可复用，block-scaled FP8 使用原路径。

复用许可只存在于一次 capture 内，并在每个逻辑节点检查。运行时键包含实际输入地址、
完整形状与步长、M/K、量化模式、NVFP4 scale bits，以及 codes/scales 缓冲区地址。
输入与输出重叠、未知 view、节点跳过、capture 重启或失败都会撤销许可。
每张图仍保留第一份量化 kernel，因此每次 replay 都重新量化当前输入。

接入单行 decode、pooled decode 和完整矩阵 prefill；逐 lane 的旧验证路径保留原量化。
原 GEMM、权重 allocation、输出、缩放顺序、vendor swizzle、调优选择与资源销毁顺序均保留。
没有增加 GPU 缓冲区，没有新增与 GPU 型号绑定的 tile 或阈值。
现有硬件调优仍负责每次原始 GEMM 的配置。

本 checkpoint 的 64 组 MLP gate/up 都满足条件。此次范围是这些相邻 MLP 投影；
此前研究中的其他共享输入投影尚未接入。

## 已执行验证

- CUDA release 全 target 严格 clippy、格式化、布局与仓库策略检查通过。
- 默认 CUDA 测试：58 通过，0 失败；74 个需要显式启动的测试保持 ignored。
- 测试登记检查：579 个入口，新增 8 个，src 内测试体仍为 0。
- 4 组真实权重：首末 NVFP4 层与首末 FP8 层，M=1/4/64/256。
  每组包含普通、零、微小、稀疏、不同符号及输入变化，按 AB/BA 重放。
  256 条记录共比较 724172800 个完整输出、79872000 个 codes 字节、
  3338400 个 scale 元素，均有限且逐位相同；完整量化数据也与独立 CPU 量化器一致。
- 同一份权重、同一二进制中开启/关闭复用的整模型对照通过：
  prefill width=64/256，实际长度 1/51/64/255/256/511、非零 base、尾部 masking、
  长前缀后的 reset、每个长度后 8 步 forced decode、16 步 greedy、
  verification 与部分提交/回滚、pooled 四路停用/恢复及释放后重新绑定。
  完整 hidden、词表 logits、F32 state 和 FP8 KV 逐位一致。
- 编译源码、测试二进制和调优缓存的 hash 均保留；功能验证没有 GPU idle 门槛。

这些是数值与状态等价验证；没有把它们记作新增 GSM8K 256 题任务质量评分，
也没有把先前 prototype 或其他提交的服务数据归入本轮性能收益。

## 证据与后续测量

45 上证据目录：
`/home/r/benchmarks/inference.rs/shared-quant-reuse-2026-10-13/`

关键文件：`preparation.json`、`compiled-source.json`、`build-result.json`、
`pair-numeric.log`、`model-numeric.log`、`numeric-result.json`。
测试二进制 SHA256：
`e78ab31c8b5dd7c96cafc4404a1220a6759e067e637e654dccb0e68d1d3b7974`。
早期编译/lint/仓库策略失败保留在 `build-attempt-0/`，未作为通过证据。

代码推送并通过 GitHub CI 后，再测完整 gate/up 图的 cold/warm AB/BA 成本，
以及优化前后 main 的短提示、511-token 长提示和四路请求。
两个 CLI 从各自冻结源码编译，使用同一份调优缓存、相同请求与 MTP=0；
每个服务循环包含 warmup 与三个正式 repeat，并检查完整 token 流一致。
硬件计时前仅做有界的短暂 idle 采样。性能结论将补在本报告中。
