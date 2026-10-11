# CUDA 性能测量与调优

CUDA 执行器使用原生 Rust/cuTile kernel 和设备驻留图。测量按算子、执行图、模型和服务分层；算子加速不能替代完整模型数值、状态与服务验收。构建要求见 [CUDA 后端](../../crates/backend/cuda/README.md)。

本文说明现有工具和测量边界。正式测试与性能工作的 release、当前 native/vLLM 配置对齐和基线验收统一按 [性能基线方案](../plans/performance/baseline.md) 执行；自动校验仍待实施。现有服务对比有 [已审查缺口](../reviews/vllm-benchmark-methodology-2026-10-09.md)，历史矩阵不作为当前已验收 baseline。

## 投影基线

```sh
mkdir -p artifacts
bash tools/bench/cuda-baseline.sh 17408 5120 artifacts/cuda-baseline
```

输出目录必须尚不存在。脚本默认使用 release，保存二进制与源码 SHA256、工具链、构建日志、设备快照和原始样本。矩阵尺寸应取自模型投影清单；`cuda-baseline --catalog /path/to/model` 只解析元数据，不执行完整模型。

BF16 GEMV 候选先对照独立 CPU 点积，再用 CUDA events 测量。A/B 交替、预热、冷 L2 与原始样本的具体参数由工具记录。这条同步投影基线包含分配、清零和提交成本，不能标成裸 kernel 时间。已有可复用测量数据见 [基线目录](../../benchmarks/baselines/README.md)。

## 模型与服务

```sh
cargo build --locked --release -p infer-backend-cuda --features cuda \
  --example cuda-model-smoke
bash tools/bench/safe-run.sh target/release/examples/cuda-model-smoke \
  /path/to/model '只输出17乘23的结果。' 32 \
  --device-graph --prefill-batch 32 --mtp 0 --thinking false --temperature 0
```

模型诊断入口与生产 CLI 服务分别验收。服务运行方式见 [开发指南](development.md)，服务与 MTP 对照工具见 [工具导航](../../tools/README.md)。固定模型、prompt、seed、采样和输出长度；比较 MTP 开关时核对完整 token 序列、接受率、拒绝后的 KV/recurrent state 回滚、EOS 与容量边界。

| 层级 | 必须区分的指标 |
|---|---|
| 单算子 | 数值误差、dtype、shape、预分配范围、CUDA event 延迟 |
| 执行图 | 捕获/JIT、首次运行、稳态 replay、主机提交、prefill/decode/draft/verify |
| 完整模型 | TTFT、TPOT、有效 tokens/s、峰值显存、MTP 接受率 |
| 并发服务 | 到达率、并发、拒绝/取消、P95/P99、满足 SLO 的 goodput |

`safe-run.sh` 默认限制任务主机内存为 32 GiB、禁用任务 swap，并要求限额之外保留至少 16 GiB 可用内存；`--memory-gib` 可降低限额。报告与 profiler 输出先放在 `artifacts/`；正式基线按 [证据留存规则](../plans/performance/baseline.md#证据保留与基线更新) 持久保存，不写入使用文档作为逐轮日志。

## 通用 Attention 对照

`make check-attention` 将原生实现与隔离的 Candle FP16/BF16 基线比较。它同时限制数值误差、完整流程、核心调用、逐形状延迟、P95 与跨轮稳定性，详细契约见 [质量门禁](quality-gates.md#通用-attention-门禁)。

保留的验收结论：2026-10-07 RTX 5090 / CUDA 13.4.92 / Candle 0.11.0，31 个用例、5 轮测量中，完整流程几何平均加速约 1.25×，核心调用约 0.69×；长序列退化，结果为 `review`（非零退出），性能未放行。加速比为 Candle 耗时 / 原生耗时。原生补偿 F32 与 Candle FP16/BF16 的精度契约不同；该结果不能外推其他设备，也不能用完整流程收益掩盖核心退化。

## 能力与精度

设备能力由 driver 查询，模型 provider 声明 `PrecisionPolicy`，加载时选择合法存储和计算路径。支持原生 FP4 与支持某个高效 Tensor Core tile 是两件事；数值通过也不证明编译器生成了预期指令。NVFP4→BF16 加载转换不会恢复原始权重精度，并会增加显存需求。Hopper 转换路径不等于已完成 H200 实机验收。

NVFP4 lowering 探针为 `cuda-mma-probe`，用 `tools/bench/inspect-cutile-cache.py` 检查 cubin/SASS；固定设备上的能力证据保存在 `benchmarks/capabilities/`。变更 tile 或工具链需重做数值、lowering 与性能验证。

MLP 的 `MlpConfig.pdl` 默认关闭，仅控制 SiLU×up → down projection 依赖边。生产者 signal 必须依赖 store token，消费者 load 必须依赖 wait token，graph 保持缓冲区生命周期；具体安全证明在 `mlp/pdl.rs` 与 `mlp/pdl_consumers.rs`。显式启用前校验工具链和架构，使用 `cuda-mlp-check --pdl` 验证。正确性通过不代表已证明 PDL 加速。

## 调优规则

1. 候选先校验独立参考、尾块、非整除维度与状态语义，再测量。
2. 按实际 dtype、shape、layout、执行阶段和设备测量；prefill、decode、MTP 不共享未经验证的性能结论。
3. 构建身份、配置对齐、冷/热条件、采样统计与证据按性能基线方案核验。
4. 搜索结束后独立复测；驻留 GEMM 用 AB 搜索、反序 BA 确认，不能用最快一次样本替换默认实现。
5. 用 profiler 定位访存、提交、occupancy 或 spill，再决定融合与流水化；功能边界由通用 API 保持。
6. 解码 slot 数与 KV 容量改变会影响常驻显存、准入与并发，必须连同完整服务负载重测。

### 驻留 GEMM 自动择优

驻留 NVFP4 与按通道缩放的 FP8 GEMM 在实际图捕获前选择配置，使用同一 `tuning.json` 的独立
typed section，避免与标量 GEMV 策略混淆。缓存和迁移规则见
[CUDA 后端](../../crates/backend/cuda/README.md#自动-tile-选择)：GEMV 本地测量优先于 shipped 基线，
旧单设备条目迁移到多设备表，缺失设备身份的记录不复用。

驻留 key 隔离硬件事实、CUDA API 与真实 NVRM driver build、编译器及 kernel fingerprint、
实际 `M/N/K`、激活模式、scale/布局与数值合同；Lt 另外隔离库版本和 typed 算法配置。
同形状可复用首份真实权重的性能搜索，但每份实际权重和 scale 仍需通过所选配置的完整输出
finite/bitwise 验收。验收记录关联仍存活的权重 owner，不靠形状或可复用裸地址跳过新权重检查。

默认原生 NVFP4 搜索现有 row/column tile 及 plain/prefetch main loop，保留 K 步长、block scale
和量化算术；原生 FP8 搜索同一 K 累积顺序的 tile。选择由实测决定，原先按 SM wave 阈值放大
NVFP4 列 tile 的策略已删除。设备能力、合法布局、对齐、workspace 和容量/显存上限仍是硬约束，
不能通过测量绕过。Attention、图宽度和 GEMV/GEMM crossover 尚未全部纳入这个 chooser。

性能搜索用固定顺序的 AB 配对；赢家以新的 capture 做反序 BA 独立确认。搜索与确认分别覆盖
冷 L2 和暖 L2，非 baseline 配置在四组比较中都至少优于保守配置 2%。原始 samples、两臂名称、
中位数、比值和顺序随策略保存，读缓存时校验完整性。计时范围是预分配的“激活量化 + GEMM”
graph replay，不含模型加载、分配或首次 JIT。单候选的 capture、数值或测量失败不终止其余合法
候选；基线失败仍回退，不能据此假设 CUDA 错误已恢复。

每次模型加载的 target、draft 和 program 共享 30 秒驻留调优软预算，首次驻留身份解析、搜索或
数值验收才启动，后续新候选和新权重验收继续检查剩余预算。单次已经开始的 JIT/测量不可中断；
超时或未完成证据时回退，不把未测 baseline 当成完整调优证据缓存。`--no-autotune` 不做新的
性能搜索，但可读取有效缓存并做当前权重数值验收。诊断用 `INFER_CUDA_QUANT_TILE` 与
`INFER_CUDA_VENDOR_FP8` 的覆盖结果不写入调优证据。

### FP8 Lt 显式数值 profile

默认只允许原生 FP8 tile 候选。设 `INFER_CUDA_FP8_LT_AUTOTUNE=1` 或 `true`，才会搜索
合法 cuBLASLt 算法，并继续接受真实权重的四种激活 pattern、完整输出 finite/bitwise 门禁，
以及冷/暖 AB 搜索和 BA 独立确认。其描述符、算法可用性和零 workspace 在 capture 前准备。

cuBLASLt SDK 不保证与原生 kernel 使用相同 K 累积顺序，四种输入 pattern 的逐位一致也不是
所有输入的数学等价证明。该开关明确启用一个数值 profile，不能据此改成默认 Lt 路径。
`INFER_CUDA_VENDOR_FP8=1`（或 `true`）只用于绕过 chooser 的独立诊断，不产生 tuned 证据。
这些算子层门禁尚不能证明完整服务性能收益；正式采用仍按模型 token、状态和服务负载验收。

下列 release 入口是独立候选 sweep，可保存逐形状冷/热 L2、三轮 A/B 交替的 CUDA event 样本；
它不替代 chooser 的反序确认、实际权重验收或服务对照。
计时范围为预分配的“激活量化 + GEMM” graph replay，不含加载、分配或首次 JIT：

```sh
bash tools/bench/safe-run.sh --memory-gib 8 cargo test --locked --release \
  -p infer-backend-cuda --features cuda --lib nvfp4_prefill_performance_comparison \
  -- --ignored --nocapture --test-threads=1
```

投影自动 tile 选择与缓存规则见 [CUDA 后端](../../crates/backend/cuda/README.md#自动-tile-选择)。缓存匹配不能替代工具链或 kernel 变更后的回归验证。
