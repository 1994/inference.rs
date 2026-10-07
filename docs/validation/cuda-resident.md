# CUDA 设备驻留图验证（RTX 5090）

2026-10-06，CUDA 13.4 / cuTile 0.4.0，模型 `/home/r/models/Qwen3.8-27B-NVFP4`。

## 已实现

- `resident::DeviceProgram` 捕获完整 target token 图：497 个投影及全部辅助算子在 GPU 上执行。
- 激活按最后使用位置复用同尺寸槽位；hidden/logits 保留至回传。真实模型 activation workspace 为 **1,481,088 字节**。
- Conv 历史、GDN recurrent 状态、attention KV 常驻设备。KV 连续存储，默认 F32；`--fp8-kv` 根据模型 k_scale/v_scale 启用 target E4M3 KV，draft 保留 F32。
- decode 图与跳过词表投影的 prefill 图共享固定缓冲区；节点表和张量映射只在准备阶段处理。
- hidden/logits 合并回传；仍有一次 metadata 上传等待及一次图结束等待，尚非完全消除主机同步。
- MTP 双 RMSNorm、FC、草稿层均在图中执行；与 target 共享 embedding / lm_head。
- attention-only draft 使用逻辑位置回退，重放覆盖 suffix，掩码排除旧 suffix；禁止用此方法回退 GDN/Conv。
- 权重绑定、模型配置位于模型适配层，图执行和算子不依赖 Qwen 权重名。

## 数值检查

`cuda-resident-check` 与 Rust CPU 参考对照，覆盖变化输入、多 CTA、GDN 多头映射、归一化、部分 RoPE、分组 attention、滑动窗口、重置及草稿 KV 回退。MTP fusion 单独验证变化 token / hidden。允许 F32 与 CPU F64 累加之间的舍入差异，不要求逐浮点位一致。

真实模型短验证：提示“只输出17乘23的结果。”，greedy、非 thinking，target 与 MTP2 均输出 `391`。MTP2 实际经历 2 次接受、1 次拒绝、2 次状态恢复。该短例 MTP 没有净加速。

## 第一轮数据集结果

固定 ModelScope GSM8K / ShareGPT 各 4 条，manifest、revision、文件 SHA256 见 `artifacts/modelscope-pilot.json`。逐请求串行，greedy，presence penalty=0，最大输出 128；一次完整请求 warmup，加载与 warmup 不计时。**这是 8 条 pilot，不是完整测试集或 SLO 测试。**

| 指标 | 整步图，MTP 关闭 |
|---|---:|
| 完成请求 | 8/8 |
| 输入 token | 355 |
| 输出 token（不含 EOS） | 820 |
| 测试窗口 | 20.3843 s |
| 端到端输出吞吐 | **40.23 tok/s** |
| 达到输出长度上限 | 6/8 |
| GPU 忙碌率均值 | 96.2% |
| 内存控制器忙碌率均值 | 84.2% |
| 全设备显存 | 23,270 MiB |
| GPU 功耗均值 | 434.6 W |

8 条输出 token 与旧诊断路径全部一致。旧运行曾受并行加载干扰，只使用其输出核对正确性，**不据此发布性能加速比**。

原始结果：`artifacts/modelscope-rust-resident-mtp0.json`；硬件记录：同名前缀 `-hardware.jsonl`；二进制及 SHA256：`artifacts/resident-v1/`。此轮为逐行 attention 的 v1 基线，后续分块 attention 须独立测量。`artifacts/` 为本机产物，不入版本库。

1 秒遥测不能说明每个 kernel 的 occupancy、L2 命中率或 Tensor Core 利用率。`utilization.memory` 不是实际 GB/s / 峰值 GB/s；桌面也共享此 GPU。CPU 忙碌率包含 CUDA 同步等待，不能把约一个核的占用全部归因于计算。

## 仍未完成

prefill 辅助算子批量化和大型 GEMM 流水、设备采样、完整生产 BackendProvider / continuous batching 接线，以及 H200 实测。32-token BF16 tensor-core prefill 已接通，验证边界见末节。FP8 KV、三候选融合 GEMV verify 与设备端 recurrent prefix checkpoint 已实现；这些不等于上述剩余项目已完成。当前 F32 激活、权重解码策略与 vLLM 的动态量化精度不完全相同；对照结果必须注明，不能冒充等精度优化胜负。

PDL 与 NVFP4 MMA 探针另见 [PDL](cuda-pdl.md)、[sm_120 MMA](cuda-sm120-mma.md)。编译通过和 SASS 存在不等于端到端加速。

## 安全复现

所有 GPU 运行经 `bash tools/bench/safe-run.sh`，单任务互斥、单进程 JIT 编译、MemoryHigh=28 GiB、MemoryMax=32 GiB、MemorySwapMax=0、整组 OOM 清理、30 分钟超时。硬件监控在 MemAvailable 低于 8 GiB 时中止运行。性能测试与数值测试串行执行。

## 按实际投影形状调优

`--tune-projections` 加载模型权重后，对每种 dtype/shape 搜索 12 个 tile，并对候选赢家重新做配对测量。每个候选先检查与默认 tile 的输出一致性；独立复测不足 2% 的收益保留默认值。图回放测量使用 CUDA events、清 L2，原始 A/B 样本保存在 JSON 中。它是算子筛选工具，最终取舍还要看完整模型测试集。

```sh
bash tools/bench/safe-run.sh target/release/examples/cuda-model-smoke \
  /home/r/models/Qwen3.8-27B-NVFP4 unused 1 --mtp 2 --tune-projections \
  > artifacts/rtx5090-projection-tuning.json

bash tools/bench/safe-run.sh target/release/examples/cuda-model-smoke \
  /home/r/models/Qwen3.8-27B-NVFP4 unused 128 --device-graph \
  --tuning artifacts/rtx5090-projection-tuning.json \
  --thinking false --temperature 0 --presence-penalty 0 \
  --dataset artifacts/modelscope-pilot.json
```

调优结果按 GPU 名称与 SM 架构校验；模型适配器只把 dtype/shape 的选择映射到 TensorId。H200 必须单独生成和验证配置。cuTile 0.4 的 `allow_tma` / `latency` 是 load/store 级参数，不能写成 entry hint；`num_worker_warps_per_cta` 等入口 hint 的收益同样需要实测。

## vLLM 0.31 实际对照（MTP 关闭）

本次已完成真实对照，不再缺少 vLLM 结果。相同输入 token、采样配置、128-token 上限、并发 1、一次请求 warmup：

| 方案 | 输出 token（不含 EOS） | 请求总时间 | 输出吞吐 |
|---|---:|---:|---:|
| Rust 设备图 v1 | 820 | 20.384 s | 40.23 tok/s |
| vLLM 0.31 | 821 | 12.441 s | 65.99 tok/s |

Rust 当前未追平。Rust 的总 prefill 时间为 5.675 s；扣除 prefill 后的请求时间为 14.709 s / 814 次 target forward，约 **18.07 ms/步**，包含采样等主机工作，不能冒称纯 kernel 时间。

vLLM 使用 FlashInfer NVFP4 GEMM、FP8 KV、BF16 查询及其原生量化执行；Rust 为权重解码与 F32 激活/KV。7/8 请求的输出 token 序列不同，仅一条完全一致，因此此处是同一模型包/请求配置的方案吞吐对照，**不是等精度、等输出工作量对照，也不是质量胜负**。6/8 请求触及长度上限，不能推出完整题目解答速度。

vLLM 首次加载和 JIT 初始化约 683 s，排除在计时外；受控进程组内存峰值 13.7 GiB、swap=0，无 OOM。配置限制 `max_num_seqs=1`、`max_num_batched_tokens=512`、`max_model_len=4096`、显存比例 0.85，关闭 prefix cache。

报告：`artifacts/modelscope-vllm-safe-mtp0.json`；硬件：`artifacts/modelscope-vllm-safe-mtp0-hardware.jsonl`；统一汇总：`artifacts/resident-vllm-summary.json`。

## 后续实测：形状调优、分块 attention 与 MTP2

同一份 8 请求 pilot，全部排除加载和 warmup，输出不含 EOS：

| 方案 | 输出 token | 时间 (s) | tok/s |
|---|---:|---:|---:|
| Rust v2：分块 attention + 已验证投影 tile，MTP0 | 820 | 18.190 | 45.08 |
| Rust v2，顺序 MTP2 verify | 820 | 20.367 | 40.26 |
| Rust 节点交错、独立 GEMV 的 MTP2 verify | 820 | 19.615 | 41.81 |
| Rust 三候选共享权重 tile 的融合 GEMV，MTP2 | 820 | 19.473 | 42.11 |
| Rust FP8 KV，MTP0 | 820 | 18.202 | 45.05 |
| vLLM 0.31，MTP0 | 821 | 12.441 | 65.99 |
| vLLM 0.31，MTP2 | 813 | 6.306 | 128.92 |

三候选投影保持 F32 累加，是融合 GEMV，**不是 tensor-core GEMM**。MTP 的 Conv/GDN 快照留在设备上，拒绝后提交已接受前缀；KV 通过逻辑长度排除旧后缀。部分批次的 padding 也必须回退。`--sequential-verify` 保留顺序对照。

Rust F32 各路径的 8 条 token 序列完全一致。FP8 KV 的序列有变化；短上下文吞吐没有明显收益，因此保持显式开启。FP8 检查包含跨 32-token attention 块、滑动窗口和独立 E4M3 最近偶数舍入参考。

MTP2 当前没有净加速，默认深度仍是 0。84.1% 的草稿接受率并不足以抵消 draft、验证和状态处理成本。不能把接受率换算成预期吞吐，也不能凭 41.81→42.11 的单次差异宣称稳定收益。

v2 的 GPU 忙碌率均值 95.2%、内存控制器忙碌率 88.9%、显存约 23,163 MiB、功耗 429.2 W。vLLM MTP0 为 96.75%、84.92%、27,653 MiB、373.61 W；MTP2 为 100%、84.33%、27,771 MiB、368.48 W。后者仅有 6 个计时窗口内样本，不能支持尾延迟或 kernel 利用率判断。

原始数据前缀：`artifacts/modelscope-rust-resident-v2-*`、`modelscope-rust-batch-mtp2`、`modelscope-rust-fused-batch-mtp2`、`modelscope-rust-fp8-mtp0`、`modelscope-vllm-safe-*`；每组都有独立硬件 JSONL。没有 H200 实测，也没有并发 goodput、长上下文 TTFT 或 P99 ITL 结论。

## 三 token prefill

`--prefill-batch 3` 显式启用，默认 1。它把相邻 prompt token 的投影放到同一融合 GEMV 中，Conv/GDN/attention 仍按因果顺序执行。兼容 MTP0、MTP2，其他深度需使用顺序 verify。

首次复用 verify 图的测量：820 token / 17.533 s = 46.77 tok/s，总 prefill 4.438 s，8 条输出与单 token 路径一致。后续专用 prefill 图跳过中间 logits 投影和回退快照；其结果单独记录，不能混用第一轮数字。

### 专用 prefill 图结果

MTP0 三次相同配置测试分别为 **47.48、47.56、47.51 tok/s**，中位数 **47.51 tok/s**；每次 820 个输出 token，8/8 输出序列均与 F32 单 token 对照一致。首次测量总 prefill 为 4.176 s；扣除 prefill 后约 16.09 ms/target step，包含采样与主机工作，不是 CUDA event 测得的纯 GPU 时间。

MTP2 + 专用 prefill 为 **44.32 tok/s**（820 token / 18.502 s），8/8 序列相同。MTP0 首次运行 GPU/内存控制器忙碌率为 95.35% / 76.82%，显存约 23,646 MiB、429.77 W；MTP2 为 91.11% / 51.63%，约 24,446 MiB、407.88 W。三候选批量处理降低了内存控制器忙碌比例，不能据此推断算子已达到计算上限；仍需 kernel counters。

各次受控运行成功、swap=0；最新运行进程组内存峰值约 5 GiB。原始结果 `artifacts/modelscope-rust-prefill3-v2-mtp{0,2}.json`、MTP0 的 `-repeat2/-repeat3` 及同前缀硬件 JSONL。二进制快照 `artifacts/resident-v3/`。

可入库的数字与原始文件 SHA256：[汇总基线](../../benchmarks/baselines/rtx5090-qwen-resident.json)。投影策略数据：[形状基线](../../benchmarks/baselines/rtx5090-resident-projections.json)。最终判断仍是：**本轮优化有实测收益，但尚未追平 vLLM，剩余优化未全部完成。**

## 32 token BF16 Tensor Core prefill（开发验证）

`--prefill-batch 32` 使用独立 prompt 图，与 MTP verify 图共享模型状态但不共享回退快照。矩阵乘法采用 `32×64×64` tile，BF16 操作数、F32 累加；BF16、channel-scaled FP8、NVFP4 权重均可进入此路径，量化权重按 tile 解码。NVFP4 的 block scale 在转换为 BF16 之前相乘，全局逆 scale 在累加之后应用；FP8 channel scale 在累加之后应用。

尾块通过 `kv_position=-1` 标识 padding lane。Conv/GDN 不更新状态、KV 不追加、attention 输出零；其余逐行运算与矩阵乘法允许计算无效 lane，但只回传有效 hidden 和最后一个有效 token 的 logits。因而尾块无需复制或恢复 recurrent 状态。终端 lm_head 也使用批量 MMA。

这会把 prompt 投影输入舍入到 BF16，不能声称与此前 F32 GEMV 逐位等价。decode/MTP verify 仍使用 F32 GEMV。结果 JSON 用 `prefill_math`、`precision/limitation` 和 `target_kv_cache` 分开记录，不再把该模式描述为全程 F32 激活。

验证证据：

- `artifacts/resident-prefill32-multik-check.log`：65×48 / 97×80（含跨 K tile），三种非均匀权重、不同 channel/block scale、BF16 舍入前的 F32 SiLU、完整块＋17-token 尾块、接续 decode；Conv/GDN/F32 KV/FP8 KV 状态与独立 Rust CPU 参考比较。
- `artifacts/resident-prefill32-tail-smoke.json`：真实 27B，49 输入 token，MTP2，输出 `391`；实际经历 2 次接受、1 次拒绝、2 次恢复。受限进程组成功结束，内存峰值约 5.1 GiB、swap=0。
- [sm_120 BF16 MMA 证据](../../benchmarks/capabilities/sm120-prefill-bf16-mma.json)：三种投影的 release SASS 均包含 `HMMA.16816.F32.BF16`。这是 BF16 Tensor Core 路径，不是原生 NVFP4 MMA。

短验证不是正式性能基线，也不证明完整模型质量。尚未重跑 vLLM 基线；按开发顺序完成其他缺口后统一更新。当前剩余开销包括每个投影的逐 lane pack/unpack、逐 lane 辅助算子、32 份 metadata 上传/hidden 回传以及串行 GDN/attention。H200 加载转换已实现、实机验证仍缺失；设备采样及生产服务接线也尚未完成。

## Hopper NVFP4 加载转换（5090 上验证）

`CudaTarget::nvfp4_storage` 明确选择 `sm_90 → DecodedBf16`、`sm_120 → NativePacked`，未知架构拒绝未验证的 NVFP4 策略。转换代码独立于模型权重名；模型适配器只读取 payload 并绑定转换后的 BF16 投影。报告新增 `nvfp4_loading_policy`。直接 native FP4 API 在 Hopper 上仍拒绝启动。

转换逐个矩阵执行，输出 staging 预算 1 GiB，检查维度/长度/预算溢出、有限非负 E4M3 block scale、正有限 global scale及 BF16 溢出。保留原始模型文件；BF16 舍入不等于恢复原始未量化权重。

真实矩阵验证：`layers.9.mlp.up_proj.weight`，17408×5120，packed 44,564,480 字节 → BF16 178,257,920 字节。在 RTX 5090 上完成转换后 GEMV，所有输出与转换后 BF16 权重的 F64 累加参考比较，最大绝对误差 **5.38e-8**。这个误差衡量 GPU 投影实现，不衡量相对于原始 FP4 或原始未量化模型的质量。受控进程组峰值 466.4 MiB、swap=0，无 OOM。

```sh
CARGO_BUILD_JOBS=1 CUDA_TOOLKIT_PATH=/opt/cuda cargo +stable build --release \
  -p infer-backend-cuda --features cuda --example cuda-nvfp4-conversion
bash tools/bench/safe-run.sh target/release/examples/cuda-nvfp4-conversion \
  /home/r/models/Qwen3.8-27B-NVFP4
```

证据：[转换验证记录](../../benchmarks/capabilities/nvfp4-bf16-conversion.json)，原始报告 `artifacts/hopper-weight-conversion-5090.json`。**H200 硬件执行、显存峰值、完整模型质量与性能仍未验证。** 生产 BackendProvider 接线仍属于剩余开发。

转换后的 BF16 投影还通过 `artifacts/resident-hopper-conversion-check.log` 的 65×48 / 97×80 非均匀权重、decode、三候选 verify、32-token prefill、SiLU 后舍入与尾块连续性检查。`artifacts/hopper-policy-native-regression.json` 确认 5090 模型仍采用 `NativePacked`，同一 49-token 提示输出 `391`，MTP2 的 2 次接受、1 次拒绝、2 次恢复与此前短验证一致。

## 库级加载与请求状态隔离

权重读取/转换已从 `examples/model_smoke/weights.rs` 移到 `loading` 库模块，诊断和设备图 runner 共用；Qwen 静态 KV scale 名称解析也只有一份实现。新增 `loading::LoadedModel::open` 和 `sequence(capacity)`：前者持有只读 GPU 权重，后者为每个请求分配独立 Conv/GDN/KV、激活与 CUDA 图。这个入口不创建 CPU reference 状态，也不依赖示例目录代码。

`cuda-loaded-model-check` 用真实模型创建两个 sequence，再释放 LoadedModel。两个 sequence 的 prompt logits 相同，先推进一个请求再推进另一个，请求均输出 `391`。验证了图保留共享权重的生命周期和独立请求状态。受控运行峰值 4.9 GiB、swap=0。

原示例回归使用 MTP2、FP8 KV、32-token prefill，仍输出 `391`，实际经历接受/拒绝与恢复。数据：`artifacts/cuda-library-model-check.json`、`artifacts/cuda-loading-mtp-regression.json`；可入库摘要：[库级加载验证](../../benchmarks/capabilities/cuda-library-loading.json)。Clippy、格式、策略检查通过。

**这是生产接线的依赖拆分，不是 BackendProvider 验收。** 还需要实现并验证引擎级的 admission/状态预算、submit/poll ticket 所有权、错误清理、runtime/scheduler/CLI 服务接线。库级多请求隔离也不代表已支持 GPU continuous batching 或 paged KV。


## 同步 BackendProvider 与按需回读（2026-10-06）

`executor::CudaBackend` 将上述 resident graph 接入 `infer_spi::BackendProvider`，由
`CudaKernels` 注册 14 类逻辑操作。模型/图/精度/操作和 kernel ID 在提交前与加载时编译的计划核对；
注册成本为引导占位值，不是性能测量。权重布局适配留在 `loading`，硬件查询留在已审计的 `device` 边界。

- 每个请求拥有独立 Conv/GDN/KV、激活、graph 和游标；共享只读模型权重。
- 完成票据绑定 backend，禁止重复消费，消费前禁止重置、释放或新提交。
- 部分执行失败返回失败票据，并使相关状态失效；设备尚未排空时继续保留所有权。
- 分配前检查请求数、配置预算和实时空闲显存，保留 1 GiB 显存余量。
  张量保守估算包含无复用激活、projection scratch、验证 checkpoint、KV 和 256 MiB graph/allocator 余量。
  这是 admission 估算，CUDA driver 的额外分配仍可能返回 OOM，不能把它当作物理内存精确计量。
- Logits/None 请求不再回读隐藏状态；32-token prefill 跳过逐 lane hidden 回传。
  Full 请求保留完整隐藏状态历史。现有 MTP 示例继续使用需要 hidden 的接口。

真实模型验收命令（先按仓库 CUDA 环境构建）：

```sh
cargo +stable build --release -p infer-backend-cuda --features cuda --example cuda-provider-check
bash tools/bench/safe-run.sh python3 tools/bench/hardware-monitor.py \
  --output artifacts/cuda-provider-check-hardware.jsonl -- \
  target/release/examples/cuda-provider-check /home/r/models/Qwen3.8-27B-NVFP4
```

首次 SPI 验证：两个请求均生成 `[18, 24, 16, 248046]`（`391`），Full/Logits 输出契约、
请求隔离、非法游标/kernel、重复预留、容量限制、完成前释放/重置、重复 poll、reset/replay 均通过。
受限进程峰值主机内存约 4.9 GiB、swap 0；原始证据 `artifacts/cuda-provider-check.json` 与同名前缀的 log/hardware 文件。

**边界：这是同步 provider。** 多请求顺序执行，各请求内部使用 CUDA Graph；尚未实现 CUDA CLI/服务选择、
异步 GPU flight、物理 paged KV/prefix、设备采样或 GPU continuous batching。
本次是功能验收，不更新此前 tok/s 基线，也不构成 P99/SLO 或 H200 生产验收。

后续通过完整 `Engine` 再提交两条真实模型请求，调度/增量输入/CPU 采样/完成回收链均得到相同生成序列，
每个 tick 的状态不变量检查通过。证据 `artifacts/cuda-provider-engine-check.json`；
可入库摘要：[CUDA BackendProvider 验证](../../benchmarks/capabilities/cuda-backend-provider.json)。
引擎联调总进程约 10 秒、主机峰值 4.9 GiB、swap 0，包含模型加载和图准备，**不是稳态吞吐数据**。


## CUDA CLI / HTTP 服务接线（2026-10-06）

Linux `infer-cli --features cuda` 现可选择 CUDA（含 `auto` 优先选择），正式 `serve` 入口加载同一 native
`LoadedModel` / `CudaBackend`。部署配置见 `examples/cuda-runtime.json` 和 README 的 NVIDIA 启动命令。
`--gpu-memory-utilization` 同时约束权重与状态：payload 超出预算时先拒绝，加载后仅将剩余预算交给状态分配器。
当前默认 prefill 为 32；KV block 数和 upload staging 覆盖等未实现选项会明确拒绝。

发现并修复了真实服务路径的超时覆盖：`RuntimeHandle::start` 原先用默认 frontdoor 的 250 ms
覆盖引擎配置的资源准备超时，首次 graph 准备因此返回 503。现在默认启动保留
`RuntimeConfig.resource_timeout_us`；显式 `start_with_config` 仍使用调用方指定的控制超时。

受保护验收 `tools/bench/check-cuda-service.py` 检查：

- `doctor` 报告 CUDA 可用，`/v1/models` 返回已加载模型；
- OpenAI chat 单请求与两个同时发起的请求均返回 `391`；
- 手动 greedy/非 thinking 覆盖生效，省略采样字段的请求可完成；
- SIGINT 正常退出并排空请求，无残留模型进程。

原始报告 `artifacts/cuda-service-check3.json`，硬件记录同名前缀 `-hardware.jsonl`；
[可入库验收摘要](../../benchmarks/capabilities/cuda-cli-service.json) 含二进制和原始报告哈希。
总任务约 9 秒、主机峰值约 4.8 GiB、swap 0，包含加载与图准备，不能作为稳态吞吐。

这补齐了上一节的 CLI/服务选择缺口。执行仍为同步 provider：HTTP 并发与调度正常工作，但 GPU 请求顺序执行；
服务内 MTP、设备采样、GPU 连续批处理和物理 paged KV/prefix 等仍待开发。
CUDA per-op tracing、layer probes 和诊断 fork 也未提供，入口不会伪造这些数据。


CLI `run` 的附加验收发现旧轮询用 tick 序号充当微秒，导致真实 GPU 请求被报告为 4 µs。
`support::execution` 已将设备后端切换为 `Instant` 单调墙钟，并从请求提交前开始计时以覆盖状态准备；
CPU 测试后端仍保留确定性逻辑时钟。`artifacts/cuda-cli-run-check.json` 仅能证明生成正确，
其中旧 timing 数字无效，不能用于任何性能对照。验收脚本 `--cli-only` 现检查 GPU E2E 不再是这种虚假步数。

修正后的 CLI 实跑仍生成 `[18, 24, 16, 248046]`，E2E 1,758,701 µs、TTFT 1,710,925 µs、
最大 TPOT 16,124 µs。证据 `artifacts/cuda-cli-run-check2.json` 与[入库摘要](../../benchmarks/capabilities/cuda-cli-run.json)。
这些是首次请求、包含状态/graph 准备的功能验证数据；不能当作已预热的性能基线。
这轮早期验收时请求状态仍在释放时销毁；后续有界 graph/state 复用的实现与复测见下节。


## 有界状态 / CUDA Graph 复用（2026-10-06）

`CudaBackend` 释放请求后保留健康的空闲状态/图，按二次幂容量档位和输出类型匹配。
再次预留先重置 GPU 状态和逻辑游标，恢复新请求的实际容量限制；不复用请求 ID、tokens 或隐藏状态历史。
缓存与活跃请求共用总数量和 admission 字节预算；缓存不匹配或预算不足时先淘汰空闲项。
失败状态不参与匹配；未消费完成票据时禁止 trim、释放、重置或再次提交。
`pool_inspection()` 提供分配/命中/淘汰及缓存预算统计，`trim_state_pool()` 可显式释放空闲图。

真实模型检查覆盖：复用后 logits 与首次执行逐元素相等，Full/Logits 隔离、同档位不同逻辑容量的越界拒绝、
显式 trim、两状态数量限制淘汰，以及三状态数量上限下由 6 GiB 字节预算单独触发的淘汰；
后一组启用 FP8 KV。两组仍生成 `391`，完整 Engine 调度检查也通过。

| KV | 未保留图时的预留 | 已保留图的三次预留 |
|---|---:|---:|
| F32 | 463.798 ms | 0.533 / 0.541 / 0.543 ms |
| FP8 | 464.554 ms | 0.496 / 0.473 / 0.486 ms |

这是同容量、kernel 编译缓存已预热时的**状态/图准备**墙钟，不是 decode 时间或端到端吞吐。
每组仅一次未复用、三次复用测量；不可用这个比值声称整段推理加速。
进程峰值主机内存约 4.9 GiB、swap 0。原始记录 `artifacts/cuda-state-pool-{f32,fp8}.json` 与同名前缀遥测；
[入库验证摘要](../../benchmarks/capabilities/cuda-state-pool.json) 保留原始哈希。


服务路径复测共七个请求（四个相同顺序请求、两个并发请求、一个省略采样设置的请求），均完成；
六个 greedy 请求均返回 `391`，正常退出排空通过。相同顺序请求客户端墙钟分别为
1922.403、148.482、148.960、148.378 ms。
首请求仍含首次执行准备，因此这不是关闭/开启缓存的严格 A/B，不能把差值全部归因于池化；
也不是完整数据集吞吐或 vLLM 对照。原始数据 `artifacts/cuda-state-pool-service.json`，
[服务复用验收摘要](../../benchmarks/capabilities/cuda-state-pool-service.json) 包含报告和二进制哈希。
受限进程峰值主机内存约 4.8 GiB、swap 0。


### 大容量状态分配边界

删除 F32 state / FP8 KV 内部固定 2 GiB 限制，改为在分配该组状态前汇总 shape/字节并校验实时显存余量，
保留 1 GiB 余量；上层 `sequence_budget` 与引擎总预算仍然生效。
CUDA 异步分配器的回收使用独立流，因此淘汰/trim 后使用所属 context 的回收屏障，再查询显存；
仅发生在淘汰路径，命中复用无需该全 context 等待。

实际模型分别以 F32 KV / FP8 KV 成功预留 **8320 token**（物理档位 16384）：
保守 admission 预算分别为 4,795,621,376 / 3,185,008,640 字节。短前缀 logits 与小容量基线逐元素一致，
释放、trim 和后续 Engine 请求通过，进程峰值主机内存约 4.8–4.9 GiB、swap 0。
[验证摘要](../../benchmarks/capabilities/cuda-large-state.json) 含原始报告、遥测和采样显存峰值。
**这验证大容量分配与短前缀正确性，尚不是完整 8k prefill/质量/吞吐验收。**


## 异步 metadata 与批量回读（2026-10-06）

metadata 不再每次临时分配设备张量、H2D 同步后再 D2D 拷贝。Rust/cuTile kernel 通过标量参数
直接写入常驻 metadata，在模型图的同一流排队；更新操作的资源租约由图保留到回放完成。
批量 hidden/logits 读回改用 `DeviceOpVec`，删除单独的图回放等待和逐 lane 的上层 `sync_on`。
**这还不是完全异步回读**：cuTile 0.4 的 `CopyDeviceToHostVec::execute` 为保证 Vec 初始化，
内部在每次 D2H 后同步；外层一次 `sync_on` 并不能消掉这些等待。
32-token Full prefill 删除 32 次 metadata 上层同步和 32 次多余的图/逐 lane 上层同步，
但必要输出仍触发依赖内部的同步，首次 reset 另计。
此轮尚未实现 pinned 异步回读；随后实现见下节。MTP external hidden 上传和跨 prefill 块异步执行仍待完成。

实测发现，直接使用 `i32` kernel 参数会触发 cuTile 0.4 自动 `DivHint` 特化，新 token/position
属性组合在热路径上产生 JIT 停顿。最终实现通过 `f32` 参数传递原始位模式，再在 kernel 内 bitcast
回 `i32`；没有浮点算术或精度转换。负数、i32 极值和 NaN 位模式均逐位验证通过。

5090 release 微基准交替运行两条路径，每轮 200 次，共六轮；包含 metadata 更新、极小的 copy 图
回放和最终同步，kernel 已由正确性检查预热：

| 路径 | 六轮中位数 |
|---|---:|
| 原临时上传 + 同步 | 12.433 µs |
| 固定特化的异步标量更新 | 7.339 µs |

该局部调用墙钟下降 **41.0%**；这不是模型吞吐提升百分比，也不是 GPU kernel 纯执行时间。
[验证摘要](../../benchmarks/capabilities/cuda-async-metadata.json) 保留六轮数据、报告/二进制哈希、
硬件遥测路径及被淘汰的整数参数试验。正式数据集和 vLLM 基线尚未因本项更新。


最终实现通过设备图 63 项数值检查，包含 Conv/GDN、分组 attention、FP8 KV、32-token 尾块、
单步续写与 verify 前缀回退。真实 27B 模型分别通过 F32/FP8 Provider、Engine 与状态池检查，
均生成 `391`；FP8 KV + prefill32 + MTP2 生成 `[18, 24, 16, 248046]`，接受 2、拒绝 1、
恢复 2，与此前相同用例一致。受限进程主机峰值 4.9–5.1 GiB、swap 0。
[整模型回归摘要](../../benchmarks/capabilities/cuda-async-transfer-regression.json) 记录原始报告、
硬件遥测与可执行文件哈希。这些短用例不替代完整数据集质量和吞吐验收。

更新后的 CLI/HTTP 服务七请求回归也通过（含默认采样与并发客户端），主机峰值 4.8 GiB、swap 0；
证据已合入同一回归摘要，服务正常退出排空。


## Pinned D2H：删除依赖内层同步（2026-10-06）

`device/readback.rs` 现在为单步 decode、prefill 和 verify 保留 pinned host staging，
通过 CUDA driver 的 `cuMemcpyDtoHAsync_v2` 排队全部必要输出，模型图与读回共用一次最终 `sync_on`。
CPU 只在完成后把 staging 转成返回 Vec；服务 `BackendProvider` 整体仍是同步接口执行，
不能把步内异步 D2H 等同于请求间异步流水。

所有权边界：设备输入先取得 cuTile 的读租约，pinned buffer 在提交前交给 execution context 保活；
未确定完成时不读取主机数据，失败的回读器不能再次使用。64 MiB 的单回读器容量限制在分配前检查，
同尺寸 staging 会复用，保守 state admission 现在也包含各 lane 的 pinned hidden/logits 存储。
正常热路径没有逐张量等待；异常退出的资源回收由提交上下文保活/排空处理。

5090 release 微基准使用 33 个图内 D2D 拷贝，分别回读 32 行 hidden + 1 行 logits，或只读 logits；
六轮交替，每轮 100 次，均包含 CPU Vec 构造。旧路径为上一轮已合并外层等待的 `DeviceOpVec/to_host_vec`：

| 输出 | 旧路径中位数 | pinned 中位数 | 局部下降 |
|---|---:|---:|---:|
| 33 路 | 323.380 µs | 160.613 µs | 50.3% |
| 仅 logits | 110.425 µs | 99.339 µs | 10.0% |

[微基准摘要](../../benchmarks/capabilities/cuda-pinned-readback.json) 包含原始数据/哈希和遥测。
更新输入后的逐值一致性、可选输出、buffer 复用、容量越界、不同流和 poisoned owner 拒绝通过。
没有注入真实 driver fault；这些局部耗时不是整模型 tok/s 或 vLLM 对照。


## 32 token prefill 批量 activation arena（2026-10-06）

此前 width-32 prefill 的每条 lane 持有独立 arena slot：每个 Linear 节点先把 32 行输入逐 lane
`pack_row` 进 `[32, columns]` 工作区，批量 GEMM 后再逐 lane `unpack_row` 回各 slot；
Norm/GatedNorm/Split/SiLU/Add 等辅助算子也逐 lane 各录一次 kernel。

现在 width-32 改用单个批量 arena：slot 是 `[32 * size]` 的一维平坦缓冲区，GEMM 输入处单次
`view(&[32, columns])` 直接复用 slot 显存，输出同样落在 slot 上，`pack_row`/`unpack_row`
两个 kernel 连同每个投影 64 次逐 lane 拷贝一并删除。辅助算子按 `dispatch32` 分派：输入全部
非常量的 Norm/GatedNorm/Split 与 unary/binary 在整个 `[32 * size]` 上只录一次（Batched）；
Embedding/RoPE/Conv/GDN/Attention 仍逐 lane（Row），但输入改为 slot 的安全行视图，输出经
`Tensor::from_foreign` 行张量直接写回 slot。cuTile 0.4 没有可写行视图的 safe API，该 unsafe
边界已登记进 `tools/check/policy.py` 白名单并带 `#[expect(unsafe_code)]` 与 SAFETY 注释；
slot 由 `BatchGraph._arena`
常驻锚定，字段声明在最后以保证引用它的图先 drop。hidden/logits 回读增加 skip/len 区间：
一次 pinned D2H 读回 `[0, tokens * hidden)` 与末行 logits，host 侧再切行。width-3 验证路径
与 `linear_batch.rs` 未动；decode/MTP verify 仍走原有逐 lane arena（Flat 模式）。

验证证据：

- `artifacts/batch32-resident.log`：设备图 63 项数值检查全部 PASS（含 prompt32 对 CPU 参考
  3e-4 容差、Conv/GDN、分组 attention、FP8 KV、尾块、verify 前缀回退）。
- `artifacts/batch32-provider-f32.json` / `batch32-provider-fp8.json`：真实 27B F32 与
  FP8 KV + 池压回归均生成 `391`。
- `artifacts/batch32-mtp2.json`：prefill32 + MTP2 生成 `[18, 24, 16, 248046]`，decode 统计
  逐字段与改造前一致（接受 2、拒绝 1、恢复 2）。
- pilot 数据集（8 请求、822 decode token）：用恢复全部改动前的源码重建基线二进制，同配置
  各跑一次，两条二进制的 decode token 序列 **822/822 逐位一致**。

| 指标（pilot 合计） | 改造前 prefill32 | 批量 arena | 变化 |
|---|---:|---:|---:|
| prefill 耗时 | 1.562 s | 0.767 s | -50.9% |
| 请求墙钟 | 16.510 s | 15.544 s | -5.9% |
| decode tok/s | 49.79 | 52.88 | +6.2% |

[验证摘要](../../benchmarks/capabilities/cuda-batched-arena.json) 记录两次报告/二进制哈希、
硬件遥测路径与逐 token 比对范围。prefill32 把 prompt 投影输入舍入到 BF16，与 prefill3 的
F32 GEMV 之间的固有差异不受本项影响，两者仍不能声称逐位等价。每条 32 token 的 32 次 metadata
更新和 Row 模式辅助 kernel 仍逐 lane 串行；8 请求 pilot 不是正式吞吐验收，vLLM 基线尚未
因本项更新。
