# 纯 GEMM 完整质量、Split-K 成本与 FP4 库能力（2026-10-13）

两轮整模型测试确认，补齐实际权重的 M4/M256 调参表没有带来明显端到端收益。
完整串行 256 题的所有生成 token 与原 WY 相同。细分 N 和当前 Split-K 原型也未在
主要 NVFP4 投影上通过冷热 AB/BA 性能门槛；实验代码未进入 main。

新增独立验证确认：45 的 cuBLASLt 可用 NVFP4 输入直接生成 F32 输出，
四种几何全部通过完整量化乘积的 CPU F64 参考。
这提供了下一项实现方向，但当前只完成库能力诊断，尚未证明完整量化图或 serving 加速。
性能逼近 vLLM 的目标仍未达到。量化模型默认仍是 scalar，本文整模型数据显式开启原 WY。

## 纯实际权重 GEMM 表的整模型结果

旧 CLI 与 GEMM 候选使用同一个 release binary，SHA256
`4be5d78d71c995e773c9571b0bf23fadb57796d8f5602d8440`。

候选只补充此前两个成功实际权重调参 pass 的 M4/M256 GEMM 决策，
GDN 表保持原样，不包含失败的宽图半列 GDN、并行 Prepared 或新 N32 候选。
原 cache SHA256 为 `3a1a22bb5ca4ea7d8499a92d50ed52637b1ab7092f91a03b80d7ef7bc551ef74`，
候选为 `29155665684f387f080dae6b6452b47e87c055632de31f2646668227b06af498`。

两轮进程顺序为 old/GEMM/vLLM 和 vLLM/GEMM/old，所有六项后台资格检查通过。
每轮每场景三次测量，以下取两轮进程中位数的中位数，单位 ms。
实际输入 51/511 token，四并发各 52 token；MTP=0，每路生成 64 token，prefix reuse=0。
vLLM 运行时继续冻结为官方 3fe917abb，不能标作另一个源码 main 的构建。
这是两进程轮次诊断，未达到五轮性能验收。

| 指标 | 原 WY | GEMM 表候选 | vLLM | 候选相对原 WY 耗时变化 |
|---|---:|---:|---:|---:|
| 短提示 TTFT | 37.472 | 38.467 | 49.120 | +2.655% |
| 短提示 TPOT | 16.180 | 16.125 | 14.968 | -0.339% |
| 短提示 wall | 1057.533 | 1054.302 | 992.551 | -0.305% |
| 长提示 TTFT | 179.665 | 175.079 | 59.997 | -2.553% |
| 长提示 TPOT | 16.162 | 16.136 | 15.013 | -0.166% |
| 长提示 wall | 1199.054 | 1192.458 | 1006.641 | -0.550% |
| 四并发 TTFT | 146.575 | 151.155 | 87.063 | +3.125% |
| 四并发 TPOT | 19.523 | 19.515 | 16.410 | -0.041% |
| 四并发 wall | 1380.614 | 1385.367 | 1122.444 | +0.344% |

长提示首 token 单轮分别降低约 5.3% 和增加约 0.3%，方向并不稳定。
候选长 TTFT 仍约为 vLLM 的 2.918 倍，单请求解码约慢 7.5%，四并发 wall 约慢 23.4%。
不将不到 1% 的整请求变化描述为明显性能进展，也不采用整个缓存作为新默认。

p0 的 batch4 有两请求 token 不同；原 WY 的 p0/p1 之间同样两请求不同。
候选两轮及原 WY p1 则一致。这不足以把差异归因为 GEMM 表，也不足以认证并发数值行为。

## 原 256 题完整串行质量

原输入、原 plan、自然 EOS、max512、temperature=0、thinking off、MTP=0，
没有删除 Length 或缺答案请求。全部 256 个结果完成，prefix reuse 为零，
CLI/cache 前后不变，后台 GPU compute 保留且不变。
独立 `score-task-quality.py` 从完整原始文本复算原 2% 与单侧 95% 门槛。

| 基线 | 基线答对 | 候选答对 | 回退 | 新增答对 | 回退率上界 | 验收 |
|---|---:|---:|---:|---:|---:|---|
| scalar | 240 | 243 | 1 | 4 | 1.840% | pass |
| 原 WY | 243 | 243 | 0 | 0 | 1.163% | pass |
| vLLM | 243 | 243 | 4 | 4 | 3.540% | not_demonstrated |

候选与原 WY 所有 256 请求的 token 完全一致，不只正确题数相同。
相对 vLLM 的逐题回退仍未满足门槛。此完整 suite 的并发上限是 1，
因此它验证串行 M1/M256 路径，不能用来证明新增 M4 决策的并发模型质量。

原始结果 SHA256 `658fc3db39a19823696097fd1b1b7822e9c96398df8b75078bfe77692e2e380f`；
独立质量汇总 SHA256 `c3327e67c2850bfa9375a8018cb8aaa551d2e68d35224d0e6ef1e616a2dda7da`。
输入 suite 为 `adb429cdc275400fef3c3e6331b942a38237b2d5ca07f4aad5948ebf06fb94e6`，
plan 为 `991ac5b21d7862270477e561aaa9e6fd1a5803626e014870890fbad60f416c83`。

## N32 和 Split-K 的完整图成本

N32 加入原硬件 tuning 候选集，保留量化与 K 顺序，仅在隔离源码测试。
NVFP4 三个大投影均选择原 baseline。FP8 `[4,1024,5120]` 的冷/暖反序确认
分别为 1.040/1.077 倍，另两个较大 FP8 投影仍选择 baseline。
这里使用合成权重，不能宣称实际模型收益。N32 源码和样本已归档，未合入。

Split-K 原型只编译在测试模块。保留原量化和 256 元素 scaled MMA，
把 K 分为 1/2/4/8 个窗口，先写 F32 partials，再按窗口顺序相加并乘原 alpha。
完整 CUDA graph 包含量化、全部 partial 写入和 reduce，临时显存全部列入报告。
三种尾部几何、两种输入 fixture，加三种实际尺寸几何共 36 个 case，
全部输出通过独立 CPU F64 量化乘积参考，归一化门槛沿用 NVFP4 独立参考的 1e-4。
本组合成输入最大误差 5.910e-8；这不能保证实际 checkpoint 或完整模型质量。

下面只列 split=2 的新鲜反序确认，单位微秒；所有大尺寸都另有正序与其他 split 的原始样本。

| M/N/K | 冷原图 | 冷 Split-K | 暖原图 | 暖 Split-K | 临时字节 |
|---|---:|---:|---:|---:|---:|
| 1/5120/17408 | 40.976 | 43.424 | 24.448 | 23.808 | 655360 |
| 4/5120/17408 | 41.584 | 44.448 | 25.888 | 22.912 | 655360 |
| 4/17408/5120 | 41.728 | 47.456 | 28.416 | 29.184 | 2228224 |

三类大投影的全部 1/2/4/8 候选都未同时通过冷热 AB/BA 至少 2% 的门槛。
部分 warm 改善不能掩盖 cold 退化；更大的 partial 数还增加写回与归并。
这否定当前实现作为通用候选，并未证明所有 Split-K 算法或其他硬件均无效。

严格 release/CUDA all-target clippy、测试构建及 53 个普通测试通过，
GPU 完整数值与成本测试通过。初次编译所有权错误及首次 JIT 名称冲突的日志都保留；
失败尝试没有产生有效性能记录。源码、binary、可应用 patch 已归档，开发树恢复为 main。
通过测试的 binary SHA256 `d800f738a1c2b00ffe20dad36b4dc187508a458ef7249e3c2cb6db0c4506f5f2`。

## cuBLASLt NVFP4/F32 能力验证

检查本机 CUDA header 和 [NVIDIA cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)，
使用 documented TN、VEC16_UE4M3、F32 compute/output，按官方 128x4 scale 布局转换。
实际加载的 Lt version 为 130800，不能用 header 或文档版本代替运行库身份。

| M/N/K | F32 输出与完整 CPU F64 参考 | 最大归一化误差 | workspace 字节 |
|---|---|---:|---:|
| 16/128/512 | pass | 5.390e-8 | 0 |
| 1/5120/17408 | pass | 4.419e-8 | 184320 |
| 4/5120/17408 | pass | 5.215e-8 | 737280 |
| 4/17408/5120 | pass | 5.215e-8 | 0 |

真实调用与 CUDA graph replay 均成功，后台启动/捕获/结束检查一致。
warm bare graph 样本包含在原始结果，但排除了激活量化和 layout 转换，
因此不与另一进程的完整 native graph 相除，不登记服务加速倍数。
当前仍是合成、已经量化的 operands；真实权重、完整量化图和整模型尚未验收。

下一步应把原量化、动态 scale 布局转换、Lt matmul 及必要 workspace 纳入同一张图，
用同一实际权重测完整冷热 AB/BA。候选只有在数值与模型质量通过后，
才能接入硬件 tuning 和资源预算；库版本、driver、shape、精度、layout、workspace
都参与身份，缺库或不支持的设备保留原 fallback，不按 5090 名称选择。

## 原始证据

共同根目录：
`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/`。

| 目录 | 内容 |
|---|---|
| `gdn-gemm-cache-only-perf-v1` | 六进程完整报告、cache manifest、资格检查与两轮汇总 |
| `gdn-gemm-cache-only-quality-v1` | 原 256 题完整结果、独立三组评分、token 对照与执行身份 |
| `gdn-gemm-column32-v1` | 两次普通验证、NVFP4/FP8 全输出门槛与六种合成尺寸完整图实验 |
| `gdn-nvfp4-split-k-probe-v1` | 失败/成功构建与 JIT、36 个独立参考 case、完整图样本及源码归档 |
| `gdn-nvfp4-lt-capability-v1` | 官方 header ABI、库身份、四种几何完整独立参考与图能力验证 |

Lt 结果 SHA256 `aa557647ec68a7e0c760c94fb3cd69cf3bc46577233100504d38f6e0f8b5c0d4`，
执行 Python 源码 SHA256 `ab48157ceca924cfff48d01b3730b43b2837550730d758954d89a273ea248578`。
后续工作沿用[性能基线与 vLLM 对齐方案](../plans/performance/baseline.md)，
并保留[并行 WY 完整质量失败](cuda-parallel-wy-quality-and-decode-cost-2026-10-13.md)的限制。
