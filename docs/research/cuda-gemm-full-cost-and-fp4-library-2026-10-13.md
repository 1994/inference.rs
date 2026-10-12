# 纯 GEMM 质量、Lt 完整图成本与裸图计时纠错（2026-10-13）

两轮整模型测试确认，补齐实际权重的 M4/M256 调参表没有带来明显端到端收益。
完整串行 256 题的所有生成 token 与原 WY 相同。细分 N 和当前 Split-K 原型也未在
主要 NVFP4 投影上通过冷热 AB/BA 性能门槛；实验代码未进入 main。

新增独立验证确认：45 的 cuBLASLt 可用 NVFP4 输入直接生成 F32 输出，
四种几何全部通过完整量化乘积的 CPU F64 参考。
后续 Rust 完整图已测原量化、动态 scale 布局转换与 Lt matmul，八种 SDK 算法共 72 个 case
全部通过独立参考，但三个主要尺寸没有候选同时通过冷热 AB/BA 至少 2% 的门槛。
本轮没有新增 serving 收益。早期 Python 裸图存在 capture 流错误，其性能计时已作废。
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

### 裸图计时纠错

早期 Python 原型在 capture 前缓存了 Torch CUDA stream，capture 内的 Lt 调用仍提交到旧 stream。
专项复现读取实际 CUDA graph 节点，并在重放前清空输出：四种几何的旧 capture 都是 **0 节点**，
重放后的非零输出数也为 0。因此旧结果里的约 3–4 微秒不是 GEMM 计时，全部作废，
也撤回旧文档对该原型 graph 计算重放成功的断言。capture 前真实 Lt 调用的独立数值结果仍有效。

修正版在调用时获取当前 capture stream。四张新图分别包含 1/2/2/1 个节点；
清空输出后重放，全部非零输出恢复，并再次通过完整 CPU F64 参考。
修正后的裸 warm 中位数分别是 4.640/18.416/19.072/45.936 微秒。
这些数字仍排除量化/layout，来自独立进程，不与 Rust 完整图相除，也不登记模型收益。
专项执行的后台启动/捕获/结束检查全部一致。

## Rust 完整量化/layout/Lt 图

原型只在 cfg(test) 下复用已有 Lt ABI 和 CUDA stream/context 检查，F32 输入及输出保持不变。
同进程 baseline 保留原量化与 M16/N64、K256 scaled MMA；候选保留同一个量化器，
另记录 GPU scale swizzle 和 Lt matmul。权重 scale 布局在 capture 前准备，动态输入转换全部计时。
固定量化全局尺度 1.25、权重尺度 0.75，使用原 alpha；尚未使用实际 checkpoint。
描述符、workspace、输入输出及其 scale 的所有者保留到 graph 销毁后；必要初始化在 capture 前完成。

首次 GPU 尝试发现不足 32 行时，swizzle 的无效输入 tile 被访问。
修复后跳过无效 32 行块，保留初始化的零 padding，并逐字节验证所有活动尺度及全部 padding。
失败日志保留，没有使用该尝试的性能数字。严格 release/CUDA all-target clippy、
构建及 53 个普通测试通过，修复后的首个算法 9 个 GPU case 全部通过，
binary SHA256 为 `e64dd6372a8994d62e6d7e493eaf2cdc7fc55c0d1e96c87e0ebe77f8b9355aa9`。

随后把 SDK 返回的八种算法逐个纳入相同完整图，未按设备型号选算法。
三类尾部形状、两种 signed/zero fixture 和三类主要尺寸，合计 **72 个 case**，
所有输出通过独立 CPU F64 完整量化乘积参考，最大归一化误差 5.215e-8，低于原 1e-4 门槛。
本组合成数据也全部与 native 输出 bitwise 相同，不能由此认定真实权重或完整模型质量。
24 个大尺寸/算法组合均测冷/暖 AB 和新鲜 BA，各保留 30–100 个原始样本。

以下只展示每个尺寸在两种顺序下冷缓存较好的候选及其反序数据，单位微秒。
算法序号指当前 SDK 返回顺序，不是可以持久化的算法身份；完整记录保留 typed config 和 workspace。
这张表展示淘汰依据，不把冷最优候选当作已采用配置。

| M/N/K | 算法序号 | 冷 native | 冷 Lt 完整图 | 暖 native | 暖 Lt 完整图 | workspace 字节 |
|---|---:|---:|---:|---:|---:|---:|
| 1/5120/17408 | 2 | 40.768 | 45.728 | 24.768 | 33.808 | 143360 |
| 4/5120/17408 | 2 | 41.056 | 46.160 | 26.528 | 31.488 | 573440 |
| 4/17408/5120 | 5 | 41.648 | 50.144 | 28.256 | 44.576 | 835584 |

八种候选的冷缓存完整图都没有胜过 native；所有尺寸都无候选同时通过冷热 AB/BA 至少 2%。
前两个尺寸的首个算法暖缓存加速比约为 1.03/1.14，同时冷缓存耗时增加约 14%/16%，仍未通过门槛。
这否定当前三节点 Lt 原型的直接采用，不代表其他布局、融合实现或其他硬件上的 Lt 均无效。
额外权重 scale 布局每个大投影为 5570560 字节，动态输入 scale 布局及 workspace 也逐项记录；
本轮没有向正式资源预算或 tuning 注册一个未经验收的候选。
若后续融合实现通过门槛，仍由硬件完整图测量选择；库版本、driver、shape、精度、layout、
workspace 都进入缓存身份，并计入额外显存，缺库或设备不支持时使用原 fallback。

全算法 binary SHA256 为 `5f8c9ff2fda5ad1698b91b35152271067756d8c1367f68b336baa61033ef07d6`。
严格检查、普通测试及全部 GPU 测试通过，后台 GPU 进程始终保留且状态一致。
源码、完整样本、可应用 patch 均归档，实验代码没有合入 main。
后续更有必要检查可消除的重复量化和线性层图节点；目前仅确认 native 分别量化，
checkpoint 的 56 对 NVFP4 gate/up 输入全局尺度逐字相同，还未证明可以安全跨节点复用或带来性能收益。

## 原始证据

共同根目录：
`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/`。

| 目录 | 内容 |
|---|---|
| `gdn-gemm-cache-only-perf-v1` | 六进程完整报告、cache manifest、资格检查与两轮汇总 |
| `gdn-gemm-cache-only-quality-v1` | 原 256 题完整结果、独立三组评分、token 对照与执行身份 |
| `gdn-gemm-column32-v1` | 两次普通验证、NVFP4/FP8 全输出门槛与六种合成尺寸完整图实验 |
| `gdn-nvfp4-split-k-probe-v1` | 失败/成功构建与 JIT、36 个独立参考 case、完整图样本及源码归档 |
| `gdn-nvfp4-lt-capability-v1` | 原始真实调用数值能力；裸图计时因空 capture 作废 |
| `gdn-nvfp4-lt-capability-stream-fix-v2` | 空图专项复现、节点计数、正确流 capture 和清空输出后重放验证 |
| `gdn-nvfp4-lt-complete-graph-v1` | Rust 首个算法完整图、失败/成功尝试、独立参考与 AB/BA |
| `gdn-nvfp4-lt-heuristics-complete-graph-v1` | 八种算法、72 个独立参考 case、24 组完整图成本及源码归档 |

原 Lt 数值结果 SHA256（其裸图计时作废） `aa557647ec68a7e0c760c94fb3cd69cf3bc46577233100504d38f6e0f8b5c0d4`，
执行 Python 源码 SHA256 `ab48157ceca924cfff48d01b3730b43b2837550730d758954d89a273ea248578`。
纠错结果 SHA256 `aaac544fe9d3151914284c5136ac65fa9bf025a19372759a475200257328efa9`；
全算法执行 SHA256 `32a725dce895ea4370cdbbb6d956ad9b031dcaad95b34e094250438da8999629`。
后续工作沿用[性能基线与 vLLM 对齐方案](../plans/performance/baseline.md)，
并保留[并行 WY 完整质量失败](cuda-parallel-wy-quality-and-decode-cost-2026-10-13.md)的限制。
