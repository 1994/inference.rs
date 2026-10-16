# GEMM 完整成本、五轮量化复用与大批块 Lt（2026-10-13）

两轮整模型测试确认，补齐实际权重的 M4/M256 调参表没有带来明显端到端收益。
完整串行 256 题的所有生成 token 与原 WY 相同。细分 N 和当前 Split-K 原型也未在
主要 NVFP4 投影上通过冷热 AB/BA 性能门槛；实验代码未进入 main。

新增独立验证确认：45 的 cuBLASLt 可用 NVFP4 输入直接生成 F32 输出，
四种几何全部通过完整量化乘积的 CPU F64 参考。
后续 Rust 完整图已测原量化、动态 scale 布局转换与 Lt matmul，八种 SDK 算法共 72 个 case
全部通过独立参考，但三个主要尺寸没有候选同时通过冷热 AB/BA 至少 2% 的门槛。
上述小批量解码实验没有新增 serving 收益。早期 Python 裸图存在 capture 流错误，其性能计时已作废。
追加的静态量化复用五轮 serving 对比，长提示 TTFT 中位数下降约 2.5%，逐轮方向不稳定，
解码变化约 0.3%，四并发 wall 未改善。追加的 M256 完整 Lt 图在合成输入上通过数值参考，
gate/up 约加速 1.6 倍、down 约 1.3 倍，尚未接入模型或完整质量验收。
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
这一阶段仅确认 native 分别量化，checkpoint 的 56 对 NVFP4 gate/up 输入全局尺度逐字相同。
后续静态复用与大批块测试结果见下文。

## 静态相邻投影量化复用：算子通过，五轮模型收益很小

checkpoint 的 56 对 NVFP4 gate/up 输入尺度相同。实验仅在相邻两个 Linear 使用同一个
SSA 输入、相同 F32 全局量化尺度且不写 state 时，跳过第二次量化；
排除可能被 capture 省略的 logits 前驱，不使用重放时的指针缓存。
FP8、不同输入/尺度、间隔其他节点均保留原行为。这个实验不增加 GPU 工作区或硬件参数，
默认关闭；不是已合入 main 的优化。

五种尾部几何乘三种模式，加三种主要尺寸，共 18 个 GPU case。
每张图重放 signed、新输入含零、全零三种输入，所有输出 finite 且与原图 bitwise 相同，
全零输入清除先前量化结果；不同输入或尺度确实量化两次。
同时原量化器/MMA 的独立完整参考测试通过，严格 CUDA release all-target clippy、
构建及 56 个普通测试通过。下面是同进程两次完整投影的反序结果，单位微秒。

| M/N/K | 冷原图 | 冷复用 | 暖原图 | 暖复用 |
|---|---:|---:|---:|---:|
| 1/17408/5120 | 76.928 | 75.776 | 71.360 | 70.288 |
| 4/17408/5120 | 78.448 | 76.848 | 71.728 | 70.736 |
| 256/17408/5120 | 200.816 | 185.728 | 198.080 | 185.936 |

大 M 的双投影图约加速 6–8%，解码双投影只约 1–2%，不能当作整模型收益。
接着用同一个冻结 release binary、相同原 WY/GEMM cache，仅切换实验开关，
与冻结 vLLM 3fe917abb 做五轮交错对比。顺序为 old/reuse/vLLM、vLLM/reuse/old、
reuse/old/vLLM、old/vLLM/reuse、reuse/vLLM/old。
51/511 token 提示、四路各 52 token，MTP=0，每路 64 个输出，prefix reuse=0；
启动与三次预热均排除，每轮每场景三次测量，共 15 次场景测量。
全部十五个进程的后台 GPU 状态和实际实验开关检查通过。
下表取全部非预热样本中位数，单位 ms；TTFT/TPOT 的四并发样本为每个请求，wall 为组 makespan。

| 指标 | 原 WY | 量化复用实验 | vLLM | 实验相对原 WY 耗时变化 |
|---|---:|---:|---:|---:|
| 短提示 TTFT | 35.252 | 35.303 | 49.940 | +0.144% |
| 短提示 TPOT | 16.166 | 16.102 | 14.972 | -0.396% |
| 短提示 wall | 1054.031 | 1053.274 | 993.922 | -0.072% |
| 长提示 TTFT | 180.735 | 176.227 | 58.559 | -2.494% |
| 长提示 TPOT | 16.132 | 16.092 | 15.025 | -0.250% |
| 长提示 wall | 1198.844 | 1191.813 | 1005.430 | -0.586% |
| 四并发 TTFT | 148.766 | 148.477 | 84.421 | -0.194% |
| 四并发 TPOT | 19.405 | 19.351 | 16.424 | -0.277% |
| 四并发 wall | 1371.478 | 1374.455 | 1122.570 | +0.217% |

长 TTFT 每轮下降分别为 4.692%、4.041%、-0.155%、-0.207%、0.966%，收益并不稳定。
相对 vLLM，候选长 TTFT 仍为 3.009 倍，单请求解码约慢 7%，四并发 wall 约慢 22.4%。
完整质量 suite 尚未运行，不采用这一实验作为默认，也不把算子收益报告为模型提升。

所有串行 short/long 的 token 在两模式及五轮之间一致。
前三轮 old/reuse 的 batch4 分别有 4/1/1 请求 token 不同，后两轮一致；
old 自身和 reuse 自身跨轮也有 batch4 token 变化，因此不能仅据此归因于复用，
也不能宣称并发数值已验收。独立数值通过不替代完整质量及并发正确性检查。

冻结 CLI SHA256 为
`49fb1a6b22ec0a30455b74d0d4bb25654ee97f9da2bb33923b53cdcce294c1e2`；
源码基于 9f36949a0709，manifest 明确 `source_dirty=true`。
原 cache 保持 `3a1a22bb5ca4ea7d8499a92d50ed52637b1ab7092f91a03b80d7ef7bc551ef74`。
GPU runner 的旧 Lt 范围标签来自脚本复制，保留原记录；
`gdn-quantization-reuse-v1/summary-measured.json` 明确纠正实际执行范围为原 NVFP4 独立参考与量化复用重放。

## M64/M256 提示批块的完整 Lt 成本

小批量解码的淘汰结果不能推断大 M 提示块。追加三种 prefill 几何、每种最多请求八个
SDK 候选，使用 `GemmConfig::baseline(rows)` 的 M64/N64 Plain 原生图，
不是先前解码测试的 M16/N64。候选仍包含原量化、动态尺度布局转换、Lt 与 F32 输出，
权重尺度转换在 capture 前准备；完整 CPU F64 量化乘积参考只按相同合成 fixture 计算一次，
随后逐个核对每张图的全部输出，数值和布局门槛未改变。

首次尝试将“SDK 无第八个候选”当作测试失败；现明确记录 typed unavailable，
实际数值、capture 或运行错误仍直接失败。三类几何的全部 24 个查询槽位都有记录，
其中 M256 gate/up 只返回 7 个可用候选，另外两个各 8 个。
23 个可用完整图全部通过独立参考，全部候选的最大归一化误差为 2.954e-5，
低于原 1e-4 门槛；部分算法与 native 非 bitwise 相同。
下表选出的三个候选最大误差为 5.873e-8，均与 native bitwise 相同。
上一版误把这三个最优候选的数值范围扩展为全部候选，现按完整原始记录纠正；
一个不可用候选没有性能或数值结果。
每个可用图均保留冷热 AB 与新鲜 BA 各 30–100 样本，后台 GPU 状态一致。

以下每行选择四种测量中最弱加速比最大的候选，展示反序数据，单位微秒。

| M/N/K | 算法序号 | 冷 native | 冷 Lt | 暖 native | 暖 Lt | workspace 字节 |
|---|---:|---:|---:|---:|---:|---:|
| 64/17408/5120 | 0 | 47.792 | 47.584 | 38.272 | 40.192 | 0 |
| 256/17408/5120 | 0 | 106.160 | 67.632 | 108.048 | 65.792 | 0 |
| 256/5120/17408 | 1 | 143.088 | 108.240 | 151.568 | 108.128 | 20971520 |

M256 gate/up 的四项加速比为 1.583/1.639/1.570/1.642，down 为 1.317/1.398/1.322/1.402；
均通过四项至少 2% 的完整图门槛。M64 没有通过四项门槛的候选。
原 serving cache 没有 M256 GEMM 条目，M256 正好使用本实验的 baseline；
M64 gate 则有更强的 N128/Prefetch8 调优项，不能用此较弱 M64 基线认证部署。

这仍是固定全局尺度 1.25/0.75 的合成数据，没有实际 checkpoint 或完整模型质量认证，
没有 serving 新结果，也没有注册 production tuning 决策。
实际接入时需检查原始权重全局尺度，并将库/driver/shape/布局/精度/算法 config 和 workspace
纳入硬件自动测量的缓存身份。每个大投影额外权重 scale 布局为 5570560 字节；
down 候选还需 20 MiB workspace。若逐投影永久分配 workspace 会放大峰值，
应验证序列执行的共享 scratch 生命周期，并计入模型准入，不能只比较 matmul 耗时。

严格检查、构建与普通测试通过，全部可用 GPU 测试通过；
binary SHA256 为 `2714b7c239af79f2fa44511b8ab0998fd3966603e73bf19f530a7768afcdd018`。
源代码、可应用 patch 及完整样本已可恢复归档，未将未验收 runtime 代码合入 main。
下一步优先实际权重和整模型验证大 M 库路径；重复量化的小收益不足以解释剩余差距。

## 实际 checkpoint 权重：独立量化与完整乘积

后续直接提取 Qwen3.8-27B-NVFP4 的 layer 0/55 gate/down packed 权重、全部 block scale，
以及各自真实的 input/weight global scale，冻结原始视图偏移、dtype、shape 和 SHA256。
输入仍是两种确定性 signed/zero fixture，未把算子测试标作模型推理。
原生基线按设备完整 profile、M/N/K 读取与 serving 相同的原 cache，
M64 gate 实际使用 N128/Prefetch8；M256 无原条目，使用 M64/N64 Plain fallback。

参考不再使用 GPU 量化结果作为输入。CPU 按公开 FP4/E4M3 码表独立编码，
保持原 F32 乘除顺序和 ties-to-even，逐字节核对全部 GPU input code/scale；
再用原 checkpoint 字节独立解码，计算全部输出的 CPU F64 乘积，最后乘相同 F32 alpha。
每个固定输入/权重/shape 只生成一次独立参考，所有候选分别核对完整输出和量化字节。

四种真实权重、M64/M256、两种 fixture、每组最多八个 SDK 候选，共 128 个查询槽位。
124 个可用完整图全部通过，另外 4 个 M256 gate 的第八个候选明确 unavailable。
全部候选最大归一化误差为 4.897092e-06，低于原 1e-4 门槛；
44 个候选与 native 非 bitwise 相同，不能宣称模型 token 不变。
每个可用候选保留四组冷热 AB/BA 原始样本，asset/source/binary 前后不变，
两次完整 GPU 测试的后台状态一致。严格 CUDA release all-target clippy、构建与 53 个普通测试通过。

同一算法要在两种 fixture 的全部八项测量中均至少加速 2%，才算通过算子性能门槛。
下表按八项最弱加速比最大选择候选；算法身份用完整 typed config 保存，SDK 序号不作为缓存身份。

| 实际权重 | M | 最佳候选在八项测量中的最小加速比 | workspace 字节 |
|---|---:|---:|---:|
| 0.mlp.down_proj | 64 | 0.941 | 5242880 |
| 0.mlp.down_proj | 256 | 1.290 | 20971520 |
| 0.mlp.gate_proj | 64 | 0.946 | 0 |
| 0.mlp.gate_proj | 256 | 1.595 | 0 |
| 55.mlp.down_proj | 64 | 0.952 | 5242880 |
| 55.mlp.down_proj | 256 | 1.285 | 20971520 |
| 55.mlp.gate_proj | 64 | 0.945 | 0 |
| 55.mlp.gate_proj | 256 | 1.617 | 0 |

M256 的两个 gate 权重约加速 1.6 倍，两个 down 约 1.29 倍或更高；
M64 全部未通过完整门槛。只适合通过实际硬件/shape 测量选择，不能统一切换到库实现。
最优 down 需要 20 MiB workspace，全部大投影另有 5570560 字节的 scale 布局副本，
仍需共享 scratch 与准入预算，不能按层重复永久分配。

这些结果支持继续运行时接入，尚未做新 CLI 的完整模型质量或 serving 对比。
现有所有显著收益仍限定在 GEMM 算子，本文未增加已合入的模型性能声明。
验证 binary SHA256 为 `9b04846ba8181a8d9a966c455f9307c7cd4f9ccd60ed3adc1c429ec7fe897de7`；
四视图资产 manifest SHA256 为 `6bed9c5bff64696b5b1d8d5b07ac0db677fed063d992f52ab7f8619c2ddc5f7a`。
全部源代码与可应用 patch 已冻结归档，本次发布只更新文档，开发树恢复 main。
实验 FFI helper 放在 tests 目录，未满足仓库 unsafe 边界策略，不能据此宣称原型可直接合入；
运行时接入须放入有所有权和上下文校验的 device 边界。

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
| `gdn-quantization-reuse-v1` | 静态 SSA 复用、18 个重放 case、独立原量化参考、完整图成本及范围标签纠正 |
| `gdn-quantization-reuse-runtime-v1` | 冻结 dirty diagnostic CLI、源码/hash 与 release 构建身份 |
| `gdn-quantization-reuse-perf-v1` | 五轮十五进程、全部测量报告、开关/后台资格及 token 变化 |
| `gdn-nvfp4-lt-prefill-complete-graph-v1` | M64/M256、23 个可用候选和一个 unavailable、完整参考/成本、失败尝试与源码归档 |
| `gdn-nvfp4-lt-checkpoint-graph-v1` | 四个真实权重视图、CPU 独立量化、124 个完整图和四个 unavailable、原 cache 基线与联合选择 |

原 Lt 数值结果 SHA256（其裸图计时作废） `aa557647ec68a7e0c760c94fb3cd69cf3bc46577233100504d38f6e0f8b5c0d4`，
执行 Python 源码 SHA256 `ab48157ceca924cfff48d01b3730b43b2837550730d758954d89a273ea248578`。
纠错结果 SHA256 `aaac544fe9d3151914284c5136ac65fa9bf025a19372759a475200257328efa9`；
全算法执行 SHA256 `32a725dce895ea4370cdbbb6d956ad9b031dcaad95b34e094250438da8999629`。
后续工作沿用[性能基线与 vLLM 对齐方案](../../plans/performance/baseline.md)，
并保留[并行 WY 完整质量失败](cuda-parallel-wy-quality-and-decode-cost-2026-10-13.md)的限制。
