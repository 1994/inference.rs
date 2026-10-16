# F32 GDN 数值定位与完整模型质量验收

日期：2026-10-12。源码基点 `989bb6b94c522098dc7cadd004a72aa506f18e68`，开发和 GPU 验证位于
`r@192.168.1.45`。用户确认的验收方向为保持 F32 recurrent state，使用独立数值参考与完整模型任务质量。
Neumann WY 候选已完成完整模型质量测试并失败，未进入主分支生产调度。
分块三角求逆及 F32 操作数精度修复已通过相对原实现的完整 256 题质量门槛；固定工作量服务对照的长提示 TTFT 降低 26.06%，仍未达到 vLLM 目标。
后续自动列宽候选的质量资格另行记录，不能用独立 kernel 误差通过代替完整模型验收。

## 单步阶段定位

使用同一份 F32 输入、初始状态和 `KH=16, VH=48, D=128, LANES=1, count=1`，
分别运行未修改的标量和 chunk entry，再通过独立 entry 导出 14 个阶段。
每个探针同时比较最终 state/output 与未修改 entry，以发现观察本身对编译结果的影响。

| 观察 | 结果与适用范围 |
|---|---|
| 原始 Q/K、平方和、beta、decay | 导出值一致，两边最终控制结果也一致 |
| prediction | 首次可信的非零阶段差异，max abs `8.9406967e-8`；两边最终 state/output 保持各自未修改控制结果 |
| delta | max abs `5.9604645e-8`，控制结果保持 |
| 未修改 state | 59,412 / 786,432 元素逐位不同，max abs `2.9802322e-8` |
| 未修改 output | 4,634 / 6144 元素逐位不同，max abs `7.4505806e-9` |
| normalized Q/K、部分 state/readout 探针 | 改变了一侧或两侧控制结果，不能据此断定生产中间值 |

这一定位支持浮点求值差异的解释，尚未证明具体的 FMA contraction 或 reduction lowering。
独立主机 F64 标量递推从原始 F32 字节构造参考，使用原有逐元素 `max_abs < 2e-5` 门槛：

| Entry | state max abs / relative L2 | output max abs / relative L2 |
|---|---|---|
| scalar | `6.8955014e-8 / 7.0692495e-8` | `8.6710194e-9 / 1.4662978e-7` |
| chunk | `6.8955014e-8 / 7.0739752e-8` | `1.1462435e-8 / 1.4775889e-7` |

仅此单步结果不代表 64 层量化模型的任务质量合格。之前完整模型的 hidden/logit 漂移和生成差异仍保留。

## WY 独立参考矩阵

复用既有 32-token WY 候选：严格下三角 interaction 的有限 Neumann inverse，
三项 BF16 residual MMA 与 F32 累加，state 保持 F32。
本轮只提取 GDN 候选，未带入旧实验的 attention 或其他修改。

独立 F64 递推覆盖 `(KH,VH,D,LANES)` 为 `(2,4,32,32)`、`(2,6,128,32)`、
`(2,6,128,128)`、`(16,48,128,128)`。
每个形状包含空块、单 token、masked tail 与负起始 offset，两种 value split。
16 组形状/边界、32 次 kernel 执行、64 项 state/output 检查全部通过原有 `2e-5` 上限，
最大 state 误差 `1.6527631e-6`，最大 output 误差 `4.7070655e-7`。

原有 split 末位差异单独记录。本轮没有把它当作独立 oracle，也没有放宽 F64 的逐元素误差上限。
MoE、实际 checkpoint 输入的长递推误差及跨设备范围仍需各自证据。

## 完整模型质量协议

使用 27B 完整模型，三个 arm 顺序运行：同一实验二进制的 scalar control、WY prefill，
以及固定官方 `3fe917abb13160d60bea5330399f5c09ca282348` vLLM 发行构建。
两种 native arm 只改变 prefill Delta；decode 保持原标量路径，MTP=0。
native activation 规则保持 F32，vLLM 使用其 BF16 activation 路径，两个 arm 的 recurrent state 都是 F32。

题集在模型运行前固定：官方 [GSM8K](https://github.com/openai/grade-school-math) 的
`3101c7d5072418e28b9008a6636bde82a006892c` 测试集，源 SHA256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
排除先前 pilot 行，seed=20261012，选择 256 题；128 道零例提示为 96–205 token，
128 道八例提示为 1769–1884 token。训练示例另以固定 seed 选择，并保留行号。

复用模型自身 tokenizer/template，关闭 thinking，temperature=0，自然 EOS，512 token 输出上限。
每题独立标识放在共享示例之前；native 每个请求必须实际复用零 token。
标准答案按明确的 `#### number` 行用 Decimal 精确比较，达到长度上限而没有答案仍计错并单列。

预先声明的任务回归容差为 2 个百分点，单侧置信水平 95%。
[评分工具](../../../tools/bench/score-task-quality.py)从原始文本独立复算，
同时报告短/长提示正确率、格式缺失、长度上限、回归与新增答对的题号。
这是固定算术任务子集，不能外推为全部能力通过。三个 arm 均完整运行 256 题，独立工具复算如下：

| Arm | 总正确 / 256 | 零例 / 128 | 八例长提示 / 128 | 缺失明确答案格式 |
|---|---|---|---|---|
| native scalar | 240（93.75%） | 116 | 124 | 8 |
| Neumann WY | 120（46.875%） | 118 | 2 | 119 |
| 固定 vLLM 构建 | 243（94.921875%） | 116 | 127 | 9 |

WY 相对 scalar 有 124 道回退、4 道新增答对；单侧回退率上界为 `0.53763897`，未通过预声明质量门槛。
相对 vLLM 有 128 道回退、5 道新增答对，同样未通过。WY 长题出现重复、错误算术和不完整输出，
不能把它更短的生成时间当作性能提升。scalar 与 vLLM 的本题集准确率差为 1.171875 个百分点；
这一点估计不能代替置信区间或全部能力验收。主分支默认 GDN 策略保持原状。

## 求逆稳定性与实际 checkpoint 输入

新增弱衰减、相关 K 的 F64 回归用例（32 token，beta 输入 8、a_log 输入 −10）复现 Neumann WY 失稳：
最大 state 误差 `112.0055746`，示例 GPU state `−45.29988` 对 F64 参考 `−0.02828961`。
它揭示了此前衰减较强的随机小值矩阵未覆盖的数值范围。

测试专用的独立 graph memcpy 在原 Delta entry 前后截取 Q/K/V、beta、alpha、权重与状态，
完整模型使用实际长题 `gsm8k:402:eight_shot`。前三个 recurrent 层、前四个 64-token replay
同时检查 scalar 观测前后 hidden、logits 与全部状态逐位一致，全部通过。
由原始 F32 字节独立计算 F64 递推：首层首块 scalar state/output 最大误差为
`1.0402392e-5 / 9.7231279e-7`，Neumann WY 为 `1.2210167 / 0.17688752`。
因此真实输入已复现显著 kernel 误差，并非只有模型后续的量化放大。

首块真实 K 最大相关性约 `0.98367`，最大 decay 约 `0.99999994`。
下三角矩阵幂的最大值依次为 `8.26、213.39、5805.25、741.68`，最终 inverse 最大绝对值仅为 1。
CPU F32 模拟的 Neumann inverse 最大误差 `8.5524e-4`，递归分块三角 inverse 为 `1.3030e-7`。
CPU 模拟只用于定位相消；GPU 三项 BF16 residual MMA 修复仍需单独验证。

实际 state 的数值尺度也暴露了固定绝对门槛的适用范围：前三层前四块的 scalar 中，
首层后续三块 state 对 F64 的最大误差为 `2.3153e-5` 至 `3.5472e-5`，
原实现同样超过随机小值测试的 `2e-5`，relative L2 仍为 `4.19e-7` 至 `4.73e-7`。
保留这些失败与原门槛，不能用随机矩阵通过代替真实输入验收。修复后的完整质量结果与固定工作量服务性能见后文。

## 稳定求逆与 F32 操作数精度修复

在独立副本中改用递归分块三角 inverse，避免全矩阵幂的巨大中间值。
仅三项 BF16 residual MMA 的稳定版仍有首层 output `4.2774e-5`、state `1.6436e-4` 误差，
因此继续使用三段 BF16 分解，并保留二阶项的六个 F32 累加乘积，补足原 F32 操作数的有效位。
state 始终为 F32；这不是将模型状态改成 BF16。

原独立参考矩阵、64/256-token 实际图宽度边界、新增弱衰减相关 K 用例均通过原 `2e-5` 门槛。
弱衰减回归用例的修复后最大 state/output 误差为 `4.2721e-8 / 7.2049e-8`。
实际 checkpoint 的前三层、前四个 64-token replay 全部通过 scalar 观测前后的逐位一致性检查。
独立 F64 复算中，首层首块修复后 state/output 最大误差为 `1.4302e-5 / 1.5044e-6`。
全部实际 output 通过原绝对门槛；一项 state 为 `2.3564e-5`，仍保留为绝对门槛失败，
其 relative L2 为 `2.4286e-7`。原 scalar 在同组真实输入中有三项 state 绝对门槛失败。
这些检查仅隔离每次调用从实际 F32 初态开始的 kernel 误差，不能代替整个模型任务验收。

沿用同一 256 题、相同采样/输出上限及原评分协议，修复后答对 243/256（94.921875%）：
零例 119/128，八例长提示 124/128。相对原 scalar 有 1 道回退、4 道新增答对，
原保守单侧回退率上界为 `0.01839577`，通过预声明的 native-control 质量门槛。
相对 vLLM 有 4 道回退、4 道新增答对，虽然总正确率相同，保守回退率上界 `0.03539722`
未通过该更强的逐题回退门槛，不能宣称相对 vLLM 的统计验收全部通过。

稳定版仍在一个 CTA 内串行处理多个块。上下文并行实验把 local transition/output 跨块展开，
使用真实首层输入的独立 CPU F64 代数校验已通过，state/output 最大差为
`2.1316e-14 / 5.3291e-15`；GPU 普通输入的独立参考通过，完整模型与直接 WY/CP 成本验收仍未完成。
所有候选均未进入默认生产调度，后续列宽、图宽度与策略选择须接入已有按设备身份缓存、
冷/暖缓存配对确认的 tuning；不能凭 5090 上的单次计时固定选择。

| 新证据目录 | 内容 / 二进制 SHA256 |
|---|---|
| `gdn-wy-stable-v1` | 稳定三角 inverse、三项乘积的残余精度问题；test `64d9bd88342ff08b101359df8b00a27dcbb506c1c50f50fbd0545ca3d3ea692c` |
| `gdn-wy-stable-f32-v1` | 六项乘积、完整数值矩阵和实际 F64 参考；test `c16767c4ff80de575d0feea7faba7e52f1ea53b55a17ec130d6b377f512f9aaa` |
| `gdn-wy-stable-f32-quality-v1` | 256 题完成输出、独立评分和 manifest；CLI `46ff6a53a74b09c14771568b502a671e135196faaedf0af1384c77a85310eddf` |
| `gdn-cp-f32-v1` | 并行转移公式与隔离的 GPU 原型；CPU 代数与普通 GPU 独立参考通过，完整模型与真实弱衰减资格未完成 |



## 稳定候选的固定工作量性能

相同稳定版 release CLI 的 scalar/WY arm 与固定官方 vLLM 构建，使用两轮新服务进程，
顺序为 scalar→WY→vLLM 与 vLLM→WY→scalar。每个 case 排除一次 warmup，
保留三次测量；每个 arm 总计 36 个测量请求。全部请求固定生成 64 token、实际 prefix reuse 为零，
MTP=0，模型文件、输入 token、资源上限和 tuning table 固定。
两轮的三个对照均通过外部 GPU 进程和显存的严格起始/捕获一致性检查。
以下为两轮进程内中位数的中位数，只有两个进程样本，不能称为五进程统计认证。

| Case / metric | scalar ms | stable WY ms | vLLM ms | WY / scalar | WY / vLLM |
|---|---:|---:|---:|---:|---:|
| short TTFT | 40.866 | 38.408 | 50.237 | 0.940 | 0.765 |
| short TPOT | 16.111 | 16.151 | 14.813 | 1.002 | 1.090 |
| long TTFT | 242.160 | 179.050 | 60.037 | 0.739 | 2.982 |
| long TPOT | 16.169 | 16.134 | 14.854 | 0.998 | 1.086 |
| long request wall | 1262.104 | 1195.914 | 996.100 | 0.948 | 1.201 |
| batch4 TTFT | 173.394 | 155.550 | 84.982 | 0.897 | 1.830 |
| batch4 TPOT | 19.533 | 19.393 | 16.408 | 0.993 | 1.182 |
| batch4 group wall | 1407.224 | 1378.328 | 1120.976 | 0.979 | 1.230 |

稳定候选降低 long TTFT 26.06%，两轮分别降低 24.99% 与 27.14%；decode 策略保持原状。
short TPOT 单轮 WY/vLLM 为 1.07965 与 1.10118，不能凭汇总低于 1.10 宣称每轮门槛全部通过。
long TTFT、batch4 和整请求时间仍未逼近预声明的 vLLM 1.10 倍目标。
这些 MTP=0 数据与之前 MTP=2 对照属于不同协议，不能混用其数值来宣称整体代际提升。

独立固定工作量 GDN graph 对照包括输出转置，并同时测 cold/warm、AB/BA。
64-token 图宽度、value block 128 的速度比为 1.738–2.235；block 64 为 1.989–3.080；
block 32 为 2.194–2.886。256-token 下分别为 2.312–2.376、3.023–3.214、3.142–3.257。
这说明 winner 随图宽度与缓存状态改变，不能把本验证设备的某个列宽写死。
计时只是新硬件 tuning 的候选证据，不能覆盖数值门槛或取代完整服务验收。

服务报告、后台资格和全部日志保存在 `gdn-wy-stable-f32-perf-v1`，独立汇总为 `summary.json`。
GDN graph 原始计时在 `gdn-cp-f32-v1/wy-cost.log`，二进制 SHA256
`d38d8f21dfc7628c88a7301ea52298f1c4cf009bfffff1b171e27ab8bb77da64`。
测量脚本首轮误用 MTP=2 的预期报告文件名；已完成的 MTP=0 control 原始报告及错误执行记录均保留，
仅更正报告路径后继续其余五个服务进程，没有筛选、重跑或删除模型测量。

## 证据身份

持久实验根目录为
`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles`。

| 子目录 | 内容 / release 二进制 SHA256 |
|---|---|
| `gdn-stage-localization-v1` | 原始输入、14 阶段字节、F64 参考、控制误差和 evidence manifest；test `f32298a73573e77fb13f6232d0ddc31f2e57d10ded471cff98cfdca045cb010e` |
| `gdn-wy-quality-v1` | 候选源码、构建与 64 项 F64 误差；test `ad518032faf9ed61bda973a510e4e04421aed175e6541324c935815a2f7be899` |
| `gdn-model-quality-v1` | 三组各 256 题、失败评分与 evidence manifest；experimental CLI `cde2cd8c523df2ffba285534a0e8f2d042b78ff41f8ce12c93fd93a5fab96544` |

新增 `gdn-actual-input-v1` 保存观察一致性、实际输入字节、F64 参考与不稳定求逆回归；
对应 test SHA256 为 `87403c6b5d5647db5c3498deb91a182e86344685998b8c75822da4e196b0260e`。

构建、GPU 数值和完整模型运行分别受 8/16/24 GiB protected scope 保护。
新启动的 Chrome GPU 进程完整保留；数值/质量运行记录已核验的桌面进程，
这些运行不提供合格的服务性能基线。5090 仅为本次验证设备。
后续 kernel/线程布局/图宽度选择需接入按设备与 workload 身份缓存、独立配对确认的 tuning。

## 自动列宽原型的数值与模型资格

独立原型复用已验收的稳定递归分块三角 inverse 和六项 residual MMA，recurrent state 与
activation 规则保持 F32。量化模型通过已有 `--chunked-recurrent` 启用该路径；默认仍为 scalar。
支持的算法几何是 D=32/128、32 的倍数且不超过 256 的 prompt graph，MoE 保持原路径。
首次原型以 D、D/2、D/4 做有限搜索，根据实际硬件测量选择，代码没有按设备型号选择分支。

新 `resident_recurrent` 节沿用统一 tuning table 的原子更新。身份含完整 DeviceProfile、驱动 API
和实际驱动构建、架构、canonical Tile IR、assembler、编译选项及相关环境开关、数值协议和
KH/VH/D/LANES。每个候选先验证普通与弱衰减相关输入的独立 F64 参考；paired graph timing
包括输出 transpose，冷/暖 AB 搜索后 BA 确认，两种缓存模式均要求至少 2% 改善。
不完整、重复候选、缺少确认或非有限数据不能成为有效缓存。新搜索共用每模型单独的 30 秒
recurrent allowance；正在执行的 JIT/测量不可中断，因此它是停止追加搜索的预算，不是硬超时。
关闭 autotune 仍读取有效完整缓存；缓存未命中使用完整列宽，非法证据/不支持形状回退 scalar。
手动列宽诊断参数仍独立于自动调优缓存。

同一二进制中的 canonical IR 不变，按架构、编译环境与形状只生成一次身份，后续相同层复用；
编译环境变化仍会使缓存未命中。GPU 测试中的 48 次相同层查询合计 0.252376 ms。
首次模型试运行遇到资源准备 30 秒超时及后续控制通道超时，两次失败全部保留。
修复身份重复计算并完成必需的 GPU 内核编译后，最终版本首次模型 warmup 成功，无重试。
不能用这个预热后的成功单独宣称全新 CUDA cache 的冷启动时限也已验收。

最终 release 构建、CUDA all-targets 严格 clippy、format、50 项常规单元检查、4 项 tuning 检查
以及 3 项 WY GPU 参考测试通过。独立 GPU 输出记录 100 项逐元素 state/output 误差，最大
`1.66562148e-7`，通过原 `2e-5` 门槛。实际自动搜索选择 value block 32，确认阶段普通/相关
输入最大误差 `1.15873437e-7`；这是本验证机的测量结果，其他设备使用自己的身份和测量。
列宽 32 的完整质量结果见下段。此版本未通过原模型质量资格，也没有新的合格服务性能结论；不能把此前 value block 128 的性能数字归于新配置。

证据目录 `gdn-tuning-f32-v1`；CLI SHA256
`261df93ff13b53aa9ede6f2e52e17744be2ef13f10801224e16b9c78225238e1`，unit SHA256
`87c1945346f5502fcd28d16a3b6fd81a0d99f6fa92cac8c1e257c47116ede322`，硬件 table SHA256
`635fb8c73c5908654091e2e767128d3846cafb9c1bea3c5ada52037bd76b6302`。
`source-built-final` 保存与二进制 source manifest 一致的源码；`final-evidence-manifest.json`
与原始日志保留全部验收身份。质量目录为 `gdn-tuning-f32-quality-v1`，性能目录为
`gdn-tuning-f32-perf-v1`。

完整原题集测试记录 256/256 请求，列宽 32 为 242/256：零例 119/128、八例 123/128。
相对 scalar 的 240/256 有两道回退、四道新增答对，总正确率增加 0.78125 个百分点；
原预声明的保守逐题回退率单侧上界为 `0.0243874146`，超过 2%，资格为 `not_demonstrated`。
回退题是 `gsm8k:1030:eight_shot`（长度上限内没有明确最终答案）和
`gsm8k:1172:eight_shot`（明确的错误算术答案）。相对固定 vLLM 的 243/256，
有五道回退、四道新增答对，上界 `0.0406256185`，同样未通过。
不能以总准确率提高掩盖逐题质量门槛未通过，也不能因此宣称全部任务能力下降。
原始结果 SHA256 `d90b05cac9f16617b3b449f7b21ce7d961515177d00142bbade0f524dbae6b46`。

后续原型将四分之一列宽移出自动候选，不按设备型号分支；保留完整与一半列宽，
用冷/暖 AB 和 BA 确认决定是否采用一半。新的 schema 明确标记 `columns-full-half-v2`，
旧三候选的缓存不能成为新决策。手动四分之一列宽保留为诊断参数。
`gdn-tuning-f32-half-v1` release、严格 clippy、50 项常规单元检查与 4 项硬件 tuning 检查通过，
实际选择列宽 64；CUDA 算术源码与已经通过 100 项独立参考的版本逐字节相同。
对应 `gdn-tuning-f32-half-quality-v1` 已完成完整模型资格，详见下节；新固定工作量服务性能见最后的两轮对照。

## 上下文并行原型与下一步成本方向

三阶段 CP 原型在普通输入上完成 144 项 GPU state/output 检查，覆盖六种几何、四种边界和
三种列宽，全部通过原 `2e-5`。最大 state/output 误差为 `2.17523451e-7 / 3.05893418e-8`。
它未完成弱衰减真实 checkpoint 和完整模型质量验收，未进入生产选择。
CP 原型输出转置也计入成本；它额外物化 dense M/N/初态矩阵，并执行 D×D affine prefix
组合。现有独立 scalar 对照没有显示稳定优于 serial WY 的证据，且不是直接的 WY/CP 配对，
不能把两个实验的 cache 干扰差异当成同条件速度结论。

剩余优化应围绕已测成本：MTP=2 pooled request 的窄 prompt graph 仍是 64 token；本次 MTP=0
长提示实际使用 256-row 图，不能给它套用窄图归因。扩大 pooled 图宽度可以增加 GEMM 并行度与权重复用，但之前私有宽图增加
显存并使 pool 丢失槽位，所以需要共享 arena 或改变图/状态的资源所有权，而不是直接固定 256。
图宽度最终须按硬件、资源预算和 workload 测量选择。另一个源码可见的成本是 value 列分区
重复计算相同 head 的 QK interaction 与 inverse；准备结果与 state apply 分离可能减少重复工作，
但需要新的直接配对成本和数值/模型质量验收。dense affine CP 的当前数据不支持先强制上线。

## 完整／半列宽自动选择的生产入口验收

本次提交把稳定 F32 WY 接入已有 `--chunked-recurrent`，并按硬件自动选择已具备质量资格的
完整或一半列宽。量化模型默认仍使用 scalar；四分之一列宽保留为手动诊断，不参与自动选择。
候选状态/activation 仍为 F32，canonical 身份只在同一编译环境与形状下复用，缓存字段和
独立 F64/冷暖 AB/BA 确认完整保留。`columns-full-half-v2` 不接受旧三候选 schema。

最终二进制使用完整／半列宽选择入口完成同一 256 题，答对 243/256：短提示 119/128，
八例长提示 124/128。相对原 scalar 240/256 有一道回退、四道新增答对，保守单侧上界
`0.01839577` 小于预声明 2%，通过 native-control 质量门槛。回退题为
`gsm8k:1172:eight_shot`。相对固定 vLLM 243/256 有四道回退、四道新增答对，上界
`0.03539722` 未通过更强的逐题回退门槛；总正确率相同不能替代这个统计资格。
两个 native 质量版本都有完整原始输出，失败的列宽 32 未被替换或删除。

release 构建、严格 CUDA all-targets clippy、format、50 项常规单元检查和 4 项硬件 tuning
检查通过；GPU 算术与通过 100 项 F64 检查的版本逐字节相同。新的 CLI SHA256 为
`00c804f8b29376a8a0c9516f2e6cc4a479efb6194eb775b532742e9e819f58ca`。源码及硬件缓存位于
`gdn-tuning-f32-half-v1`，完整质量及独立评分位于 `gdn-tuning-f32-half-quality-v1`。
此次资格限于这个固定模型/算术题集和验证设备；服务性能用新的
`gdn-tuning-f32-half-perf-v1` 两轮三 arm 对照验证，不沿用旧 value block 128 的收益数值。


## 完整／半列宽选择入口的历史固定工作量性能

`d8e0a37` 对应 CLI `00c804f8…`、64-row 图的缓存列宽 64，用同一 native 二进制的 scalar 模式作对照，
并比较固定官方 `3fe917abb` vLLM runtime。两轮启动顺序为 scalar/WY/vLLM 与 vLLM/WY/scalar。
三种实际输入为短提示 51 token、长提示 511 token、四并发各约 52 token。
每进程包含三种工作量，各测三次；每种 arm 跨两轮共 36 个测量请求，每进程 18 个。
三个工作量各保留一次 warmup 并排除计时。所有请求均完成、恰好生成 64 token、实际 prefix reuse 为零。
六组比较的外部 GPU 进程身份与显存占用，在启动与首次观察到模型 GPU 进程时均相同。
两个 native arm 的缓存前后 SHA 不变。数值/任务质量使用各自独立门槛，不把生成 token 完全一致设为条件。

下表取两轮各进程中位数的中位数，再计算比值；单位 ms。wall 在四并发时为整组耗时。

| 工作量／指标 | scalar | F32 WY | vLLM | WY/scalar | WY/vLLM |
|---|---:|---:|---:|---:|---:|
| 四并发 wall | 1414.045 | 1390.004 | 1128.246 | 0.9830 | 1.2320 |
| 四并发 TTFT | 173.454 | 154.154 | 82.855 | 0.8887 | 1.8605 |
| 四并发 TPOT | 19.538 | 19.473 | 16.548 | 0.9967 | 1.1767 |
| 长提示 wall | 1262.194 | 1197.171 | 1002.100 | 0.9485 | 1.1947 |
| 长提示 TTFT | 244.417 | 180.135 | 65.731 | 0.7370 | 2.7405 |
| 长提示 TPOT | 16.157 | 16.143 | 14.857 | 0.9991 | 1.0865 |
| 短提示 wall | 1055.641 | 1056.756 | 977.971 | 1.0011 | 1.0806 |
| 短提示 TTFT | 41.431 | 37.568 | 44.814 | 0.9067 | 0.8383 |
| 短提示 TPOT | 16.065 | 16.155 | 14.799 | 1.0056 | 1.0917 |

长提示 TTFT 减少 26.3004%，两轮各自减少 27.5330% 与 25.0563%；decode TPOT 基本不变。
两轮相对 scalar 的 1.10 延迟阈值均通过，相对 vLLM 均未通过。长提示 TTFT 仍为 vLLM 的
2.7405 倍，四并发 wall 为 1.2320 倍；目标尚未达到。短提示 TPOT 汇总为 1.0917 倍，
但第二轮为 1.1033 倍，不能只用汇总数字宣称逐进程全部达标。
这仅是两轮 MTP=0 的证据，不是五进程统计验收，也不外推为 MTP=2 或其他模型。
旧固定列宽与本次选择入口没有直接配对，不能从不同实验推断列宽本身额外贡献了多少服务收益；各图宽度的缓存与 replay 归因见最后的诊断。

原始报告、缓存、执行身份与硬件监测位于 `gdn-tuning-f32-half-perf-v1/paired`，
`summary.json` SHA256 为
`7ff4f93025f492b1608839e54c6c75d99ee63070342ca850c7616b72246628f0`。

## CI 修复验证

初次主分支 quality run `38209473275` 在 Linux/macOS 的仓库 policy 检查失败：
CUDA entry 使用了禁止的 `too_many_lines` 豁免。修复拆分累计衰减与三角求逆辅助函数，
并给 host 调参、独立参考和元数据尺寸补充命名常量；没有修改 lint 规则或数值误差门槛。
policy、layout、测试清单、format、release CUDA all-targets clippy、50 项常规单元检查、
4 项 tuning 检查和 3 项独立 WY GPU 数值矩阵测试已通过。
辅助函数导致 canonical 身份变化，旧缓存正确失配；新 AB/BA 搜索仍选 64。
新二进制 `4be5d78d…` 的完整模型质量和固定工作量性能独立重验，不将上表归于新二进制。
证据位于 `gdn-tuning-f32-ci-fix-v1` 与对应 quality/perf 目录。

新二进制已完成完整 256 题，仍为 243/256（119/128 短提示、124/128 八例长提示）。
独立评分相对 scalar 一道回退、四道新增答对，原上界 0.01839577，通过原 2% 门槛；
相对 vLLM 四道回退、四道新增答对，上界 0.03539722，仍未证明更强逐题统计资格。
256 道生成 token 与修复前列宽 64 的合格版本逐题相同。该观察补充独立评分，不能替代其质量协议。
首次模型 warmup 成功，未发生重试；binary/cache 前后 SHA 不变，实际 prefix reuse 为零。

## CI 修复后二进制的最新性能与图宽度核验

代码提交 [46d0144](https://github.com/1994/inference.rs/commit/46d0144c1a706459609dde5f4e4f017ad6d79731) 的
[完整 quality CI](https://github.com/1994/inference.rs/actions/runs/38213339892)和
[docs CI](https://github.com/1994/inference.rs/actions/runs/38213340006)均通过，包含 Linux/macOS、
工具、安全、MSRV 与四种发布目标的打包。性能测试与 CI 并行运行。

新 CLI SHA256 为 `4be5d78d71c995e773c9571b0bf23fadb57796d8d44439c96ec1d8f5602d8440`。
第 0 轮顺序 scalar/WY/vLLM，通过严格背景资格；第 1 轮反向顺序因外部 GitHub Desktop
显存由 123/114/108 MiB 变化而未通过原一致性规则，原始输出全部保留。补测第 3 轮反向
vLLM/WY/scalar 三组对照均通过。汇总只使用第 0、3 轮，两轮各进程中位数的中位数，
每种 arm 共 36 个测量请求；输入、64-token 固定输出、MTP=0、实际 prefix reuse=0、
同一 native 二进制与冻结缓存均核验通过。未因性能结果排除样本。

| 工作量／指标 | scalar ms | F32 WY ms | vLLM ms | WY/scalar | WY/vLLM |
|---|---:|---:|---:|---:|---:|
| 短提示 TTFT | 44.604 | 37.184 | 48.258 | 0.8336 | 0.7705 |
| 短提示 TPOT | 16.166 | 16.162 | 14.956 | 0.9997 | 1.0806 |
| 短提示 wall | 1064.471 | 1056.415 | 990.729 | 0.9924 | 1.0663 |
| 长提示 TTFT | 245.623 | 180.212 | 60.309 | 0.7337 | 2.9882 |
| 长提示 TPOT | 16.128 | 16.134 | 15.004 | 1.0004 | 1.0753 |
| 长提示 wall | 1263.368 | 1197.139 | 1005.666 | 0.9476 | 1.1904 |
| 四并发 TTFT | 172.318 | 150.891 | 84.463 | 0.8756 | 1.7865 |
| 四并发 TPOT | 19.371 | 19.486 | 16.586 | 1.0059 | 1.1749 |
| 四并发 wall | 1401.486 | 1379.957 | 1139.394 | 0.9846 | 1.2111 |

长提示 TTFT 相对同一二进制 scalar 减少 26.6304%，decode TPOT 基本不变；相对 vLLM
长提示 TTFT 仍约 2.99 倍，四并发 wall 约 1.21 倍。两轮各自的 native-control 延迟阈值均通过，
vLLM 延迟阈值均未通过。只有两轮合格进程样本，不能宣称五进程统计验收或全部模型达标。
原始目录 `gdn-tuning-f32-ci-fix-perf-v1/paired`，汇总 SHA256 为 `cfcf6eb88c98a7707c6b5ce80983aeabe5e4f0bf60f9d80c8bfef3e611ea92a7`。

原来的 “WY64” 表头把策略入口和实际列宽混在了一起，现修正为 F32 WY。
冻结 table 只有 `[KH=16,VH=48,D=128,LANES=64]` 的条目，选择 value block 64；
在 `--no-autotune` 下 256-row 图 cache miss 回退完整 128 列。
单独的执行诊断观察到 prefill task 输入宽度 51、52、256、255，长提示按 256+255 调用；
结合 `prefill_batch_readout` 的实际 dispatch，51/52-token 提示选窄 64-row 图，
256/255-token 提示选宽 256-row 图。因此当前长提示收益来自稳定 F32 WY 算法，
不是已证明的半列宽调参收益；该工作量没有直接比较完整与半列宽。
生产质量结果同样限定为这个选择入口与冻结 table 的组合，不能解读为所有层 replay 都是半列宽。

诊断目录 `gdn-replay-width-diagnostic-v1/attempt1`，汇总 SHA256 为 `0b07ff71ace0594cfcddcb8dd4c61d2c7ce464679e184cebbd0f57fbf9ea2add`。
初次诊断漏复制 checklist，启动后在测量前失败，原日志保留；修正后诊断完成。
诊断开启 host phase probe，计时不计入上表。
