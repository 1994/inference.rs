# F32 GDN 数值定位与完整模型质量验收

日期：2026-10-12。源码基点 `989bb6b94c522098dc7cadd004a72aa506f18e68`，开发和 GPU 验证位于
`r@192.168.1.45`。用户确认的验收方向为保持 F32 recurrent state，使用独立数值参考与完整模型任务质量。
Neumann WY 候选已完成完整模型质量测试并失败，未进入主分支生产调度。
分块三角求逆修复正在独立实验副本验证；有效性能提升尚未验收。

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
[评分工具](../../tools/bench/score-task-quality.py)从原始文本独立复算，
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
保留这些失败与原门槛，不能用随机矩阵通过代替真实输入验收。修复的完整模型质量和配对性能仍待验证。

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
