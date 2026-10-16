# vLLM 优化方向复核（2026-10-13）

当前方向有一部分正确，但主次要调整。当前四路 decode 的时间主要在 GPU 图内，Rust 的主机端开销不是这份诊断中的主要差距。优先验证完整 FP8 大投影与共享 KV 的 prefill attention，并消除同输入的重复量化；GDN 时间块准备已测出成本空间，也复现了质量退化，先定位数值差异。单行 head、局部投影融合和继续扩大 tile 搜索不足以解释长提示 2.75 倍的耗时差距。

## 当前结果与版本

C5 首轮同机、同输入、MTP=0、实际 prefix reuse=0，每项三次测量：长提示 TTFT 169.510 ms / vLLM 61.589 ms；长提示 TPOT 16.196 / 14.718 ms；batch4 完成时间 1391.632 / 1138.348 ms。只有一轮，当前 C5 自身的完整质量和稳定性验收仍未完成。

测量 vLLM 为官方 3fe917abb13160d60bea5330399f5c09ca282348 wheel，日志确认 BF16 activation、F32 recurrent state、FlashInfer GDN、FlashInfer attention、Cutlass FP8 和 FlashInfer Cutlass NVFP4，MTP=None，token budget=2048。2026-10-13 又从官方 upstream fetch，并将干净 main 快进到 df6c7ad03194a3051a197d57ca4a94e4a063014e；研究中的新 main 不能标作本轮被测 wheel。

3fe 到 df6 的 Qwen GDN 包装、packed recurrent/conv 和 NVFP4 FlashInfer 路径没有实质算法变化。NVFP4 C++ 的相关迁移文件只改 include 或完全相同；CUDA FP8 CUTLASS 的对应变化为注释。FLA ops 主要改变 Triton 地址写法，FlashInfer attention 移除旧 cascade 路径。不能用“最新 main 又变快了”解释当前测量差距。

## 用实际成本约束优化预期

下面重算的是原 WY 的 CUDA node-event 诊断，不是 C5 的精确设备分解。原始二进制、cache 与 C5 不同，节点计时也有额外开销。各类中位数相加只用于选方向。

| 511-token 两块 prefill | 诊断时间 |
| --- | ---: |
| 线性层相关操作（含量化） | 83.863 ms |
| GDN | 51.568 ms |
| Attention | 18.483 ms |
| 最后一块的词表 head（已包含在线性层中） | 1.770 ms |

184 次四槽活跃的旧 decode 图中位数为 21.714 ms。按照 IR 顺序绑定真实 checkpoint 格式后，逐次求和的中位数：目标层 FP8 8.318 ms、NVFP4 6.172 ms、BF16 B/A 1.926 ms、终端 FP8 head 0.767 ms。FP8 目标层约占全图 38.3%，线性层整体约 79%。这些比例不能替代未插桩 TPOT。

B/A 的 1.926 ms 是 M4 图成本，不能用于填补单路约 1.5 ms 差距。M1 已完成的部分真实权重诊断中，两路原投影约 cold10 / warm8.5 微秒，融合最快约 8.3 / 6.6 微秒；外推 48 层仅约 0.1 ms 量级，且完整 sweep 尚未完成。M4 即使局部快四倍，按旧图预算也只约降低 6.65% 图耗时。

2026-10-13又复用C5二进制进行了两event模型图边界诊断。同一进程中，排除warmup及部分活跃后的186次四路decode：GPU模型图中位数18.802ms，executor墙钟19.046ms，逐次图外差值中位数0.241ms（selector、D2H、同步、metadata与profile写入等，不能直接叫CPU浪费）。这为当前M4优先优化GPU图内提供证据，仍不能替代真实M1计时。

该诊断没有改runtime/cache，物理guard合格，20个请求都完成64token且prefix reuse=0。相比旧C5，3个单路测量的token完全相同，12个并发测量中3个token流不同；不能直接用两次wall差值认证插桩开销，更不将其作为新性能提升。首次启动误复用了含long的旧checklist，服务在正式工作负载之前拒绝；保留该attempt，第二次声明真实short/batch4子集后完成。证据：
`vllm-direction-review-2026-10-13/c5-graph-only-profile-plan-v1/execution-v2/analysis.json`，
SHA256 `1affa8095aed13297de69b96949c2ea92354e938e2c1806fe081241f32e76f69`。

## F32状态不等于相同运算量

冻结的vLLM/FlashInfer源码确认，GDN wrapper保留BF16 QKV，只将state、g、beta转F32；Q/K归一化先算F32再写回原BF16类型。CP实现的QK/KK主体使用BF16输入和F32累积；三角逆矩阵scratch为FP16，后续T再转BF16；部分state参与矩阵乘之前也会转BF16，state的最终存储仍为F32。fixup默认HMMA含单TF32乘积，不是所有中间计算都保持完整F32有效精度。

本实现的WY32对每个逻辑乘积用三个BF16分量、六次MMA及F32累积重建更高精度；还包含残差转换。双方“F32 recurrent state”描述相同，数值工作量却不同。调tile能改变占用和数据复用，不能消除这些显式乘积，也不能从六次乘积直接推出整个GDN慢六倍。C5日志确认FlashInfer GDN已选用；具体CP kernel选择仍是冻结源码派发条件推断，没有新trace，不写成本轮逐kernel测量。

这是性能对比的必要解释，不是降低生产精度的授权或质量证明。保持F32状态、既有独立参考门槛与完整模型质量；额外数值策略要单独声明、测量和验收。逐行源码/hash证据：
`vllm-direction-review-2026-10-13/flashinfer-gdn-f32-state-workload-audit-v1.json`，
SHA256 `7d8da178d8685cb77177cf40a2aa076f7a8f211f881dc5d25d916af5881c45ab`。

## 共享输入与中间结果：减少工作，不依靠扩大 tile

实际源码与 checkpoint 元数据确认：当前 recipe 将 64 组 gate/up、48 组 GDN qkv/z、16 组 attention q/k/v 分别执行，每次都重新量化同一个 normalized 输入。可识别每张完整模型图中的 144 次重复量化，其中 88 次为 per-token FP8、56 次为 NVFP4。已逐字节核对全部 56 层 NVFP4 gate/up 的 input/weight global scale 一致；对其他 checkpoint，必须检查 input 身份与版本、量化 scheme、K、row metadata 和 scale bits，不匹配就使用原路径。

先共用 codes/scales、保留原 GEMM、原权重与原输出，是比合并权重更小的改动。共用只在每次图内成立，不能把上一次 replay 的结果当成新输入。vLLM 的合并投影还在源码结构上减少 144 个 Linear 调用；合并不会减少权重总字节和乘法总量，需要另行准备权重布局、选择 tile 和验证数值。

历史 `fp8-input-preparation-cost-v1` 的 synthetic 量化-only 试验已删除过 48 个重复 FP8 量化，M4 约省 0.276 ms、M256 约 0.573 ms；不能把它重复包装成新发现，也不能外推为实际整模型收益。当前新增价值是确认真实 recipe 和全部 NVFP4 scale 的共用条件，并设计安全 capture 接入。这是合理的小改进，不能替代大 GEMM 主线。

另外，本轮 vLLM 日志明确 `custom_ops=['none']`、`fuse_norm_quant=false`、`fuse_act_quant=false`。FP8 QuantFP8 按配置走 native 表达式再交给 Inductor，不能把未执行的 `forward_cuda` 量化路径归作本轮实测。vLLM 的 CUTLASS FP8 在 GEMM epilogue 做 `(raw*weight_scale)*input_scale` 并输出 BF16；我们的 Native 已融合 `(raw*input_scale)*weight_scale`，输出 F32，只有库候选为保留两次原舍入另做 scale kernel。不得把交换乘法次序当作逐位等价。

NVFP4 Lt 候选还先写 linear scale 再 swizzle；vLLM 对应 API 直接写目标布局。该生产者优化只影响当前采用库路径的 M256，不能解释 M4 decode。Silu/Multiply 中间物化也可以减少，但已有局部融合试验需要补齐实际模型成本。源码及 224 份相关 NVFP4 scalar 原始值证据：`vllm-direction-review-2026-10-13/linear-work-and-epilogue-audit-v1.json`，SHA256 `bdbe8ddb468cfbe61e18e03139d749ce129fe611b972199ae10746013101b230`。

## 1. GDN：优先并行准备，保留 F32 状态

当前 full128 WY 每个 value head 的 CTA 串行循环 32-token 块。256 行是 8 块，512 行是 16 块。每块有 15 个 precise_mma 调用；每个调用含六个 BF16 分量乘积并用 F32 累积，共 90 个源码 MMA 调用；这不是硬件 MMA 指令计数。full128 的基本 head grid 为 48 个 CTA，时间块准备、归一化、QK/KKT 和三角系统均在这条串行链上。

FlashInfer 在适用设备与 head 数条件下选择 CP。45 的当前几何中，511 token 的 T-precompute 按八个 64-token 块与 48 head 展开为 384 个 CTA；MN-precompute 和最终 state propagation 并非全部并行。结构差异是先把可独立的准备工作分散到时间块，再处理状态依赖，而不是简单增大 graph 宽度。

又只读复查了已有的 3fe917 Torch CUDA trace：此前 MTP=2 的 long 诊断实际出现 TPrecompute、MNPrecompute、FixupHmma、Prefill 四组 CP kernel，各 48 次；对应 grid 为 384、48、96、48。四组设备 duration 合计 6.811459 ms，只能用于说明该旧进程真实采用了这个结构，不能与当前 C5 MTP=0 的请求或 F32 算术作同口径比值。batch4 trace 还出现非 CP 的 FullyFusedDeltaRule，说明调度后的批形状会改变算法选择。证据 `vllm-direction-review-2026-10-13/existing-vllm-cp-trace-reuse-v1.json`，SHA256 `a7f6fbeab00b95aeb6b889842f620a6ba0230ad9575668b4e9b6107640aeed73`。

可借鉴的是选择原则：当 batch×head 的独立 CTA 不足以覆盖设备 SM 时，展开时间块准备；块长同时平衡准备吞吐、串行状态传播和 workspace。FlashInfer 源码用设备 SM 数、序列长度及 head 数估算 wave 和成本平衡。我们的参数应进入 tuning，并由独立数值参考和实测决定，不能固定为 5090 的一个阈值。

此前 prepared 整模型质量失败不能归因于时间预计算拆分本身：原测试使用自动选择的 full/half 混合；后续五例 wide_full_prepared 只固定宽图 full128，窄图仍为 half64。这些结果不是窄/宽均 full128 的独立控制。

该隔离已完成：复用原 ce910773… 二进制，两臂窄/宽均固定 full128，仅 prepared 臂开启并行准备。相同输入、cache、F32 state；两臂 guard 都合格，背景两个进程及其显存一致，prefix reuse=0，正式请求没有重试。

六个既有敏感案例中，fused128 为6/6，prepared128 为1/6。prepared 的六份 token 流与先前被拒绝的混合列宽结果逐位相同，因此固定全128列没有解除已有退化。五个失败均缺原评分所需最终答案格式：806计算70%且自然EOS，但缺####；其余四例达到原512输出上限，不能据此宣称五题均发生算术错误，也不能改变评分器来取得通过。

单个宽256块的 GDN 插桩设备时间约26ms降至19ms，约减少27%；这是单次带node-event的诊断，不能算作合格服务性能。当前 prepared 原型继续保持禁用。下一步定位存储边界和浮点求值差异，并扩大独立参考至真实中间量；没有修复前不再重复完整256题或调列宽。该结果支持时间块准备有成本空间，但尚未证明这一实现满足质量要求。

证据：`gdn-wy-prepared-full128-isolation-v1/paired-summary.json`，SHA256 `5e84b8a9bae4eb7ad34352684705855bbee7e0b19664f450fd81b1697dc3c4a7`；`compiled-source-checksum.json`保留实际编译九文件身份。

## 2. FP8：补齐大行数库验证，不重复已失败的小行数试验

当前桥已支持 E4M3 输入、F32 compute/output、FAST_ACCUM=0：原输入量化后执行 Lt raw F32，再按原 raw * input_scale * weight_scale 顺序处理输出。描述符和算法在 capture 前准备，权重仍使用原驻留 allocation。

C5 的实际 FP8 cache 中 vendor_candidates 均为 false，运行未打开 FP8 Lt 搜索；已有生产搜索还要求全输出逐位等于 native。这说明当前候选未使用库搜索，不能据此判定库候选也无收益。更早的 vendor-fp8/checkpoint5 实际已测过四类真实权重、M1/M12/M64/M128，使用零 workspace 的第一个可用算法：M1 gate/down cold 约0.98–0.99倍、warm约1.008–1.009倍，QKV/head更慢；M128 gate约1.229倍cold、1.548倍warm，down约1.110/1.051倍，QKV约1.096/1.124倍。这仅一轮、每项32个抽样F64输出，没有完整参考与模型验收。不能再把它写成“库路径从未测过”，也不重复该批单行试验。

下一步沿用已有桥，对真实 FP8 大投影测完整量化、Lt、scale 图，优先补齐 M256，并记录 M4；M1 仅在算法/workspace或融合结构发生实质改变时再测。候选保持独立 F64 量化乘积参考和原 scale 顺序；不同 native 位模式单独记录，生产逐位 gate 暂不改变。完整质量通过前只作为诊断。当前仅支持零 workspace，是否允许额外 workspace 必须按实际预算和完整图收益评估。

六份真实 checkpoint 权重在 M4/M256 共 12 组的完整 CPU F64 参考已经完成，包含原 activation、独立 FP8 codes/scales、完整 raw/scaled 输出及两次 F32 舍入的 epilogue 参考；384 个额外 scalar F64 输出与完整参考严格一致。随后 GPU 完成 12 份 native 和 96 个候选实例的 129392640 个 F64 输出比较，失败与非有限值均为 0；149094400 个量化 codes 完整逐字节一致，全部 input scale 此次 F32 bits 也一致。原输出门槛与量化门槛没有改变。

完整 quantize＋GEMM＋原两次 RN scale 的同物理缓冲区成本完成 3 轮 cold/warm AB/BA，共 1152 条配对记录。以一份权重发现算法、固定同一算法到另一个 layer 端点作同进程检查，M256 warm 结果：

| 投影 | 原 Native | Lt 完整图 | 局部加速 |
| --- | ---: | ---: | ---: |
| gate，layers56/63 | 161～163 µs | 115～117 µs | 1.39～1.40 倍 |
| down，layers56/63 | 199～201 µs | 175～176 µs | 约 1.14 倍 |
| GDN out，layers0/54 | 约 58 µs | 约 52 µs | 约 1.12 倍 |

M4 warm 仅约 1.00～1.04 倍，cold 均不超过 1 倍，当前不值得强推 decode Lt。M256 gate/down 的 Native64×64 是实际未缓存 fallback，不能称为超越已调优 Native；GDN out 使用 T5 实际选中的64×128。还应比较最佳合法 Native，再评价库收益。

此次 48 个 M256 候选实例普通 fixture 全部与 Native 输出逐位一致；M4 合计 391103 个位差。普通 fixture 不能代替生产四组 fixture 的 exact gate。现阶段未修改 cache、生产选择或精度门槛，完整模型质量、独立服务配对均未验收。一次实际 GPU invocation 完成全部数值和成本；第一次 admission 因桌面 util5～6% 未启动，保留记录。第二次允许记录背景利用率后执行功能阶段，紧邻成本仍按原三连续≤2%门槛，样本1/0/0通过；背景身份、显存和全部源/二进制/cache/参考均保持一致。证据 `vllm-direction-review-2026-10-13/fp8-phase-a-test-source-v1/complete-reference-and-cost-summary-v1.json`，SHA256 `85dfc2a7650c17c93211cf5fc09ea1e9ada95d15c5785e6ab3f1dc6709e9c8c8`。

## 3. Attention：共享 query/GQA 的 KV，而非只切开关

默认 prefill attention 仍是一行 query、一个 head 的 SIMT 计算，重复加载 KV。vLLM/FlashInfer 用 query tile 和 GQA packing 共享 KV。现有 native tiled 变体已存在，但默认关闭；旧单元测试不能代替当前真实 FP8 KV、head_dim256 与整模型质量。

现有 tiled kernel 的循环上界来自整个 chunk 的末位置，所有 query tile 都遍历该上界，然后才 mask 未来 key。512 宽图可能增加早期 query tile 的无用矩阵运算。可按每个 CTA 的实际最大 query 位置截断 causal 块，同时共享多个 query 的 KV；参数和合法路径由能力、shape、预算与实测选择。

按 CTA 截断的测试候选已通过真实 D256、24 query head、4 KV head、FP8 KV 的完整数值检查。64/256/512 三宽度共 15 组，覆盖尾行、count=0/1、负位置、滑窗和页表排列：127795200 个 F64 输出满足原 `1e-4*max(1,abs(reference))` 门槛；裁剪前后 51118080 个输出逐位一致；157286400 个 append 后 FP8 codes 完整验证。

同物理输入、KV、output 的完整 append+compute 成本也已完成 72 次 cold/warm、AB/BA 配对。QT16 截断候选的 warm 中位数对比 SIMT：

| 图宽度与实际输入 | SIMT | QT16 截断 | 局部加速 |
| --- | ---: | ---: | ---: |
| width64，count51，base0 | 54.168 µs | 38.824 µs | 1.395 倍 |
| width256，count255，base256 | 788.256 µs | 210.688 µs | 3.741 倍 |
| width512，count511，base0 | 1173.760 µs | 356.440 µs | 3.293 倍 |

其中仅“截断未来块”相对已有 QT16 tiled 的 warm 收益分别约 0.76%、7.74%、14.18%；主要局部收益来自 query tiling。QT32 整体更慢，width64 cold 甚至慢于 SIMT。三个输入的 base/count 不同，不能据此选全局图宽或固定 QT16。

生产默认保持原路径。这些结果属于实际几何的独立 fixture，SIMT→tiled 并非逐位一致；仍缺实际 checkpoint 中间量、完整模型质量与未插桩服务验收，不能记作 TTFT 提升。首次 launcher 的进程环境检查失败；第二次完整数值通过后，成本前 idle 为 3/3/1%，未满足原连续三次 ≤2% 门槛，未开始计时。第三次独立 cost-only 运行绑定第二次数值证据、idle0/0/0 后完成；旧失败和资格状态均保留。证据 `attention-prefill-causal-bound-v1/validation-summary-v7-lint2.json`，SHA256 `d975c00179770d04c264cf6f6a38463dd7ae3e704a253fa6e237960ca8983b24`。

## 4. 单行 head / 512 宽图：资源优化，收益不能夸大

单行 head 的实际 CPU arena 布局已核对：256 行从 379158528 降至 125872128 字节；512 行从 758317056 降至 250750976 字节。真实模型状态、KV、完整最后 logits 的功能门禁已通过，64/256 两宽度比对 2486254592 个元素，无差异或非有限值；512 尚未加入默认或合法宽度梯级。

head 自身仅约 1.77 ms，无法填平当前约 108 ms 的 TTFT 差距。其主要作用是节省 prompt arena，给更宽图和并发驻留留预算。主体投影同为 M64 tile 时，256+255 与 512 仍共八个 GEMM M tile；单行化后的最后 head 在这两种策略中都仅一个 M tile；WY 的总 32-token 块数也不减。减少图重放有价值，但不能宣称所有权重读取自动减半。

## 实施顺序

1. head 的64/256真实模型功能门禁已通过；保留资源优化，暂不全局固定512。
2. M4/M256 全输出独立 F64 参考已完成，继续测实际 FP8 大投影已有库桥的完整成本；不重复已有无收益的 M1 首算法。同时接入按 input/version/量化契约共用结果，保留原 GEMM 与权重。
3. Attention 三路径的真实几何 fixture 数值与完整 append+compute 成本已完成。继续补实际 checkpoint 中间量、完整任务质量和未插桩服务收益，候选通过后按能力/shape/预算进入 tuning；不因局部 QT16 快而固定全局 tile。
4. full128 Prepared隔离已完成且出现质量退化，先定位数值差异；没有修复前不扩大质量跑数，也不把局部27%记为最终收益。
5. B/A、gate/up合并、packed state launch和ReplaySSM按各自成本预算补齐；继续按整请求和batch周期评价。

ReplaySSM保留为decode与推测状态容量方向，不能用来解释当前prefill差距。已有独立F32实验在48层链上，普通batch1窗口4约1.218倍、batch4窗口8约1.325倍；推测T3在batch1反而变慢、batch4约1.36–1.40倍。它们比较的是Triton参考链而非Rust整模型，不能转记成当前TPOT提升。[已有研究](cuda-replayssm-gdn-2026-10-12.md)和[作者说明](https://dao-lab.ai/blog/2026/replayssm/)均需区分输入记录、输出计算、flush与commit；接入前继续保留F32状态及完整回滚契约。

所有硬件参数接入 tuning：能力/SM 数、shape、数值 profile、driver/library、workspace 与可用预算共同决定选择。5090 只作验证。F32 state、独立参考、完整模型质量和未插桩服务测量保持验收要求。

## 原始证据

45 上共同目录：
`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/`

- `gdn-nvfp4-lt-runtime-perf-v5/status-first-cycle.json`
- `vllm-direction-review-2026-10-13/latest-source-delta.json`
- `vllm-direction-review-2026-10-13/cost-bounds.json`
- `vllm-direction-review-2026-10-13/decode-runtime-path-evidence-v1.json`
- 上一级 `vendor-fp8/checkpoint5-summary.json`（历史 FP8 库诊断）
- `gdn-wy-current-profile-v1/cuda-events.jsonl`，SHA a268b1e90f3cf228fb1717a50cdcb121bb20c185f81c168eb373c48530c76f0b
- `small-projection-ba-v2/readonly-baseline-failure-evidence-v1.json`
- `prefill-last-head-readout-v1/source-identity-v5.json`
- `prefill-last-head-readout-v1/validation-summary-v5.json`
- `vllm-direction-review-2026-10-13/prefill-source-audit-v1.json`

源码参考：[Qwen GDN](https://github.com/vllm-project/vllm/blob/3fe917abb13160d60bea5330399f5c09ca282348/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)、[runner 的最后行选择](https://github.com/vllm-project/vllm/blob/3fe917abb13160d60bea5330399f5c09ca282348/vllm/v1/worker/gpu_model_runner.py)。
