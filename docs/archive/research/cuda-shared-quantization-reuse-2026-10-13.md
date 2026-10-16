# CUDA 相邻 SwiGLU 量化复用（2026-10-13）

本轮补完相邻 gate/up 投影的输入量化复用，代码提交为
`4fafeb78593cb992a7f9b85994269329dec53815`，基线为
`08969ed2a931abe6fea9b985cdfb5be6671a4bf2`。
开发、编译、GPU 验证和推送均在 `r@192.168.1.45`，直接进入 main。
[代码 CI](https://github.com/1994/inference.rs/actions/runs/38356244761) 的 10 个 job 全部通过后，
再运行本轮性能对照。

完整 gate/up 图有稳定的局部收益；单请求服务耗时收益较小。
256 题原质量评分通过，但四路服务的严格 token 一致性门槛未通过，
不能宣称所有端到端场景已验收，也不能把局部图的百分比当作整模型收益。

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

## 数值与状态验证

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

## 256 题任务质量

使用冻结 GSM8K 256 题集合与原评分器，MTP=0、并发 4、每题最多生成 512 token，
保留自然 EOS、原题目顺序、zero-shot/eight-shot 各 128 题及原非劣门槛。
两版均为 **245/256（95.703125%）**；zero-shot 为 119/128，
eight-shot 为 126/128，逐题评分回退与提升均为 0。
两版均有 6 条生成达到长度限制。原评分器的 95% 单侧回退率上界为
1.1634%，低于声明的 2% 门槛，`quality_accepted=true`。
这一结论只覆盖该固定集合与生成预算。

两版有 48/256 条完整 token 流不同。同一候选二进制、同一缓存重跑前 16 题，
仍有 6 条 token 流变化，证明该服务配置存在同一二进制的生成波动；
这个有界诊断不能证明所有跨版本差异都由调度引起。
两版服务配置读回及实际记录的 resident GEMM 选择一致。
上述同一二进制、固定 capture/输入的数值与状态对照通过，
但不能据此宣称整个并发服务生成逐位一致。

质量启动脚本原本额外要求完整 token 流一致，该额外断言失败。
原评分门槛没有修改，额外失败与诊断原始记录均保留。

## 完整 gate/up 图成本

计时范围包含 **输入量化 + gate GEMM + up GEMM**。
实际首末 NVFP4 层、首末 FP8 层，M=1/4/64/256；
两版共享相同物理输入、输出、权重及工作区。
所有独立量化、完整输出数值检查先通过，再进行 3 轮 AB/BA、
cold/warm 计时，共 192 条成本记录。
每格合并同编码首末层与各轮/顺序的图耗时中位数，正数表示降低延迟。

| 编码 | M | warm 延迟降低 | cold 延迟降低 |
| --- | ---: | ---: | ---: |
| nvfp4 | 1 | 1.87% | 1.79% |
| nvfp4 | 4 | 2.28% | 1.74% |
| nvfp4 | 64 | 5.54% | 4.82% |
| nvfp4 | 256 | 8.95% | 8.00% |
| fp8-channel | 1 | 4.43% | 4.27% |
| fp8-channel | 4 | 4.56% | 4.37% |
| fp8-channel | 64 | 5.16% | 4.86% |
| fp8-channel | 256 | 5.76% | 5.70% |

cold 包含计时外的 L2 清理；warm 不清理 L2。
这些数字只代表该完整投影对的图成本，原记录的
`performance_accepted=false` 表示其本身不授予整模型服务验收。

## 五轮服务对照

两个 CLI 来自各自冻结源码；相同原 checkpoint、F32 state、FP8 KV、
冻结调优缓存、输入与 profile。MTP=0，三种请求均生成 64 token，
prefix cache 开启但正式请求复用量均为 0，每个场景有 warmup 和 3 次正式 repeat。
顺序为 AB/BA/AB/BA/AB；合计每版 90 条正式请求。
硬件计时前仅要求 30 秒内连续 3 次 GPU 利用率不超过 2%，没有终止其他进程；
所有 heavy run 使用仓库 safe-run，host memory 限制 32 GiB、swap=0。

表中为五轮场景中位数的中位数；括号为延迟降低百分比。
TPOT 是每个输出 token 的时间，总耗时是请求组 wall time。

| 场景 | TTFT：基线 → 候选 | TPOT：基线 → 候选 | 总耗时：基线 → 候选 | 完整 token 门槛 |
| --- | --- | --- | --- | --- |
| 短提示 | 40.702 → 42.537 ms（-4.51%） | 16.135 → 15.997 ms（+0.86%） | 1057.471 → 1050.513 ms（+0.66%） | 通过 |
| 511-token 长提示 | 184.655 → 179.900 ms（+2.58%） | 16.121 → 16.007 ms（+0.71%） | 1199.632 → 1189.243 ms（+0.87%） | 通过 |
| 四路请求 | 162.534 → 163.712 ms（-0.72%） | 19.342 → 19.275 ms（+0.34%） | 1384.403 → 1385.980 ms（-0.11%） | 未通过；8 条不同 |

- 短提示 TTFT：五轮延迟比范围 0.88302～1.14220。
- 短提示 TPOT：五轮延迟比范围 0.99033～0.99512。
- 短提示 总耗时：五轮延迟比范围 0.99174～0.99781。
- 511-token 长提示 TTFT：五轮延迟比范围 0.96220～0.99757。
- 511-token 长提示 TPOT：五轮延迟比范围 0.99153～0.99772。
- 511-token 长提示 总耗时：五轮延迟比范围 0.99026～0.99467。
- 四路请求 TTFT：五轮延迟比范围 0.97898～1.01942。
- 四路请求 TPOT：五轮延迟比范围 0.99190～0.99789。
- 四路请求 总耗时：五轮延迟比范围 0.98960～1.00947。

短提示 TTFT 中位数由 40.702 ms 升至 42.537 ms，慢 4.51%；
其轮次波动较大，最后一轮延迟比 1.14220 超过原 compare 的 1.10 上限。
因此短提示虽然 decode 和总耗时改善、token 门槛通过，
也没有满足每轮全部延迟指标通过的要求。长提示五轮的这些延迟比均小于 1。

所有轮次与原严格 compare 结果均保留，没有重跑筛选或删除 token 不同的样本。
四路场景累计 8 条正式输出流不同，
因此该场景耗时只作为观察数据，不能当作已验收收益；
整体严格 token 门槛为 `global_strict_token_gate_accepted=false`。
短提示、长提示的 token 对照通过；小于 1% 量级的总耗时改善应按该测量范围理解，
未据此宣称一般负载都有稳定收益。

为隔离本次复用，计时使用冻结选择并关闭在线调优；
实际日志为 `autotune-disabled`，主要采用已有保守配置。
五轮实际 resident GEMM 选择全部一致，缓存与二进制 hash 不变。
这不是此前最佳调优 C5 prototype 的比较，也没有重新运行 vLLM；
不能用这些结果更新此前的 vLLM 差距结论。
生产路径保留现有自动调优接口，此次没有新增固定 GPU tile。

## 构建记录与可复核证据

45 上证据目录：
`/home/r/benchmarks/inference.rs/shared-quant-reuse-2026-10-13/`

关键文件：

- 源码/构建：`preparation.json`、`compiled-source.json`、`build-result.json`、
  `cli-build-result.json`、`code-ci.json`、`final-audit.json`。
- 数值：`pair-numeric.log`、`model-numeric.log`、`numeric-result.json`。
- 质量：`quality/paired-summary.json`、两版完整请求/评分报告、
  `quality/token-differences.json`、
  `quality-candidate-repeat/repeat-summary.json`。
- 成本与服务：`pair-cost.log`、`pair-cost-summary.json`、
  `paired/plan.json`、全部 10 份服务报告、5 份原严格 compare、
  `paired/numeric-by-case-p*.json`、`paired/summary.json`、
  `paired/continuation-policy.json`。

最终复核 598 个编译相关文件，无改动。
测试二进制 SHA256：
`e78ab31c8b5dd7c96cafc4404a1220a6759e067e637e654dccb0e68d1d3b7974`。
基线 CLI SHA256：
`c51106bcef70ef737bb6574e2232648b0d4955b36a3eacd8058f29c4ccd8b601`。
候选 CLI SHA256：
`a944bc62f20f3546aaad0b746188e4af4dde99d5c9924271c1fb2e5eb2edeb37`。

首次两个源码快照复用同一 Cargo target 目录时，候选产物被错误复用。
通过二进制 hash 相同和异常短的构建时间发现后，保留该失败产物，
清理相关 release 包并重新构建，得到不同的已确认候选产物；
正式质量与性能均使用修正后的 CLI，没有采用错误产物的性能数据。
早期编译/lint/策略失败保留在 `build-attempt-0/`。
质量额外 token 断言失败、首轮服务严格断言失败也保留，未记作通过。
