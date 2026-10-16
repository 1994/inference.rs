# CUDA 小投影的解码路径一致性（2026-10-13）

后续入口修正（2026-10-13）：本记录两个候选均未通过完整质量门槛。`INFER_CUDA_PRECISE_SMALL_DENSE` 和 `INFER_CUDA_POOL_SMALL_F32` 的生产入口现已移除；候选内核和 A/B hook 只编译进 CUDA 测试程序。下文环境变量描述属于旧实验制品，保留用于复核历史证据。

原服务实现的小 BF16 投影在私有 decode 使用 F32 GEMV，在 pooled decode 使用
两项 BF16 输入分解的 Tensor Core GEMM。并发请求开始时可能先执行一步私有 decode，
再绑定 pooled slot，因此同一二进制的生成会受这条算术路径选择影响。
本研究先解决可复核的计算一致性；没有端到端性能验收，实验默认关闭。
短提示 TTFT 慢 4.51% 的因果来源仍未确认，不能用这个数值差异解释全部耗时回退。

## 原路径的定位

原候选 `8f9cce72c7142e2822de7b21e7e29c7dc09d609f` 的同二进制/缓存服务诊断，
原前 16 个输入、output=512、并发=4 重复两次，均为 14/16，却有 8 条完整 token 流不同。
16 个请求两次都各执行一次完整 prefill。不同流对应第一次 decode 是否使用私有路径；
其他直接进入 pooled 的请求流一致。该相关性不是单独的因果证明。
原完整 256 题质量评分已通过，严格 token 流断言仍失败，见
[相邻输入投影复用研究](cuda-projection-quant-reuse-2026-10-13.md)。

Qwen3.8-27B-NVFP4 的 48 个 GDN 层包含 96 个 BF16 `in_proj_a/b`，真实形状为
N=48、K=5120。隔离测试使用冻结 511-token 输入的 179-token 前缀，完整模型，
MTP=0、原冻结缓存、pool width=4/capacity=1024；三条路径消费同一组 8 个 token。

把私有小投影改为原 pooled 两项分解后，完整 5120 维 hidden、248320 维 logits、
所有 F32 state 和 FP8 KV 与 pooled 逐位相同。原私有 GEMV 对照有约 203 万个
输出元素不同。这证明该 checkpoint/轨迹下的小投影算术足以解释整模型路径差异。

## 独立精度检查及失败记录

全部 96 个原 BF16 权重、普通/零/稀疏/宽指数四种输入，检查 M=1、M=4
及原私有 GEMV 的全部输出，共 384 个 fixture、110592 个输出。
参考直接从 checkpoint 的 BF16 原始 bits 移位解码，并执行独立 F64 dot；
没有用候选 GPU 输出来生成参考。

门槛沿用原 small GEMM 测试：`abs(expected)*2e-5+5e-4`，严格小于门槛，
所有输出必须有限；同一行输入的 M=1/M=4 输出还须逐位一致。

| 实现 | F64 超限输出 | M=1/M=4 行差异 | 处理 |
| --- | ---: | ---: | --- |
| 原 pooled 两项输入分解 | 303 | 0 | 精度检查失败 |
| 三项分解注入同一跨块累加器 | 1158 | 0 | 精度检查失败 |
| 三项分解、各分量块内独立 MMA、F32 补偿块求和 | 0 | 0 | 该完整参考检查通过 |

前两次失败全部发生在宽指数输入上，原 F32 GEMV 对照全部通过。
仅增加一个输入分量不能解决累加顺序中的舍入；最终版本先分别从零计算每个
128 列 K 块内的 high/low/tail 乘积，再组合该块，使用 F32 补偿求和累加各块。
该版本的最大门槛归一化误差为 0.064792（包含原 GEMV 对照），低于 1。
F32 输出/状态、BF16 权重、FP8 KV 的存储不变。

## 默认关闭的实现与整模型验证

`INFER_CUDA_PRECISE_SMALL_DENSE=1` 或 `true` 才启用；私有与 pooled 的
小 BF16 投影一致使用补偿版本。原有宽词表投影、FP8/NVFP4 量化投影使用原入口。
选择依据为既有小投影几何关系，未使用 GPU 名称或模型名称。
默认保留原算术，完整质量、CI 和成本验收完成前不提升为默认策略。

最终隔离测试先用同一完整 F32/FP8 prefix state seed 原 GEMV 对照、候选和 pool，
核对三份初始 state 完全相同，再完成 8 步共同轨迹。
候选与 pool 的 2027520 个完整输出以及首末步完整 F32 state/FP8 KV 逐位相同。
这项测试验证路径一致性，不能替代整模型任务质量。

CUDA release 严格 clippy、布局/策略检查通过；默认测试 68 通过、81 ignored。
最终测试二进制 SHA256：
`0cd855212083f2d74d9fb5ecbef1208845fa0e27bbd81d09261842b13707e93e`。
实验 CLI SHA256：
`5ce8893e61358122374ff7023da84739679fe03e10496732fbb352c264ef3a05`。
编译相关文件在冻结前后均核对，未把 CUDA 未启用的构建当作 GPU 验证。

## 真实服务复现与待完成门槛

同一实验二进制、原缓存、原前 16 个输入及原生成配置，host execution profile
记录两轮服务执行；没有添加 GPU 观察核，不用于性能计时。
两轮都为 13/16，全部完整 token 流相同。
其中 6 个请求的首次私有 decode 使用情况确实不同，仍生成相同流；
按每组唯一输入长度匹配状态，并保存全部 profile 和映射记录。
这个有界功能诊断不替代原 256 题质量，不表示得分提高。

原 256 题已完成完整配对，运行 19 分 6 秒：基线使用冻结的 8f9 CLI，
候选使用上述实验 CLI；保持原评分器、题目顺序、max_output=512、自然 EOS、
并发=4、MTP=0、原缓存。结果 **243→242/256**，zero-shot 均为 117/128，
eight-shot 126→125/128。退步 3 题（87/1089 zero-shot、781 eight-shot），
提升 2 题（241/1119 zero-shot）；原 95% 单侧退步率上界 **3.000754%**
超过原 2% 门槛，`quality_accepted=false`。

202 条完整 token 流不同，额外严格断言保留，整体也未通过。
两版的完整请求、EOS/长度限制及原评分器输出全部保存；没有筛选重跑。
精度和有界路径一致性通过不能抵消任务质量失败，该实验继续默认关闭，
不进入正式服务性能验收，`performance_accepted=false`。
后续原生 F32 批量 GEMV 对照使用既有 kernel，先检查完整独立参考和路径一致性，
再决定成本及任务质量工作，不把这次失败当作新基线。

## 原生四行 F32 对照的实际入口

后续核对确认：无 MTP 的连续批处理通过 `record_slots` 直接消费既有
`[slots,K]` 矩阵，已经不包含逐 lane pack/unpack。早期四独立指针实验
挂在旧的 `record` 入口，实际捕获计数为 0，另有 cuTile 编译失败。
v18 的源码、失败日志完整保留；不能把设想的 768 次 kernel 消除算作收益。

当前默认关闭的 `INFER_CUDA_POOL_SMALL_F32=1/true` 在真实 slot 入口复用
既有 `linear_batch::batched::dense`，仅作用于四槽的小 BF16 dense 投影。
它沿用私有 GEMV 的既有 tile，保持 F32 累加/归约顺序；宽词表、其他
dtype 和其他槽数保留原入口。没有新 GPU kernel、额外拷贝或 GPU 观察核。

v21 的全部 96 个真实权重 × 四个输入 fixture × 四行，完整独立 F64
检查共 73728 个输出，超限为 0，四行与分别执行原私有 GEMV 的逐位差异为 0。
最大门槛归一化误差为 0.064792；这不表示成本或任务质量通过。
v21 的真实四槽捕获计数为 96。四路不同轨迹经过八步，包含不同 active mask、
重新绑定，5575680 个完整 hidden/logits 输出差异为 0；首末步四槽完整
F32 state 和 FP8 KV 也全部逐位一致。默认单元测试 68 通过、84 ignored；
CUDA release 严格 clippy、布局和策略检查通过。
冻结 test 二进制 SHA256 为
`c9c86d38e2b84048379c95d61bbd1fcbf379a2a994566f28fb71abfe03bf8c47`。
v21 服务二进制 SHA256 为
`0087a99b36f1ca509748f420d0d89531ab3ccd1fc8ac8249259f842f750977b9`。
原前 16 个请求、原生成配置重复两轮，均为 14/16，完整 token 流差异为 0。
该诊断位于 `same-binary-pool-f32-16-v1/`，仍不替代原完整质量和性能门槛。

v22 增加仅用于诊断的完整 96 小投影冷/热缓存配对成本测试；实际 slot
入口没有 pack/unpack，因此两图都直接读写已有矩阵，不计算不存在的拷贝收益。
生产入口还排除带 activation quantization contract 的权重，保留其原编码检查。
这两项改动不作为 v21 CLI 的已编译内容；后续服务工作另冻结 v22 源码二进制。
v22 最终服务二进制 SHA256 为
`0e5e1143345b973ca268ed23d0617a4c2f4d61ecf5b391e29c8c00ded94c71fd`，
同样的原前 16 请求两轮均为 14/16、完整流差异 0，记录在
`same-binary-pool-f32-16-v2/`。按照 `slot_replay` 的实际顺序，用执行后槽位
分类（执行前无槽位的任务也可能在本次调用中绑定并执行 pooled graph），
最终两轮确有 4 个请求的私有路径不同，完整流仍全部相同。
首版 v21 两轮对应为 2 个，原补偿实验为 6 个；三个版本的原始 profile 和
`grouping-correlation-v2.json` 均保留。随后开始 `quality-pool-f32-v2/` 原 256 题配对。
代码提交为 `864f55c6ba5de549815d8873cf250235a21b9311`；
[完整代码 CI](https://github.com/1994/inference.rs/actions/runs/38402296377) 的 10 项检查全部通过，
[文档 CI](https://github.com/1994/inference.rs/actions/runs/38402296746) 已通过。
原 256 题配对已完成。基线 **246/256**，候选 **243/256**；
zero-shot 120→119/128，eight-shot 126→124/128。退步 4 题
（1183/87 zero-shot、541/527 eight-shot），提升 1 题（1119 zero-shot）。
原 95% 单侧退步率上界 **3.539722%**，超过原 2% 门槛，质量失败。
Length/缺答案请求均保留：基线 5 个、候选 8 个。
204 条完整 token 流不同，原额外严格流断言也保留为失败。
两版完整请求、缓存/二进制身份以及独立评分器的原报告全部保留，
没有用之前的 243 分基线替换这轮 246 分，也没有筛选重跑。

F32 路径一致性和原独立 F64 检查通过，不能覆盖任务质量失败；
`INFER_CUDA_POOL_SMALL_F32` 继续默认关闭，不进入正式服务性能验收。
在代码 CI 全通过后，仅执行预先声明的三轮 96 投影冷热配对成本诊断，
不将该诊断登记为完整质量或服务性能通过。

三轮投影成本均已完成，冻结 test SHA256 为
`96ff3e960052c8409994243ed8d0a947af0599d7244205388f82704f5185e1b6`，
缓存未改变，每次进程前在有界窗口内取得三次 GPU utilization≤2%。
两图都包含 96 个完整真实 BF16 小投影，输入/输出为实际 slot 矩阵布局，
不含整个模型的其他算子，也不添加打包/解包收益。

| 缓存 | 原两项 BF16，ms | 四行 F32，ms | 耗时降低 | 三轮速度比 |
| --- | ---: | ---: | ---: | ---: |
| cold L2 | 1.678848 | 1.374944 | 18.10% | 1.217–1.224× |
| warm L2 | 1.671328 | 0.921952 | 44.84% | 1.812–1.813× |

表中耗时为三轮中位数的中位数。此完整局部路径确实更快，但原任务质量失败，
因此仍不接受为默认或正式服务性能结果。三轮全部日志与汇总保留在
`pool-cost-v1/2/3.log`、各自执行 JSON、`pool-cost-summary-v1.json`。
速度与质量的两个结论均保留，不能用局部加速抵消四题退步。

旧 SwiGLU 五轮数据另复核了原服务自身的 TTFT 记录（未重跑或筛选）：
客户端每轮中位数再取中位数为 40.702→42.537 ms，服务端为
36.279→37.211 ms。退化不全来自客户端统计，具体 GPU 算子、执行和
请求调度之间的因果关系仍未验证。活动请求的平均 SM 时钟约 2881–2902 MHz，
因此不能把加载前空闲时钟不同直接解释成请求内的速度变化。
分项记录为 `shared-quant-reuse-2026-10-13/ttft-component-audit-v1.json`，
不替代原性能或严格 token 流失败。

## 证据

全部开发/构建/GPU 工作在 `r@192.168.1.45`，使用原仓库 safe-run，
host memory=32 GiB、swap=0、30 分钟限制、串行运行，不终止其他 GPU 使用者。
证据目录：`/home/r/benchmarks/inference.rs/private-pooled-arithmetic-2026-10-13/`。

- 原路径/失败：`model-census-v7.log`、`small-dense-reference-v7.log`、
  `model-census-v8.log`、`small-dense-reference-v8.log`；最早测试读取了接口未返回的
  hidden，v3 的诊断入口失败保留，后续通过直接读取已有 GPU buffer 修正。
- 最终完整检查：`model-census-v10.log`、`small-dense-reference-v10.log`、
  `small-dense-precision-summary-v1.json`、`controlled-arithmetic-summary-v10.json`。
- 构建身份：`compiled-source-v10.json`、`source-v10.patch`、
  `resident-decode-path-source-v10.rs`、`clippy-v10.log`、`test-build-v10.jsonl`、
  `experimental-cli-v1.json`。v7 新测试源码恢复后与原 manifest 的 SHA256 完全匹配，
  单独保存在 `resident-decode-path-source-v7.rs`。
- 服务复现：`same-binary-precise-16-v1/` 中完整请求、两份 execution JSONL、
  `diagnostic-summary.json` 和 `grouping-correlation-v1.json`。
- 原质量配对：`quality-precise-v1/` 中原 256 题完整两版请求、缓存与 `paired-summary.json`（质量失败）。

- 原生四行 F32：`compiled-source-v22.json`、`pool-four-reference-v22.log`、
  `pool-four-model-v22.log`、`experimental-cli-v3.json`、`same-binary-pool-f32-16-v2/`。
- 完整代码 CI：`code-ci-v22.json`；原生 F32 质量失败：
  `quality-pool-f32-v2/paired-summary.json`、`rejected-pool-f32-experiment-v2.json`。
- 三轮算子成本预先计划：`pool-cost-plan.json`；不替代正式服务验收。
