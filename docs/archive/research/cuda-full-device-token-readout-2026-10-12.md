# CUDA 完整 GPU token readout：五组配对验收

本轮将精确 greedy 选择从 draft 扩展到普通 decode、serial/pooled target verification 和 pooled draft。27B 服务五组独立配对均通过：短请求 TPOT 下降 4.2%，长请求下降 4.9%，batch-4 下降 9.6%，缓存长请求下降 4.5%。这些是同一 native 制品开关对照，尚未完成最新 vLLM main 的跨引擎性能验收。

## 源码与比较边界

- 父提交：81d9e15b50fd350d0cc67f6cc0118b90c94214a6；测试运行时代码树：ae3302e5a2055a39e4ce63186c3cb82454c10ec6。合入前只补充文档，运行时代码逐文件比对冻结制品。
- release 二进制 SHA-256：a933a76e4556438344e68c5e56021e647672b403c060c7afb946abc40026eb08。
- 冻结 tuning cache SHA-256：d212f19361deeb8adf7cd321a268080d047920bcc5fa8973a251abeb404bcb7f。
- 2026-10-12 在 45 上直接拉取官方 upstream/main，vLLM 源码为 f26d9f432ced7b1087d02d29bffed2249df1a18e。既有 vLLM 性能数据仍来自已安装的 0.31.0；本轮未构建、测量最新 main，不能将旧数据重命名为最新 main。
- 验证设备为 5090。实现按设备能力与测量结果选择参数，没有按 GPU 名称写死策略。

[上一轮仅优化 draft 的实验](cuda-device-greedy-mtp-2026-10-12.md)存在 batch 回退，没有默认合入。本轮保留那份失败证据，扩大优化覆盖并重新冻结、验证、测量。

## 改动与正确性

旧路径每个采样位置将完整 F32 vocabulary logits 传回 CPU，再由 CPU 选择 token。plain greedy 所需输出只有一个 token ID。新路径在 GPU 按 F32 total order 归约，最低索引处理相同值，并检查所有有效 vocabulary 元素有限；只传回 token ID 和有效性标志。不会改变 logits 计算精度或 recurrent state 精度。

公共 IR 增加 `ExecutionTask.plain_greedy` 和 `ModelOutput.next_token`。后者表示 accepted prefix 之后的下一位置，与已接受 token 分开表达。runtime 验证 Generate、合法 greedy 参数、空 logits、词表范围与输出容量；存在 sampling penalty 等不适用条件时沿用原 CPU sampling。

普通 decode、窄 target verification、pooled draft 可使用该路径。宽 prefill 的现有输出路径保留。必要 hidden readout 与 I32 token readout 共用一条 CUDA stream 和一次 completion fence，pinned buffer 在 fence 完成前保持所有权；graph 按引用关系先于底层资源销毁。draft hidden/state 追赶仍按既有提交与回滚合同处理，本轮没有迁移推测策略归属。

两阶段归约覆盖不整除尾部、正负零、subnormal、F32 最大值并列与非获胜 block 中的无效值。共享输出覆盖零偏移、非零偏移及 inactive slot；捕获过程中发现的零偏移 launch extent 问题已通过有界行索引表达修复，没有关闭边界检查。

## 硬件自适应 tuning

归约 tile 候选为 256、512、1024、2048。搜索与独立确认包含交换运行顺序的 cold/warm 测量，并执行数值对照；确认不通过时保留可移植 baseline。缓存键包含硬件 profile、driver 版本/build、两阶段 kernel fingerprint、rows 和 vocabulary，接入既有共享 tuning cache 与原子保存机制。

本机 rows=1/4/12、vocabulary=248320 的测量选择 2048；这是本次设备的缓存结果，不是默认硬件常量。fingerprint schema 为 `f32-total-cmp:lowest-index:all-finite:two-stage-u32-bounded-offset-v3`，避免重用修复前内核的缓存。

## 验证

- 显式 CUDA feature 的 production library strict Clippy、全部 CUDA examples 类型检查及 CLI CUDA release 构建通过。
- CUDA release 单元测试 45 项通过、64 项 ignored；从 ignored 项中另行执行了 9 项真实 GPU 测试，全部通过。覆盖 reducer 三项、pinned readback 两项、resident program 三项及真实 vocabulary tuning 一项。
- IR/runtime 单元与集成测试通过，包括新增的 backend next-token 输出契约三项。v14 的这部分源码与 v13 完全相同，identity 文件绑定对应测试日志及源码 hash。
- 小模型 GPU fixture 对普通/serial/pooled 验证、部分提交与 draft continuation 的 hidden/state 做逐位对照；实际 27B 服务验证的是 emitted token/EOS/finish 一致性，没有宣称实际模型全部 hidden state 逐位比较。
- 扩展 CUDA test/example strict Clippy 仍有未修改测试中的既有告警，记录在 `expanded-cuda-unit-clippy-diagnostic-v11.json`；不将 production library Clippy 结果描述为所有 target 的 lint 全绿。

## 服务配对

同一 release binary，唯一策略差异为 `INFER_CUDA_DEVICE_GREEDY=0/1`。Qwen3.8-27B-NVFP4、MTP=2、temperature=0、64 输出 token、effective context=32768、seq=16、memory=0.88、prefix cache 开启。每个进程包括 warmup 和三次测量重复；五组独立进程对以 A/B、B/A 交替运行。

五组均符合预先固定的环境、后台负载、服务几何、冻结缓存与输入检查，没有排除组。每侧共 140 个请求（105 个测量请求），对应 token、EOS 与 finish reason 全部一致。

| 场景 | baseline TPOT 中位数/ms | GPU TPOT 中位数/ms | 配对变化中位数 | 95% 配对 cluster bootstrap 区间 |
|---|---:|---:|---:|---:|
| 短请求 | 11.214 | 10.737 | -4.21% | [-5.00%, -4.09%] |
| 长请求 | 10.938 | 10.408 | -4.94% | [-5.06%, -4.67%] |
| batch-4 | 14.966 | 13.428 | -9.62% | [-10.93%, -5.31%] |
| 缓存长请求 | 10.336 | 9.867 | -4.52% | [-5.17%, -4.38%] |

变化列先在每组内部计算百分比，再取五组中位数；不是表中两个独立中位数的比值。区间按独立配对作为 cluster，枚举 5^5 个重采样组合；进程内重复没有当作独立进程。

batch-4 group makespan 的配对中位数下降 6.19%（区间 [-9.21%, -5.39%]），baseline/candidate 中位数为 1886.095/1749.418 ms。长请求与缓存长请求 TTFT 分别变化 +0.37%/+0.31%；短请求 TTFT 区间跨零。本轮收益主要出现在 decode，不宣称普遍改善 TTFT。

本项预先约定的默认合入条件为短/长/缓存长请求 TPOT 配对中位数至少改善 2%，batch TPOT 和 makespan 无超过 2% 的中位数回退。本轮通过这些条件。跨引擎矩阵、尾延迟及最新 vLLM 复测仍按[性能基线方案](../../plans/performance/baseline.md)验收；本项结果不更新既有正式基线，不将全局性能目标标记完成。

## 可复算证据

持久证据位于 45 的 `/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/full-gpu-readout-v2`：

- `cuda-integration-v14-source`、`cuda-integration-v14-identity.json`：冻结源码、逐文件 hash、运行器与继承的 IR/runtime 测试身份。
- `infer-v14`、`frozen-cache-v14.json`：实际测试 binary/cache。
- `cuda-integration-v14.log`、`cuda-integration-v14-gpu.log`、`cuda-integration-v14-gpu-result.json`：构建、host/CUDA 和实际 GPU 测试结果。
- `paired-v14/plan.json`、每侧 environment/execution/cache/server logs、raw report、qualification 与 summary：五组原始证据。
- `paired-v14/decision-v1.json`：每组变化、统计汇总、制品身份及原始 JSON 的 SHA-256 清单。

输入 canonical SHA-256 为 855ee860b55e546ea92890deff76dcb995d410c8b4d089f8ff6e5c76f174b464。代码构建、正确性验证和正式服务测量分别运行，未与模型性能测量重叠。用户 GPU 进程未被停止。

## 下一步调查

最新 main 的实际跨引擎基线需要重新构建和固定制品。native 长 prefill TTFT 与 batch group makespan 仍有明显优化空间；下一轮先用阶段计时验证多请求 prefill 是否串行，以及 projection、GDN 与状态追赶各自占比，再选择收益最大的单项。旧诊断中 FP32 chunk GDN 的吞吐更高，但 token 数值对照不通过，仍不作为默认优化合入。
