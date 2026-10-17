# Serial MTP GPU 采样实验（2026-10-12）

把 draft 的整份 logits 回读和 CPU greedy 扫描迁到 GPU，能稳定节省约 1.6%–2.0% 的单请求 TPOT；它不能覆盖当前与 vLLM 的主要差距。batch4 TPOT 的配对中位数退化 2.97%，区间跨过改善和退化，候选不满足默认启用条件，尚未合入 main。

共同测量规则见[性能基线方案](../../plans/performance/baseline.md)，较大的执行差距见[vLLM main 调查](vllm-main-execution-paths-2026-10-12.md)。本实验没有更新 native 基线，也没有测量最新 vLLM main。

## 候选与验证范围

- main 基点：`2289160987fefc9d2b3b1872119618c7cd04426c`。
- 候选源码树：`6ef818d13dde65be85c0f54d23f64ca12319b1d5`，持久保存在 45 的下述证据目录中。
- 发布版二进制 SHA256：`0e891b33d3cb99df8a2461a346bb1678211016c2be63d8509557db5c3ba12a42`。
- 两臂使用同一个二进制，分别设置 `INFER_CUDA_DEVICE_GREEDY=0/1`。baseline 为保留原回读行为的同构建对照，不能称作未经修改的 main 二进制。
- 调参按设备能力、driver/API/build、完整 kernel identity 和 rows/vocabulary 查表；搜索 tile 256/512/1024/2048，保留冷/热原始配对计时和独立反向确认。5090 只用作验证设备，没有按设备名称分派。
- GPU reducer 用无符号整数映射复现有限 F32 的 total-order，最大值相同选最低 token ID，并检查所有有效 logits 是否有限。只回读 token ID 和验证标记；第二步 draft 复用设备隐藏状态。
- 条件为已验证的 plain greedy 参数，非 greedy 仍走原 sampler。只接入 serial MTP draft；target verification、pooled drafting 和最终 engine sampler 没有接入。

生产 CUDA Clippy、格式、布局、源码 policy、测试清单检查通过。普通 CUDA 单测 45 项通过；另外执行了两项 GPU reducer oracle、32 步隐藏状态/输出逐位对照，以及真实 248320 词表的调参确认。其他 ignored GPU 用例未声称执行。

GPU oracle 包括词表补齐边界、跨 tile 平局、正负零、正负 subnormal、F32 极值，以及远离获胜 block 的 NaN/正负无穷。模型矩阵的六组对应请求全部输出一致，共每臂 168 个请求，包含 warmup、token IDs、EOS 和 finish 信息；这不等于任务质量或全部内部 state 的独立证明。

## 配对测量

使用 45 上的 Qwen3.8-27B-NVFP4，MTP=2，greedy，每请求 64 个输出 token。short/long/batch4/hot_long 各一次排除计时的 warmup 和三次同进程重复。有效 context 32768、max-num-seqs 16、显存比例 0.88、prefix cache 开启、禁用运行中 autotune；所有前缀复用量、公共图配置和 live 进程开关已核验。

共运行 p0–p5 六组新进程 AB/BA 配对。p3 的外部 GitHub Desktop 图形显存从 118 MiB 变为 124 MiB，保留结果但不计入合格样本；补测 p5 后得到 p0/1/2/4/5 五个合格单位。没有停止或修改外部 GPU 进程。

以下为这五个单位各自 candidate/baseline 变化的中位数，负数表示延迟降低。区间为对五个配对中位数穷举 5^5 次重采样所得的描述性 95% bootstrap percentile 区间，不把同进程三次重复当独立单位。

| 场景 | TPOT 变化 | TPOT 区间 | TTFT 变化 |
|---|---:|---:|---:|
| short | -1.98% | [-2.48%, -1.49%] | +5.01% |
| long | -1.81% | [-2.28%, -1.67%] | -0.09% |
| batch4 | +2.97% | [-3.71%, +4.54%] | -1.21% |
| hot_long | -1.57% | [-2.04%, -1.38%] | +0.31% |

batch4 group makespan 中位数 -0.69%，区间 [-2.76%, +1.35%]；它不能替代每请求 TPOT 判断。没有测得 long prefill 的显著改善。

事先计划要求 short/long/hot_long 至少 2% 中位 TPOT 改善、batch4 TPOT/makespan 不超过 2% 中位退化。本轮未满足。单请求收益存在，但不批准默认启用；batch TPOT 的退化方向也尚不能以五组样本定论。保留候选供后续完整 GPU readout 集成使用，不用该局部改动宣称逼近 vLLM。

## 证据与下一步

持久目录（45）：

`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/device-greedy-mtp-v1/`

关键证据：

- `greedy-integration-v6-snapshot/`、`greedy-integration-v6-identity.json`、`infer-v6`、`cuda-tests-v6`；
- `greedy-integration-v6-tests-result.json`、`greedy-integration-v6-tests.log`、`frozen-cache-v6.json`；
- `paired-v6/plan.json`、`extension-p5.json`、逐臂 report/server/runner/telemetry/environment/execution，以及全部 qualification/summary；
- `paired-v6/decision-v1.json` 与 `completed.json`。前者绑定各原始 summary 和 qualification 的 SHA256，并记录未放行决定。

下一步先处理同一轮的完整 GPU 数据流和 pooled path，再验证真正的分块 GDN/投影融合。只消除两个 draft 的回读，不可能覆盖 target verification 和最终 engine 的完整 logits 回读，也不会减少长 prefill 的递推状态工作量。现有 engine 的 `OutputShape` 强制 vocabulary 长度的有限 logits，并在后端 accepted tokens 后再采样一个 token；后续 GPU token readout 必须显式扩展这一契约，不能伪造稀疏 logits 或挪用 accepted-token 字段绕过检查。
