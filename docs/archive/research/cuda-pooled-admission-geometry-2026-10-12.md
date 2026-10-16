# Pooled resident 图几何与显存准入修正

新建 pooled target 只捕获窄 prompt 图，原准入预算却同时计入宽、窄图。修正让 capture、projection workspace 和新建报价共享 ProgramGeometry；缓存复用保留旧程序的预算下界，因为丢弃 verification 不会销毁宽 prompt 图。该变化修正实际资源计费，数学和精度保持不变。

## 原始证据

既有诊断捕获 122 次准入拒绝。首个拒绝时 needed=1,118,322,192 bytes，physical available=894,177,280 bytes，active_states=3、maximum_states=62，pool width=4，尚有 free slots。这证明实际限制来自显存报价，并非请求上限为 3 或 slot 已满。needed 是报价，不是实测增量分配。

旧行为：4 请求同时发出，三个先运行，第四个等待约一秒。新建 pooled 程序实际上只持有 64 行 prompt 图，报价却按 256+64 行 prompt arena 计算；用于 MTP priming 的 draft 仍始终构建 private 程序，必须独立按真实策略收费。

## 资源合同

- 图几何是 verification width 与 wide/narrow prompt widths，0 表示没有捕获；不是设备名判断。
- capture、共享 projection workspace、activation arena/recurrent scratch/readback 报价使用相同描述。prefill 无 prompt 图时，保留 verification fallback 的预算。
- external/fusion lane buffer 仍按 configured width 分配并计费；不因省去某图而假设所有缓冲都消失。driver graph headroom 保持不变。
- cached shell 按与 take() 相同的 capacity/readout 匹配，报价至少为 retained charge 与 fresh requested charge 的较大值。private 程序转入 pool 后仍保留的宽 prompt 图不按窄新建程序重计。
- SPI state_reservation_bytes_for 与 reserve() 使用相同报价；free slot 不意味着 prefill 私有程序没有内存成本。
- width/kernel 继续由现有设备能力和 tuning 选择；5090 是本次验证设备。

## 验证与身份

CUDA release 生产 library Clippy、CUDA examples check、47 项 CUDA unit tests 和 release CLI build 通过；10 项真实 GPU 测试通过，包括 greedy/readback、MTP continuation/partial commit、inactive slots，以及新的 wide/narrow prompt fixture。

新 fixture 比较 63 行 prompt hidden：保留宽图的一次 replay 对比新 pooled 的 32+31 两次 replay；随后 16 步 decode hidden/logits 逐位相同，并确认 forfeit_verification 后旧程序仍保留 64 行宽图，新程序只有 32 行窄图。两项 CPU 字节断言核验省略宽图和 prefill verification fallback 的报价。

- frozen source tree：`2e155a82a01956c4a3d8c74667dd67ba051952e0`，基点 `4d962de2efbe4536699b0b0e4c974cfb0500f484`；文档更新不改变已测 production/test 源码。
- production executable SHA256：`67afa29c796d184d06f692bb90f130298c634bcaf24d1547c3aae9cd5e4baf9e`。
- CUDA test executable SHA256：`24cfba30a9d8cce898d5bcc9d6a16242c78bfe1a639df6ab6a91dad258e57647`。
- frozen tuning cache SHA256：`d212f19361deeb8adf7cd321a268080d047920bcc5fa8973a251abeb404bcb7f`；每进程独立复制，前后字节不变。
- baseline executable 是上一项已验证 full GPU readout 制品，SHA256 `a933a76e4556438344e68c5e56021e647672b403c060c7afb946abc40026eb08`；baseline source tree `ae3302e5a2055a39e4ce63186c3cb82454c10ec6`。
- 45 上完整证据目录：`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/pooled-admission-geometry-v1`。`paired/plan.json` 保存预定主指标、所有参数和 runner 身份；`paired/decision.json` 保存全部配对统计；`build-v5.log`/`gpu-result.json` 绑定构建与设备测试。前几次编译中发现并修复的模块导出/未使用接口与测试拼接问题保留在对应日志中。


## 实测

配置：Qwen3.8-27B-NVFP4，MTP depth=2，temperature=0，64 输出 token，有效上下文 32768，max-num-seqs=16，显存比例 0.88，prefix cache 开启，no-autotune。public prefill=256，窄图=64；两边只有冻结 executable 不同，环境与 tuning cache 完全相同。5 个新进程配对，AB/BA 交替，每进程 3 个测量 repeat，全部 warmup 保留且从稳态计时排除。5 对均通过输出、配置、硬件 UUID、背景进程/显存、cache 和执行文件身份检查，没有排除样本。

每边 140 个逻辑请求（包含 35 个 warmup、105 个测量请求）的 token、EOS 和 finish 状态全部一致。不能把它理解为全模型 logits/KV 逐位证明或任务质量得分。

| 场景 / 指标 | baseline ms | candidate ms | 配对变化中位数 | 95% empirical cluster bootstrap |
|---|---:|---:|---:|---:|
| short TPOT | 10.750 | 10.761 | -0.01% | [-0.81%, +0.51%] |
| long TPOT | 10.409 | 10.412 | +0.12% | [-0.36%, +0.32%] |
| hot prefix TPOT | 9.863 | 9.874 | +0.11% | [-0.21%, +0.15%] |
| batch4 TPOT | 13.322 | 13.880 | +4.33% | [+0.44%, +7.02%] |
| batch4 TTFT median | 167.512 | 212.063 | +26.15% | [+22.06%, +27.39%] |
| batch4 observed max TTFT | 1091.653 | 219.683 | -80.21% | [-80.75%, -79.86%] |
| batch4 group makespan | 1774.935 | 1106.133 | -37.76% | [-39.96%, -36.77%] |

batch4 每组固定 256 输出 token，group throughput 从 144.23 到 231.44 tokens/s；配对中位增幅 60.66%，95% 区间 [58.14%, 66.55%]。区间以整个服务进程对为单位做 exact empirical resampling，未将相关的 token 或同进程请求当独立样本。

单请求场景没有可主张的稳定 TPOT 加速；long TPOT 的区间上端超过 2%，因此本项中位数门槛通过不代表严格证明了该场景的 2% 非退化上界。observed max 是本次有限矩阵的最大值，不是持续负载 p99 SLO。

首个 batch4 warmup 在两边分别为 4.272 和 4.882 秒；额外请求的首次 capture 有成本，不能只保留稳态收益。完整 warmup、逐请求测量和失败启动检查都留在证据包中。


## 取舍与剩余工作

本项的主指标预先设为 batch4 group makespan；其他场景 TPOT 不允许中位数退化超过 2%。四请求同时执行会增加每请求间隔、prompt 首 token 的中位数；不能把吞吐/排队尾部改善写成所有延迟均改善。保留 warmup 的 capture 成本，稳态结果不包含加载/JIT。

独立诊断确认：只含 decode 的执行提交，原最多 3 个请求，新程序最多 4 个请求；对应四个不同的 slot，pool_disabled=false，全部执行成功。原诊断 122 次准入拒绝，新诊断 0 次。诊断的 28 个请求输出与 p0-candidate 完全相同。记录在 admission-diagnostic/summary.json；该带 execution profile 的运行只用于归因，不混入五组正式计时。

最新 vLLM main 于 45 直接 pull --ff-only upstream main，当前 f26d9f432ced7b1087d02d29bffed2249df1a18e；仍没有把这份 main 当作已构建的性能基线。历史 vLLM 0.31.0 的数据不可直接冒充本轮对照。整体逼近 vLLM 的目标仍未完成，见[执行路径调查](vllm-main-execution-paths-2026-10-12.md)。

下一项应按阶段计时核验 GDN chunk 算法、target/draft hidden 和提交的数据流、合并 projection 量化契约。Qwen GDN 的 QKV/Z、B/A 合并减少调用，但 NVFP4 不同全局输入 scale 不能直接取最大值后宣称等价；必须保存 checkpoint 精度和状态/输出证据。按同一资源约束继续完整跨引擎矩阵，分别判断 TPOT、TTFT、吞吐和尾部。
