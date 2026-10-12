# 最新 vLLM main 服务对照诊断（2026-10-12）

45 上的官方 vLLM main 已更新并实际运行，固定提交为 `3fe917abb13160d60bea5330399f5c09ca282348`。五组交替新进程对照中，native 的 short TPOT 为 vLLM 的约 1.29×、long TPOT 约 1.43×、batch4 完成时间约 1.72×；511-token 冷提示 TTFT 仍约 5.72×。目标尚未达到。

这些数字是诊断结果。五组中只有 p3 通过严格的后台进程及显存逐项相等检查；其余四组桌面 GPU 进程有小幅显存变化。热前缀实际复用量和数值轨迹也不同，未签发新的同条件性能基线。共同验收要求沿用[性能基线方案](../plans/performance/baseline.md)，没有放宽门槛或丢弃失败组。

## 被测制品与实际路径

| 对象 | 身份 |
|---|---|
| native 程序源码 | `b02a4fe9f47f7cbde8e88c5eb81a506eb4d408cc`，release CLI；当时仓库 HEAD `a4e97dda6706492ad5de967d869f0257fb6c72d2` 只新增文档 |
| native 二进制 | `serial-checkpoint-fusion-v1/infer`；SHA256 `052e351a8b1a97d5896f21779404e9ea08b64505743ad899f36280e96f44b819` |
| vLLM 官方源码 | `/home/r/code/vllm`；`git pull --ff-only upstream main` 后为 `3fe917ab`，工作区干净；2026-10-12 00:14:06 UTC 再次确认最新 |
| vLLM 官方构建 | `0.31.1rc1.dev289+g3fe917abb`；cu130 wheel，SHA256 `caa45d4e80c2803afe5173936573401df224d76521848d0348e293f97591f769` |
| vLLM 运行环境 | Torch 2.13.0+cu130、FlashInfer 0.7.0.post1、Triton 3.7.1；隔离的 vLLM/CUTLASS 4.8.0 目录，保留原环境 |
| 采集工具 | 干净 harness `2289160987fefc9d2b3b1872119618c7cd04426c`；该 revision 是报告工具身份，不能当作 native 或 vLLM 的程序源码版本 |
| 模型与设备 | 同一 `/home/r/models/Qwen3.8-27B-NVFP4` 制品；RTX 5090、driver 615.78.08；5090 仅作本次验证 |

native 的源码、编译文件哈希与程序哈希通过 `serial-checkpoint-fusion-v1/{compiled-source,shipping-identity,build-result}.json` 单独绑定。harness 的 `identity.source` 来自采集工作区；两边实际被测程序身份以上表和独立制品证明为准。

vLLM 的 Python 模块和 `_C_stable_libtorch.abi3.so` 均从隔离目录载入。GDN、V2 model runner、TargetDependentAR speculator 和 CUDA platform 四个源码文件与 clone 完整字节相同；依赖审计和 CUTLASS/FlashInfer GDN import 均通过。先前缺少 CUTLASS core 的失败尝试保留在 `paired/`，没有计入本表的 `paired-v2/`。

日志确认使用 V2 Model Runner、FlashInfer GDN prefill、FlashInfer CUTLASS NVFP4 GEMM。attention 使用 BF16 query、FP8 E4M3 KV，decode backend 为 XQA。缺少预编译 FlashInfer cubin 的提示触发启动期 JIT；测量发生在服务就绪与场景预热之后，未用 fallback 失败结果替代正常数据。[官方 wheel](https://wheels.vllm.ai/3fe917abb13160d60bea5330399f5c09ca282348/vllm-0.31.1rc1.dev289%2Bg3fe917abb-cp38-abi3-manylinux_2_28_x86_64.whl)

## 五组样本与资格

运行前固定五对进程，顺序为 native/vLLM、vLLM/native、native/vLLM、vLLM/native、native/vLLM。每场景一轮排除预热、三轮测量；每引擎共 35 个预热请求、105 个测量请求，其中 cold short/long/batch4 共 90 个请求，hot_long 共 15 个请求。所有请求均正常完成并生成 64 个非 EOS token，无截断。

共同请求采用相同 token 输入、greedy、MTP 深度 2，传入上下文上限 32768、最大序列上限 16、显存比例 0.88，启用 prefix cache。实际图宽度、池容量与 cache 分配仍由各引擎选择；相同传入上限不证明内部调度或显存字节分配相同。native inspection 回读 prefill width 256；vLLM 日志确认 token budget 2048。原始报告保留双方配置及其差异，未把这些实现差异消去。

native 显式设置 `INFER_CUDA_DEVICE_GREEDY=1` 并使用冻结 tuning table、`--no-autotune`。这是已实现路径的 opt-in 诊断，不是默认启用收益；冻结表 SHA256 为 `d212f19361deeb8adf7cd321a268080d047920bcc5fa8973a251abeb404bcb7f`，每次进程运行后均未变化。vLLM 使用本版本默认后端选择，并在启动阶段完成 FlashInfer 调优。

| 配对 | 启动前外部进程及显存相等 | 首次进程捕获时相等 | cold prefix 复用为零 | 热复用量相等 |
|---|---|---|---|---|
| p0 | 否 | 否 | 是 | 否 |
| p1 | 否 | 是 | 是 | 否 |
| p2 | 否 | 否 | 是 | 否 |
| p3 | 是 | 是 | 是 | 否 |
| p4 | 否 | 否 | 是 | 否 |

后台差异均来自同一 GitHub Desktop 图形进程（PID 27608），配对检查点显存变化 2–6 MiB；进程列表及其他进程的对应字段相同。没有据此豁免资格检查。运行器未发现外来 GPU compute 作业；用户已有进程和桌面均保留。统计表保留全部五组，p3 也只通过背景这一项，不能凭一组样本签发完整基线。

## 当前差距

以下 native/vLLM 比值为「五个进程中位数的中位数之比」，单位为毫秒。batch4 完成时间是四请求最后完成的时间；TTFT/TPOT 则先聚合该进程的逐请求样本。最后一列保留五组各自比值范围，不把 105 个请求当成 105 个独立进程。

| 场景 | 指标 | native | vLLM main | native/vLLM | 五组配对比值范围 |
|---|---|---:|---:|---:|---:|
| short（51 token） | TTFT | 47.529 ms | 33.618 ms | 1.414× | 1.384–1.744× |
| short（51 token） | TPOT | 10.846 ms | 8.424 ms | 1.288× | 1.273–1.289× |
| short（51 token） | 完成时间 | 735.689 ms | 570.975 ms | 1.288× | 1.275–1.304× |
| long（511 token） | TTFT | 357.379 ms | 62.489 ms | 5.719× | 5.622–6.039× |
| long（511 token） | TPOT | 10.448 ms | 7.286 ms | 1.434× | 1.410–1.441× |
| long（511 token） | 完成时间 | 1016.950 ms | 519.283 ms | 1.958× | 1.912–1.962× |
| batch4（四个短请求） | TTFT | 211.661 ms | 82.319 ms | 2.571× | 2.263–2.745× |
| batch4（四个短请求） | TPOT | 13.865 ms | 8.507 ms | 1.630× | 1.566–1.771× |
| batch4（四个短请求） | 完成时间 | 1103.157 ms | 639.515 ms | 1.725× | 1.698–1.737× |

单独看背景一致的 p3，short TPOT 为 1.273×、long TPOT 为 1.410×、long TTFT 为 5.622×、batch4 完成时间为 1.698×，同样没有接近 1.10×。这支持继续处理明显差距，但不填补五组合格复测、共同质量与完整资源对齐的缺口。

`summary.json` 还保存以整对服务进程为单位枚举 `5^5` 次重采样的诊断区间；样本只有五对，且四对背景条件不通过，这个区间不作为正式验收置信区间。

## 热前缀与数值轨迹

hot_long 输入 4095 token，声明共享前缀 3584 token。每次测量 native 实际复用 3584，vLLM 实际复用 1600。最新 vLLM 日志报告 attention block size 1600，以使 attention page 至少覆盖 Mamba page；align 模式的 checkpoint 与块边界会限制可复用状态。因此不能把 native 的 74.954 ms 与 vLLM 的 198.183 ms 热 TTFT 宣称为等量 GDN/kernel 加速。热请求总完成时间分别为 697.894 与 680.787 ms，也只描述本次缓存策略下的结果。[GDN checkpoint 实现](https://github.com/vllm-project/vllm/blob/3fe917abb13160d60bea5330399f5c09ca282348/vllm/v1/attention/backends/gdn_attn.py#L673)

native activation 为 F32、vLLM 为 BF16，recurrent state 均为 F32。每对 21 个测量请求的输出序列均不同；比较字段 `token_mismatches=21` 统计的是不同序列的请求数，并非错误 token 数。输出长度相同不代表固定 token 轨迹、相同 draft 接受率或任务质量相同。本轮没有代表性任务质量验收，也没有强制喂相同 decode 轨迹，不将服务比值归因到单个算子。

native `Completed` 事件包括 prefill 和其他 engine step；vLLM speculative counters 按 draft/accepted token 计数。汇总保留这些原始差分，但未将两种计数直接相除来声称相同 GPU 工作量或接受率。

## 优化判断与本轮否决结果

1. **长 prefill 优先改变 GDN 的递推工作量。** 511-token 冷 TTFT 约 5.72×；此前 FP8 准备微基准使用不同几何，其局部收益尚无端到端证据能覆盖该差距。vLLM 实际使用[分块 GDN](https://github.com/vllm-project/vllm/blob/3fe917abb13160d60bea5330399f5c09ca282348/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L234)。native 的现有量化 chunk 实现仍未过[完整模型数值检查](cuda-gdn-numerics-and-fp8-preparation-2026-10-12.md)，不能直接启用。真正的块内代数重排、state 重用和独立 BF16 数值 profile 是下一项算法工作；本轮没有 GPU profile 支持将 TTFT 的具体百分比划给 GDN。
2. **decode 继续减少 projection 与 MTP 依赖点。** vLLM 合并 [QKV/Z 和 B/A projection](https://github.com/vllm-project/vllm/blob/3fe917abb13160d60bea5330399f5c09ca282348/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L514)。native 已有 GPU token readout，但仍有同步与 hidden 往返；本轮已经开启该路径，单请求 TPOT 仍高 29%–43%。需要测完整 verify/propose/catch-up 周期，再按量化契约合并准备、权重读取和输出，不能承诺仅减少 launch 就会收敛。
3. **batch4 继续优化完整批量周期。** [pooled 几何准入](cuda-pooled-admission-geometry-2026-10-12.md)已经消除此前的部分资源串行等待，当前仍有约 1.72× 完成时间差距。批量 target、draft、接受/回滚和队列等待应分别关联到设备时间及请求，不能把现存差距继续笼统归因于显存不足，也不能把已合并的 serial 快照收益当作默认 pooled 收益。

本轮另尝试不复制权重的双输出 FP8 grouped projection：共享量化输入，分别读原 QKV/Z 权重和 scale。编译通过，但 cuTile 双输出 extent/网格契约触发检查，GPU 测试未完成任何数值或计时记录；`grouped-fp8-projection-v1/outcome.json` 记录拒绝，候选没有合入，构建源码已恢复。未关闭越界检查以取得计时，也未把编译成功当作加速证据。

后续硬件相关候选必须纳入 tuning，以设备能力、shape、精度、工作区和服务资源预算选择；保留合法 fallback，并同时评价完整周期及并发容量。没有加入 5090 型号分支。

## 证据与复算

45 上完整目录：

`/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/vllm-main-3fe917ab-runtime-v1`

`paired-v2/` 保存十个引擎原始报告、服务器日志、持续遥测、进程/环境/cgroup 记录、逐对比较与资格检查。`summary.json` 含其全部内容哈希；失败尝试保留在另一个目录。运行前的 `plan.json` 和采集脚本保持原样；汇总器首次因严格背景检查终止的版本保留为 `attempt-1-strict-analyzer.py`，后续汇总如实输出失败资格，没有更改原始测量或资格门槛。

| 文件 | SHA256 |
|---|---|
| `latest-git-pull.json` | `d0aecd911927d7fb7e9675001275cf0e506c74abb1ce6e40ebbdd703f2bfdff1` |
| `identity.json` | `f08845073687ad15fdf3cf5524a00551ea4439e21f9b53983ac4dd0a71a20976` |
| `dependency-check-complete.json` | `d76037e8a7d8e818924a661d72a7b3dbb2b1bb4cc4f92cfe1a9b86ba30f78ff1` |
| `gdn-import-result.json` | `7309f6c5a411230a6947fd7dfc1d6b8c9e68d2b91c9dfe187fb847c4a0f0a0c5` |
| `wheel-source-proof.json` | `e06f22a609feaaf476311ddeae1a0f3000d0bf44684fd77c92ca6f01c307cee0` |
| `profile.json` | `3e30dd17119af016b9605226b1a8ca666d09aaf4e2293a0155c41fb813755c6b` |
| `paired-v2/plan.json` | `9f1157f6bdb73f04e4e8ae6d5ae3e4f8fff101d89117fc0cc00aea85423c71df` |
| `run-pairs.py` | `2ecee32dbb106b8c31ce93407d7242ef7840d7c52cf3dbf46efd18981d095e02` |
| `analyze-pairs.py` | `c7fbc4b961e749df31f1a989a01fa136995782ff726b5f3d1d3e5ec47f3f5677` |
| `summary.json` | `cc277c32963c36167f036267395f236c1ea6b6bc87390f14cd6bb5ac1821ee6e` |

复算只读取已完成数据，不重跑 GPU：

```sh
/home/r/code/inference.rs/artifacts/vllm-compare/bin/python \
  /home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/vllm-main-3fe917ab-runtime-v1/analyze-pairs.py
```

本报告更新了最新 main 的运行证据；性能目标、共同数值/任务质量、热复用等量对照与五组严格合格基线仍未完成。
