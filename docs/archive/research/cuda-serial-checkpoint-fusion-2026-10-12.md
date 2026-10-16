# CUDA 串行验证状态快照融合（2026-10-12）

串行验证原先在 Conv/Delta 写回状态后，逐个记录独立的 device-to-device memcpy 作为回滚快照。候选复用 pooled 已使用的 writer-plus-snapshot kernel，使计算结果同时写入状态和快照，保持原计算顺序与 launch 几何。

## 条件与验收

源码基点为 5858d9783df3c1b775ddd0f7dd52267ead22811c，vLLM 官方 main 源码参考为 f26d9f432ced7b1087d02d29bffed2249df1a18e。45 的 RTX 5090 只用于验证，不按设备名、模型名或测得时延设置运行阈值；没有引入新的硬件调参。

Flat capture 返回原 activation allocation，row capture 继续保留共享 backing allocation，graph 与资源的生命周期关系不变。Inactive lane 不更新状态；可提交的前缀范围继续受实际 token 数约束。测试覆盖前一批的 inactive suffix 重新成为 active，以及部分接受后继续 decode。

严格 CUDA all-targets Clippy、47 项常规单元检查与 CUDA CLI release build 通过。13 项 GPU 检查通过，包括：

- 原 Conv/Delta kernel 与 fused checkpoint kernel 的逐位比较。
- 小型与真实头维度 (16 key heads, 48 value heads, D=128)，serial 宽度 2/3/5/9 的完整/尾部验证、部分接受、state/logits/hidden 和继续 decode。
- 既有 pooled checkpoint、pooled verify 和 GPU readout 检查。
- 同一次权重加载的混合 FP8/NVFP4/BF16 27B，511-token prompt、verify 3/2/1、commit 1/2/3，hidden/logits、循环状态及 FP8 KV code 逐位相同。

模型候选保持量化模型的 scalar Delta；这项改动不放行 GDN chunked recurrence 的既有数值漂移。

## 性能测量范围

常见请求已经走 pooled 快照融合；serial 适用于回退，以及上下文容量超出当前 pool 的 8192 tokens 等场景。固定工作量测量只计 target verification 与 GPU greedy readout，rollback 在计时之外；没有测服务 TPOT、MTP candidate generation 或 HTTP 请求，也不是与最新 vLLM main 的运行对比。

48 个 GDN 层、验证宽度 3 的 serial graph 可删除 48 × 3 × (3 Conv + 1 Delta) = 576 个独立 memcpy 节点。实际成本以独立进程 A/B 结果记录；节点减少本身不登记为性能数字。

## 固定工作量结果

同一个诊断二进制加载一次权重，两个程序分别捕获新旧 serial graph；按 AB/BA 交替回放相同的固定 token。5 个独立进程，每个先排除 8 对 warmup，再保留 32 对测量，共 160 对有效测量，没有删除进程或选择更好的重复。

| 项目 | 原 independent memcpy | writer-plus-snapshot |
|---|---:|---:|
| 进程内中位数，再取进程中位数 | 21.654829 ms | 21.119007 ms |

进程配对百分比中位数为 -2.48948%，五个进程分别为 -2.50279%、-2.51496%、-2.48948%、-2.48293%、-2.46562%。按完整进程重新采样的经验 95% 区间为 [-2.51496%, -2.46562%]。固定 token 的 hidden、选择 token 和最后的状态相同，tuning cache 未改变。

这支持启用 serial writer 快照融合，收益是有限的约 0.536 ms/次 target verify。模型只有一个 prompt 与固定 token 序列；这组数据不能替代更大上下文的服务矩阵，也不能登记为 pooled 吞吐、整次 MTP 或与最新 vLLM main 的差距缩小比例。

## 原始证据

45 上目录 `/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/serial-checkpoint-fusion-v1` 保存原 capture 源码、冻结 shipping 编译源码、编译/13 项 GPU 验收结果、生产二进制和独立计时探针。生产二进制 SHA256 为 052e351a8b1a97d5896f21779404e9ea08b64505743ad899f36280e96f44b819，shipping CUDA 检查二进制为 90f1136159581443379bd19cd17e6f722a60493eb0774b71fb5859076ee0c3ab。

tuning cache SHA256 d212f19361deeb8adf7cd321a268080d047920bcc5fa8973a251abeb404bcb7f，在 GPU 验收前后未改变。计时探针与 shipping 检查二进制分开保存；探针只新增测量测试，不进入生产源码。

结束前官方 vLLM main 又前进到 3fe917abb13160d60bea5330399f5c09ca282348，45 已再次快进同步；新增 attention warmup，不改变本轮 GDN 源码对照和 native 固定工作量证据。见 [main 再次同步](vllm-main-execution-paths-2026-10-12.md#再次同步-main)。
