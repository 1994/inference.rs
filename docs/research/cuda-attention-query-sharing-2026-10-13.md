# CUDA prefill 查询共享：数值通过，任务质量未通过（2026-10-13）

本轮调查 query tile 共享 K/V 是否能缩小长提示与 vLLM 的差距。
完整注意力局部成本降低，但冻结的 256 题原质量门槛失败，默认路径继续使用 SIMT。
没有据此运行正式服务性能对照，也没有宣称 TTFT 或 vLLM 差距改善。

## 实现与数值

保留原 FP8 KV append、F32 输出和 softmax；按 query tile 裁剪因果读取上界。
实验开关 `INFER_CUDA_TILED_ATTENTION=1` 默认关闭。启用后由硬件调优在
SIMT、QT16、QT32 中选择；缓存按硬件、驱动、编译/kernel 指纹及完整几何隔离。
选择要求全部独立参考、完整 append 与输出通过，以及 cold/warm AB/BA 确认。
共享搜索预算 30 秒；默认关闭时不执行搜索、不读取实验选择。

旧三项 BF16 product 在真实 checkpoint 中失败：输出 1.1897595，F64 参考
1.1898794388974943，归一化误差 1.00805e-4，超过原 1e-4 门槛。
没有放宽门槛。实验 kernel 改为三个 BF16 分量的六项补偿乘积，保留 F32 累加。

六项版本通过真实 Qwen3.8-27B-NVFP4 的完整 F64 注意力检查：width=64/256，
SIMT/QT16/QT32，511-token 长提示后 reset 为 51-token 短提示，全部 16 个注意力层。
576 条记录比较 396361728 个完整输出与 1207959552 个 KV codes；
最大归一化误差 2.49312e-5。15 组独立合成 fixture 的完整输出与 cache 也通过；
同一 tiled 算法采用逐 query 上界或整个 chunk 上界时逐位相同。
模型采集只在测试中增加预分配 memcpy，不修改原生产 kernel。

## 局部成本与原任务质量

六项版本的硬件搜索在 11.45 秒内完成，两种几何均选择 QT16。
计时包含原 append 和完整 compute，先通过数值检查，再连续三次 GPU 利用率 ≤2%
准入；没有终止其他 GPU 使用者。

| width | cold 加速比 | warm 加速比 |
| --- | ---: | ---: |
| 64 | 1.16944 | 1.31485 |
| 256 | 1.62636 | 1.81948 |

这些是局部注意力图成本，不能换算成整模型收益。
原冻结 GSM8K 256 题、原评分器、MTP=0、并发 4、最多 512 output token 的配对结果：
基线提交 4fafeb7 为 **243/256**，实验候选为 **240/256**。
退步 5 题，提升 2 题；原 95% 单侧退步率上界 4.06256%，超过声明的 2% 门槛。
zero-shot 为 117→115/128，eight-shot 为 126→125/128；两版均有 8 条达到长度限制。
229 条完整 token 流不同。完整 F64 检查通过不足以证明任务质量非劣。

退步 ID：837:zero_shot、87:zero_shot、154:zero_shot、1089:zero_shot、1046:eight_shot。
虽然此前已观察到同一服务二进制的生成波动，本轮保留原失败结论，
没有把所有跨版本退步归于波动，没有修改评分器或筛选题目重跑。
实验开关保留供调查使用；默认 SIMT，后续优化转向保持原数值路径的量化复用。

## 可复核记录

开发、编译和 GPU 工作均在 `r@192.168.1.45`，使用仓库 safe-run：
32 GiB host memory、swap=0、独占锁、30 分钟运行上限。
证据根目录：`/home/r/benchmarks/inference.rs/attention-serving-2026-10-13/`。

- `checkpoint-f64-v2.log`：旧三项真实模型失败，保留原输出。
- `build-cuda-v3/`、`checkpoint-f64-v3.log`、`checkpoint-summary-v3.json`：
  六项源代码、CUDA release 构建身份与完整模型参考。
- `causal-full-v3.log`、`hardware-cost-admission-v3.json`、`hardware-search-v3.log`、
  `attention-cache-v3.json`：独立参考、计时准入、原始配对样本与完整选择证据。
- `cli-build-v3.json`、`combined-cache-v3.json`、`quality-v3/paired-summary.json`：
  两版构建身份、冻结缓存、全部请求/token/逐题原评分及未通过结论。
- `quality-v3-attempt-0/`、`quality-v3-runner.log`：首次 Python 缺少 transformers 的
  启动失败；改用已安装模型依赖的解释器后完整重跑，未启动题目时的失败不混入结果。

六项参考测试二进制 SHA256：
`74777364c311317ba0389eeb6c9ee0e5e79da4389b23ec8e714e0040cebc9450`。
质量候选 CLI SHA256：
`243f8d125111069639c67af417402ce22bd6392785dff4a99b5f0e4e444e8163`。
基线 CLI SHA256：
`a944bc62f20f3546aaad0b746188e4af4dde99d5c9924271c1fb2e5eb2edeb37`。
这些实验二进制早于恢复默认关闭的最终提交，默认关闭修订只控制调优入口。
