# 通用 Attention 与模型边界验证

2026-10-07：模型执行配方迁移至 `model/recipes`，由 `ModelProvider::graph` / `draft_graph` 选择；编译器与 runtime 消费加载时绑定的执行图。目录和依赖规则见[代码布局](../architecture/layout.md)。

## 正确性与结构

- Linux 完整 `check-rust` 通过：严格 Clippy、单元/集成测试、Rustdoc、release 构建、CPU 分配门禁、tiny/grouped golden 与 CLI 请求执行。
- 自定义 provider 去掉最后一层 norm 的回归通过：包加载、HostBackend、Engine 编译均保留自定义图，未重建默认 decoder。
- 依赖门禁覆盖条件依赖，禁止模型配方进入 compiler/runtime、禁止后端进入 scheduler/state；3 个门禁回归通过。
- Attention 独立 F64 oracle 的 31 个 GPU 用例通过，五轮 native / Candle FP16 / Candle BF16 均无精度失败；另有不同 QK/V 维度 GPU 用例，以及视觉跨块和精度残差回归通过。
- 性能判定器 16 个测试通过，包含错误数值、缺失样本、局部退化、P95、抖动，以及全流程快但核心慢的拦截。
- Qwen3-VL-2B 真实模型回归通过：95 个 prompt tokens 一致，16/16 生成 tokens 匹配，视觉相对误差 3.232e-4；双方关闭 DeepStack。覆盖本仓库预处理、视觉塔、merger、MRoPE prefill 与 decode，见[本轮原始报告](../../artifacts/vision/parity-layout-refactor.json)。
- Metal 的接口迁移已完成；本轮为 Linux 验证，不代表 macOS Metal 编译和实机执行验收。

## Candle 对照结果：未放行

本次测试设备 RTX 5090，驱动 615.71.09，CUDA 13.4.92。硬件仅作为测量环境，API 和策略不按该型号分派。Candle 0.11.0 位于隔离 benchmark crate，未进入生产依赖。

每组 31 个用例、5 轮交替顺序、每轮 100 次预热和 60 个 CUDA event 样本，每样本录制 8 次设备操作并取单次耗时。记录设备、工具链、fixture 和二进制摘要，上传及 JIT 不计入时间。输入和输出契约为 F32；原生使用补偿 F32，Candle 内核分别使用 FP16/BF16，因此报告保留独立的数值误差和精度门槛，不宣称计算精度相同。

| 对照基线 | 全流程几何平均加速比 | 核心几何平均加速比 | 判定 |
|---|---:|---:|---|
| Candle FP16 | 1.248× | 0.691× | review，退出码 1 |
| Candle BF16 | 1.249× | 0.692× | review，退出码 1 |

加速比为 Candle 耗时 / 原生耗时，大于 1 表示原生更快。核心范围排除输入转换和输出整理，但仍包含 Candle 公共封装及其图内分配，不能把它称为裸 CUDA kernel 对照。带 RoPE 的两个流程用例只判定完整流程，其余 29 个用例同时判定核心。核心几何平均、每个用例和 P95 都有门槛，外围开销较低不能抵消核心退化。

部分 FP16 对照的全流程中位数（微秒）：

| 用例 | Candle | 原生 | 加速比 |
|---|---:|---:|---:|
| N=2048, D=64 | 124.980 | 468.422 | 0.267× |
| N=2048, D=72 | 264.064 | 1675.760 | 0.158× |
| N=2048, D=128 | 270.216 | 1672.104 | 0.161× |
| MQA decode, Q=1/K=257 | 35.172 | 8.648 | 4.067× |
| 分段 RoPE 流程, N=512/D=72 | 118.568 | 47.808 | 2.477× |

两种对照均有 9 个完整流程用例、18 个核心用例未满足延迟门槛；本轮没有触发跨轮抖动门槛。长序列性能差距是真实未达标项，不能据全流程平均值将替换判为通过。保留原生 runtime，Candle 仅作基线；本次没有启用 Candle 生产路径，也没有放宽门槛。

原始结果：[FP16 判定](../../artifacts/attention-gate/generic-run-2/candle-f16-decision.json)、[BF16 判定](../../artifacts/attention-gate/generic-run-2/candle-bf16-decision.json)、[原生样本](../../artifacts/attention-gate/generic-run-2/native.json)。这些是本机生成证据，不随仓库发布。复现入口为 `make check-attention`，详细要求见[质量门禁](../guides/quality-gates.md)。
