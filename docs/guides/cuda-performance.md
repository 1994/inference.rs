# CUDA 性能基线与调优

cuTile Rust 算子已有可复现的 BF16 投影基线；完整 CUDA 执行器尚未接入，因此当前数字只描述算子和同步提交流程，不代表模型吞吐。

## 复现基线

```sh
mkdir -p artifacts
bash tools/bench/cuda-baseline.sh 17408 5120 artifacts/cuda-baseline
```

脚本默认使用 release profile（可用 `CUDA_BASELINE_PROFILE=dev` 改为 debug），输出目录必须尚不存在。它会保存二进制与源码 SHA-256、工具链、构建日志、GPU 前后快照和原始样本。本机需要按 [CUDA 后端说明](../../crates/backend/cuda/README.md)设置 `CUDA_TOOLKIT_PATH` 与 `BINDGEN_EXTRA_CLANG_ARGS`。

`cuda-baseline ROWS COLUMNS` 使用 Rust / cuTile 执行 BF16 权重、F32 输入与累加的 decode GEMV。权重与输入为可复现的合成数据；每个候选先逐元素对照独立 CPU 点积，再进入测量。矩阵尺寸可以直接取自模型投影清单。

测量使用 CUDA events，A/B 逐次交替；每组至少 30 次、最多 100 次，预热预算 100 ms、测量预算 500 ms，每次测量前清理 L2，第一次 JIT 不计入。JSON 保留双方原始样本、中位数、矩阵尺寸、dtype 与 tile 参数。

## 当前结果

`17408 × 5120` BF16 投影的 release 独立复测：

| 配置 | 中位数 |
|---|---:|
| 4 × 128 | 0.187120 ms |
| 16 × 256 | 0.130672 ms（1.432×） |

详见 [release 复测原始样本](../../benchmarks/baselines/rtx5090-release-confirmation.json)。该结果不能外推到其他形状、量化类型或完整模型。更早的 debug 记录与宽归约搜索见 [baseline 说明](../../benchmarks/baselines/README.md)。

注意：当前数字包含输出分配、清零、同步提交以及可能的主机提交间隙，不是孤立的 kernel 时间；合成 BF16 矩阵不能代表 FP8/NVFP4 模型吞吐。debug 与 release、冷缓存与热缓存的结果必须分别保存，不可混合比较。

## 指标分层

| 层级 | 指标 | 状态 |
|---|---|---|
| 同步投影调用 | 中位数、原始样本、A/B 比率、数值误差 | 已实现 BF16 基线 |
| 单算子 | 预分配缓冲、GPU event 耗时、带宽、寄存器与 spill | 待实现 |
| 执行图 | Graph replay、主机提交、prefill / decode / MTP 分项 | 待实现 |
| 完整模型 | TTFT、TPOT、吞吐、峰值显存、MTP 接受率 | 待完整执行器 |

## 代码边界

模型适配通过 `ProjectionCatalog` 从实际模型绑定提取逻辑尺寸、存储类型、量化编码与 MTP 标记，不依赖 CUDA；`LinearStrategy` 提供合法 tile 候选；设备层负责形状校验、内存与 launch；kernel 只实现数学运算。

当前 Qwen 包含 168 个 NVFP4、233 个 FP8 与 104 个浮点投影矩阵，其中 8 个浮点矩阵属于 MTP。相同尺寸但 dtype、阶段或量化语义不同的投影需要独立调优。`--catalog` 只解析元数据，不代表完整模型可执行。

## 调优验收规则

1. 保留默认配置作为对照；候选必须先校验数值、非整除维度和状态语义。
2. 使用模型实际的 shape 与 dtype，分别覆盖 prefill、decode、MTP draft / verify。
3. 记录 GPU、驱动、Toolkit、Rust / cuTile、代码版本、构建模式与模型指纹。
4. 搜索结束后另做独立 A/B 复测，不能把搜索中的最快一次直接设为默认。
5. 先用 Nsight 定位带宽、occupancy、spill 或提交瓶颈，再选择融合、Tensor Core、持久化执行或 CUDA Graph，并记录每次修改的正确性与性能证据。
6. 完整模型需要固定输入与 seed，比较 MTP 开关前后的输出和吞吐，并覆盖拒绝 draft 后的 KV / recurrent state 回滚。算子加速比不能替代这些验收。

共享桌面 GPU 上的波动结果不作为通用阈值，未经独立复测的配置不声明为最优。调优缓存键至少包含 GPU 架构、kernel 版本、dtype、shape、布局和执行阶段；键不一致时回退基线配置。
