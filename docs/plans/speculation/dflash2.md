# DFlash2 模型接入方案

状态：设计方案。基于 2026 年 10 月 9 日的 `7c79d99`，目标为 `z-lab/Qwen3.8-27B-DFlash2`。

整体排期与任务状态见 [路线图](../README.md) 的 D1、D2。本文维护本项设计与验收。

DFlash2 作为独立草稿 provider 接入，共用 [推测解码 SPI 与状态协议](spi.md)。模型图由模型层定义，算法 provider 的规划由公共 workloads 层实现，CUDA 是优先落地的设备执行；其他 backend 按同一契约独立实现并报告能力。现有 MTP 的融合、位置错位和追赶方式不能直接套用到该模型。先完成兼容性检查和独立数值参考，再实现模型图、候选选择与目标验证；资源准入同时执行。

## 模型需求与接入边界

官方 [config.json](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/main/config.json) 声明 `DFlash2DraftModel`、5 个草稿层、hidden size 5120、32 个 attention heads、8 个 KV heads、head dim 128。原生 block size 为 8，块内非因果；target taps 为 `[5, 19, 33, 47, 61]`，草稿 attention 使用长度 2048 的滑动窗口，selector rank 为 256、top-k 为 16。

权重结构与单卡约束的现有记录见 [性能实验记录](../../research/cuda-performance-experiments.md)。接入时将其中的双抽头动态卷积、特征融合和 candidate selector 分别落实为模型配方与算子能力，不把它们解释为通用 Qwen decoder 的默认行为。

第一版只开放 greedy 和线性候选链。完整随机推测采样依赖实际 proposal 分布的推导与导出，不能把 selector 分数直接当成归一化的 `q`。草稿内部的候选 lattice 与 target 的树形验证是两个独立能力。

## 独立 package 与兼容性检查

草稿由独立 `DraftModelBinding` 绑定，模型 registry 按 `DFlash2DraftModel` 解析 provider。package 层负责配置、张量命名和拓扑；CUDA loader 负责设备存储、绑定与图录制。

| 检查项 | 接受条件 |
|---|---|
| target 身份 | target 架构、层数、hidden 几何及版本与草稿要求匹配 |
| target taps | layer ID 有效；明确抽取层输出的位置、归一化与位置映射 |
| tokenizer | token ID 对应关系兼容；不能只比较 vocab size |
| embedding 与 head | 声明可共享，且权重身份、形状、dtype 与设备所有权匹配 |
| 特殊 token | mask、EOS、padding 与请求采样规则一致 |
| 算子 | backend 支持动态卷积、块内非因果 attention、feature fusion 与 selector |
| 资源 | 权重、target/draft 状态、feature 历史、arena、graph 和回滚开销全部纳入报价 |

兼容性失败在加载或规划时报告具体原因。`auto` 可以依据明确策略回退 MTP 或普通 decode；显式要求 DFlash2 时，不能静默改用其他算法。草稿量化是单独的精度方案，需要记录格式和接受长度变化。

## Target 特征捕获

模型 provider 声明抽头语义，backend 在 target 图中捕获指定位置。prefill 和 verification 必须使用同一特征定义。首先通过逐层独立参考确认 layer 编号、norm 前后位置、token 位置以及 fc 融合结果，再录制常驻图。

特征留在设备端，以有界 feature store 管理历史。保存范围由参考执行语义确定；draft 的 sliding window 不自动等于 raw target hidden 的保留长度。如果 fc 投影能够在特征捕获后等价计算，可存投影后的表示以减少长期缓存，但需先通过数值对照。

verification 产生的拒绝尾部特征只在事务内暂存，提交后立即失效。下一轮使用的特征必须覆盖 target 已物化前缀；校正或 bonus token 的特征要等该 token 被 target 消费后才能取得。

prefill chunk、prefix 命中和 provider 切换都需要检查特征覆盖范围。缓存缺失时选择可验证的重建路径，并将其成本计入 TTFT 或切换成本。

## 草稿模型图与 CUDA 算子

| 模块 | 拟议实现与验证重点 |
|---|---|
| 特征融合 | 绑定多层 target hidden 的拼接与 fc 投影；确认拼接顺序和归一化 |
| 双抽头动态卷积 | 分别实现 attention/MLP 的 kernel projection、分组与两抽头边界；对照独立公式 |
| 草稿 attention | 历史 prefix 与当前 block 使用正确 mask；允许块内双向可见，保持滑动窗口边界 |
| 草稿 KV | 与 target KV 分离；定义初始化、窗口淘汰、接受后的保留与拒绝后的修复 |
| candidate selector | 实现低秩打分、top-k 和路径选择；确认排序、tie break 与路径依赖 |
| target verification | 使用目标模型自身的因果 mask；草稿的非因果语义不传播到 target |

先实现可检查的算子组合，再由 backend 进行等价融合。selector 采用分块打分与 top-k，避免为方便集成而将完整词表矩阵传回 CPU。每个融合方案都与未融合参考对照，不能用最终接受率代替算子正确性。

[Infernix 调研](../../research/infernix-performance.md) 提供了两项资源候选：对草稿的大投影按角色量化，以及另存带 token-ID 映射的低精度 shortlist head。先建立完整 head 和 BF16 草稿参考，再分别评估量化、shortlist 及二者组合。target head 保留完整词表；DFlash2 候选 ID 映射回公共词表后进入 selector，随机模式的条件 `q` 从实际候选与分数计算。codec/kernel、额外 head、映射和 workspace 全部计入准入。

## Block 执行与状态结算

原生 block size 8 对应一个 seed 与最多 7 个候选，因此 target verification width 为 8。对外参数分别表达候选数量、draft block size 和 target verify width，避免与 MTP depth 混用。

执行顺序为准备已提交前缀的 target features、生成 block、运行 selector 得到一条候选链、target 因果验证、提交接受前缀、修复 draft KV 和特征历史。具体 KV 修复遵从 DFlash2 参考实现，由 provider 声明需求，不复用 MTP 的固定追赶代码。

显存受限时，可保留原生 8-token 草稿生成，只验证候选链的较短前缀。这会浪费部分草稿计算，但可以降低 target verification 的状态成本。直接缩短 draft block 会改变输入和 mask，必须独立验证模型语义与接受长度，不能仅修改 config 常量。

Infernix 的已实现路径接受 `K=1..15`，物理 block 为 `K+1`，说明该权重下长度可作为执行参数研究，而非固定张量维度。但块内非因果 attention 会使不同长度的候选同时改变，短块不保证等于长块的前缀。本项目先在原生 8-token block 建立参考，再做可变草稿宽度的独立对照；它与只验证短前缀是两个实验。

batching 按 target program、候选布局、验证宽度、状态 recipe 和 graph signature 分组。短链可以使用有明确有效长度的固定宽度图；padding lane 不得推进真实 KV、recurrent state 或 feature cursor。

## 实施依赖与验收

| 阶段 | 交付内容 | 进入下一阶段的条件 |
|---|---|---|
| 1 | 独立 package 解析、兼容校验和完整显存报价 | 不执行模型也能给出可解释的准入或拒绝结果 |
| 2 | 特征抽头、融合、卷积、attention 与 selector 的独立参考 | 各模块通过数值、mask、窗口与边界检查 |
| 3 | 草稿图与单请求线性链 | 与参考实现的中间结果、候选路径及状态对齐 |
| 4 | 共同 verifier 和事务结算 | 拒绝后下一步与普通 target decode 的状态、输出一致 |
| 5 | 池化、常驻图与设备端读写 | 多请求、prefix、取消、OOM 回退与混合负载通过验收 |
| 6 | 草稿量化和自适应验证长度 | 在完整成本和接受长度变化下取得服务收益 |

资源方案见 [CUDA 性能优化方案](../performance/cuda.md)。较大显存设备可先完成 BF16 正确性验证；目标小显存设备需要同时解决回滚快照和草稿权重开销。降低回滚成本并不等于草稿一定能装下。

package 校验、独立参考、draft 图/selector 模块与容量分析属于 D1，其中配置检查、参考制品和独立模块可以先准备；共同 verifier 与服务接入属于 D2，等待 I1 及对应资源能力。I1 契约稳定后，draft 模型模块的接线可与 R1 的 target Record/Fold 分别推进，最终在 D2 合流。若目标容量装不下 BF16 草稿，在 BF16/完整 head 参考可用后提前评估草稿量化或 shortlist，不必等到表中池化阶段失败后再处理；量化与 R1 各自验收，GPU 运行按资源窗口排队。

正确性矩阵覆盖首 token 拒绝、中间拒绝、全接受、EOS、输出上限、prefill chunk 尾部、2048 窗口边界、prefix 恢复与 provider 切换。greedy 的最终输出对照同一 target 权重和精度下的普通 decode；中间状态须在拒绝后继续运行多步验证。

性能比较执行 [性能基线方案](../performance/baseline.md) 的构建、配置、负载与统计要求，至少包括普通 decode、当前最佳 MTP 配置和 DFlash2。引用已验收的普通 decode/MTP baseline ID，另行登记 DFlash2 候选 profile；首次接入不预设 DFlash2 已有有效基线。报告真实启用的算法、图宽度、池化状态、显存与接受长度，不能把因容量失败而退回 serial 的结果标为 pooled DFlash2。
