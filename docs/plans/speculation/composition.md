# MTP 与 DFlash2 组合方案

状态：设计方案。基于 2026 年 10 月 9 日的 `7c79d99`；同轮组合为实验能力。

整体排期与任务状态见 [路线图](../README.md) 的 D2、X1：请求/轮次选择随 D2 交付，同轮组合由 X1 独立验收。本文维护本项设计与验收。

引擎可以注册和加载 MTP、DFlash2 两个 provider，由共同协议保证 target 前缀与各自状态的一致性。请求或轮次之间选择算法先落地；同一轮让两个算法共同生成候选，需要明确组合算法、模型条件和额外成本，单独验收后启用。

共同协议见 [推测解码 SPI 方案](spi.md)，DFlash2 的模型需求见 [接入方案](dflash2.md)，性能实验条件与判定见 [性能基线方案](../performance/baseline.md)，CUDA 执行优化见 [CUDA 方案](../performance/cuda.md)。

## 上游实现与本项目的扩展范围

2026 年 10 月 9 日查阅的公开 main 分派中，vLLM 根据配置创建一个 speculator，DFlash2 与 MTP 位于不同分支；SGLang 同样按一个算法配置创建对应 worker。这些内置入口没有提供 MTP 加 DFlash2 的组合算法。[vLLM 分派](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/__init__.py)、[SGLang 分派](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_info.py)。

因此，上游可用于参考单算法接入和验证协议；本项目的同轮组合需要作为自己的实验算法实现。多 MTP head、复用 recurrent state 提交 helper，都不能据此解释为 MTP 与 DFlash2 已经叠加。

本地 [Infernix 调研](../../research/infernix-performance.md) 同样确认其 Engine 选择一个神经草稿 backend。它已组合的是 n-gram copy 与神经草稿：按 row 覆盖 proposal，并由相同 target verifier 结算；DFlash2 另可扩为 candidate lattice 树形验证。这两项是可参考的候选协议和状态能力，不是 MTP 与 DFlash2 同轮组合的已有收益证明。

## 同时支持的三种语义

| 模式 | 执行语义 | 第一阶段定位 |
|---|---|---|
| 分请求选择 | 某些请求用 MTP，另一些用 DFlash2 | 独立能力，两者共享 target 权重和验证协议 |
| 分轮次选择 | 同一请求依据接受长度、成本和资源在两者之间切换 | 自适应能力，每轮只有一个活跃草稿 provider |
| 同轮组合 | MTP 与 DFlash2 在同一轮共同影响候选链或候选树 | 实验能力，由显式 CompositionProvider 描述 |

拟议配置使用 `off`、`mtp`、`dflash2`、`auto`、`compose` 表达这些选择。`auto` 表示选择，不承诺两者同时计算；`compose` 必须绑定具体组合策略、候选预算和支持条件。两个开关同时为 true 不能成为默认组合语义。

## 共同状态与组合契约

组合 provider 不接管 target 状态，只持有子 provider 的句柄、依赖计划和候选合并规则。它必须声明输入特征、候选来源、线性或树形布局、采样模式、额外状态与失败后的处理方式。backend 仍通过同一个目标验证协议结算。

两个子 provider 可以共享只读 target 特征，但不共享可变 draft KV。未接受的 MTP token 不能被当作已提交的 target 前缀，MTP hidden 也不能替代 DFlash2 所需的多层 target taps。

最终只提交 target 接受的那条前缀。两个 draft 各自选择追赶或失效重建，拒绝分支的 feature、KV 和 recurrent state 一并丢弃。为了减少常驻成本，非活跃 provider 可以延迟修复；重新使用前必须检查位置与 epoch。

[Recurrent 状态重放方案](state-replay.md) 定义与算法互补的 target 回滚能力。MTP、DFlash2、分轮选择和同轮组合都可复用它；采用 ReplaySSM 不等于已经实现两个神经草稿的联合提议。比较组合与单算法时，应使用相同的 target 回滚策略，另将树形分支支持作为明确能力验收。

## 同轮组合的候选实验

| 实验方向 | 可以研究的算法 | 必须满足的条件 |
|---|---|---|
| MTP 辅助路径选择 | DFlash2 生成候选 lattice，MTP 给候选路径或前缀提供额外评分，再选一条链给 target | 明确评分含义和候选条件；MTP 逐位置计算与分支成本纳入预算 |
| MTP 前缀接 DFlash2 续写 | MTP 给出短前缀，DFlash2 对剩余位置进行条件草稿 | 草稿明确支持固定前缀输入和对应 mask；不存在的 target taps 不能用草稿 hidden 冒充 |
| 候选树合并 | 两者从相同权威前缀提议，合并并去重后交给 target | backend 支持树形 mask、分支位置、KV 和 recurrent state 恢复 |

优先验证第一种，它可以保留线性 target verification，避免一开始增加目标侧的分支状态。先确认是否能以低成本取得有用的 MTP 评分；若评分需要大量串行 draft 步骤，直接结束该实验，不以接受率提升代替性能收益。

第二种需要专门的模型语义验证。DFlash2 原生单 seed 的 block 接口是否接受多个固定前缀 token，应由参考实现和独立实验确认；即使 target 验证可以保证 greedy 输出正确，也不能推断这种条件方式会提高接受长度。

第三种放在树形验证能力之后。仅扩充 `ProposalBatch` 的候选数量并不能支持树形验证；每个节点的祖先可见范围和 recurrent 状态都必须正确表达。

Copy provider 可作为较低成本的先行组合实验：从已提交 prompt/output 的精确匹配前缀提出连续复制链，未匹配 row 保留神经 proposal。验证宽度按全批最长有效链与已捕获图选择，短链 padding 不提交。随机模式的 copy `q` 为 one-hot，不能沿用被覆盖神经候选的概率。该实验服务复制与编辑负载，仍需确认收益，不能替代 MTP/DFlash2 组合的条件语义。

不把“两种草稿直接拼接”设为可用策略。拼接只产生一条待验证链，不能证明后半段是在正确条件下生成，也不能证明多付出的计算能够被接受长度补偿。

## 自适应选择先于自动组合

选择器基于每轮实际新发布 token 数量，以及草稿、验证、回滚、追赶、数据传输和切换的完整成本估计下一轮策略。接受率只作为诊断项；高接受率仍可能对应更慢的服务延迟。

使用按负载类别更新的有界统计与滞回，避免每轮来回切换。候选策略必须已有可用图签名、特征覆盖和资源报价。缺少 feature 或 draft 状态时，把重建成本计入报价，而不是假定切换免费。

同一个批次可按兼容计划分组执行，再共同交付结果。是否能将来自两个 provider 的链装入同一 target verify graph，由候选布局、宽度、状态 recipe 和 backend 能力决定；scheduler 不直接增加算法名分支。

低显存下可以只常驻一个 provider 或仅使用 MTP。目标权重共享不会消除 DFlash2 权重、双份 draft 状态、特征缓存和 graph arena 的开销。

## 采样正确性与性能判定

第一版组合只支持 greedy，通过共同 target sampler 决定接受前缀。所有 penalty、过滤、EOS 和输出额度都沿用相同的请求语义。

随机模式需要推导组合后真实有效的 proposal 分布：重新排序、截断、混合、前缀选择和去重都会改变 `q`。不能沿用某个子 provider 的原始概率，也不能仅因最终又调用了 target sampler 就声明分布无偏。在完成推导和统计验收前，显式关闭随机组合。

性能比较绑定共同契约下的 profile 与 baseline ID，必须包括普通 decode、MTP、DFlash2、分轮次选择和同轮组合。组合的比较基线是同负载下最佳单算法，不能只与关闭推测解码比较。

用 `完整周期时间 / 本轮实际新发布 token 数` 衡量服务收益，并同时记录物理消费数量，避免重复计算上轮已经发布的 seed。EOS、输出上限和全拒绝轮次都计入实际收益。

只有在完整服务负载上有稳定收益，且尾延迟、显存、池化成功率与状态恢复通过验收时，某个组合策略才进入自动选择。否则保留为实验策略，默认继续使用单 provider。

## 落地顺序

先完成共同 SPI 与 MTP 迁移，然后接入独立 DFlash2，再支持按请求和轮次选择。随后实现显式 CompositionProvider，优先研究 MTP 辅助 DFlash2 的路径选择；树形合并在目标侧分支状态能力之后。

每个阶段都提供最终 token 与拒绝后状态的对照、模式切换记录、资源报价和完整周期性能。注册两个 provider 是扩展能力验收；同轮组合取得收益是另一项算法验收，分别报告。
