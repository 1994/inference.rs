# 测试推理后端删除方案

状态：设计方案。基于 2026 年 10 月 9 日的 `7c79d99`，尚未修改实现。

整体排期与任务状态见 [路线图](../README.md) 的 E1。本文维护本项设计与验收。

删除 `test-backends` feature、`test-cpu` CLI 选项、IR 中的 `TestCpu` 类型，以及 `infer-backend-host`、`infer-backend-reference` 两个完整测试推理执行器。正式产品只保留 CUDA 与 Metal 后端。运行时协议测试使用局部脚本化桩，数值正确性使用独立公式、golden 和真实设备验收，分别承担各自职责。

测试布局见 [测试组织方案](tests.md)，构建和依赖迁移见 [构建与依赖方案](build-and-dependencies.md)。

## 已保住的数值对照

删除执行器前，其中唯一的数值参考（gated-delta 状态步）已固化为
`examples/recurrent-delta/golden.json`，导出器与几何、公式、容差、以及"重放接受前缀可复现状态"的
性质一并记录，细节见 [状态重放方案](../speculation/state-replay.md)。因此删除执行器不会丢掉对照物。

## 当前问题

`test-backends` 已经成为产品的编译配置：[CLI manifest](../../../crates/service/cli/Cargo.toml) 启用两个 CPU 执行器和 IR feature，[IR hardware](../../../crates/foundation/ir/src/hardware/mod.rs) 据此改变 backend 枚举、能力和序列化类型。CLI 分派、命令、默认值和诊断因此持续携带测试分支。

[Runtime](../../../crates/engine/runtime/Cargo.toml)、[Frontdoor](../../../crates/service/frontdoor/Cargo.toml)、[Agent](../../../crates/service/agent/Cargo.toml) 和 [CUDA](../../../crates/backend/cuda/Cargo.toml) 的测试依赖这两个执行器；工作区测试会启用 IR 的测试 feature。[门禁](../../../tools/check/gate.sh) 再用额外的 CLI/IR 无 feature 检查隔离生产配置，同时构建第二份 CLI 来执行 CPU fixture。

这使状态机测试、数值对照和可运行产品共用了一套测试推理引擎。一次完整模型计算经常只是 HTTP、取消或资源协议测试的初始化依赖，却扩大了这些测试的依赖图和维护范围。

CPU 调度线程、tokenizer、采样、资源 owner 和交付管线属于正式运行时，继续保留。删除范围是作为模型推理 backend 运行的测试执行器。

## 删除范围

| 位置 | 拟议变化 |
|---|---|
| Workspace | 移除 `crates/testing/cpu/host`、`crates/testing/cpu/reference` 成员、依赖和源码 |
| IR | 删除 `test-backends` feature、`TestCpu` backend/requirements、`reference()` 和 `testing` 模块 |
| CLI | 删除测试 backend 选项、构造、分派、默认内存配置及测试诊断字段 |
| Crate manifests | 删除对两个执行器和 IR 测试 feature 的 normal/dev 依赖 |
| Build contract | 删除 `CARGO_FEATURE_TEST_BACKENDS` 及相应特殊打包防线 |
| 门禁与脚本 | 删除测试版 CLI 的构建和调用，迁移 host/golden 验证职责 |
| CPU benchmark | 去除 ReferenceModel/ReferenceKernels 依赖，使用显式的协议场景数据 |

不能将两个 crate 改名后放到 `test-support`，也不能把完整 executor 复制进公共测试 helper。测试所需的少量数值公式和静态样例分别迁移，其余执行器逻辑删除。

旧配置中的 `test_cpu` 和旧命令中的 `test-cpu` 明确返回不支持。迁移工具可以给出说明，但不保留隐式 CPU 回退。生产序列化 schema 和 capabilities 在测试构建中保持相同。

## 协议测试替代路径

Runtime、Frontdoor、Agent 的多数场景需要的是确定的完成结果、可控制的 fence、背压或错误，而不需要运行 transformer。按场景建立脚本化桩，声明本轮输入预期、输出、完成顺序和注入故障；断言必须针对正式的 runtime/state/actor 行为。

| 测试目标 | 最小替代输入 | 继续验证的正式行为 |
|---|---|---|
| 准入与背压 | reserve/submit 暂时失败后成功 | 重试、公平性、收费与资源归属 |
| 在途取消与超时 | 保持未完成的 ticket，随后显式发出 fence | 不提前回收、终止交付与真实排空 |
| 完成身份 | 重复、遗漏、错请求或 stale epoch 的结果 | 单次提交、隔离和错误范围 |
| HTTP/SSE | 固定 token/logits 和可控完成时机 | 参数映射、输出顺序、背压、慢 reader 与连接中断 |
| Agent | 固定 runtime 状态和 query/command 回复 | envelope、错误码、查询隔离与控制语义 |
| CPU allocation gate | 有界、持久的票据与固定 readout | 正式 Engine 的 tick、buffer 回收和分配数量 |

公共 helper 只复用样例、脚本、票据和断言；实际 SPI adapter 放在需要它的测试 suite 或开发 benchmark 中，不注册到模型/后端 registry。它不加载权重、不执行算子图、不维护真实 KV/recurrent tensor，也不成为 CLI 可选择的 backend。

需要 capability 的规划测试使用明确的 CUDA/Metal 描述样例，只验证元数据与选择规则，不宣称设备存在。需要模型图的协议测试使用最小显式 dataflow 和 descriptor，不再通过 ReferenceModel 构造完整网络。

## 数值测试替代路径

保留 [官方模型 golden 导出](../../../tools/fixtures/export-qwen-golden.py)、[文本 golden 导出](../../../tools/fixtures/export-text-golden.py) 和现有 fixture。CPU 上验证配置、权重绑定、tokenizer、采样和纯函数；CUDA/Metal 上验证实际算子、模型输出和物理状态。

| 原用途 | 迁移后归属 |
|---|---|
| 单算子 host 对照 | 所属测试文件中的小型独立公式，必要时生成固定参考 tensor |
| Transformer 全模型计算 | 官方参考生成的 golden，与真实 CUDA/Metal 模型执行比较 |
| recurrent 与回滚对照 | 真实设备上的 state/下一步输出检查，配合独立公式或已有 golden |
| 权重、配置与图绑定 | model/package/compiler 的无设备测试，直接检查声明、槽与形状 |
| 工作负载和采样 | `infer-workloads` 的纯算法测试及固定分布参考 |

独立公式不能调用被测 kernel 生成 expected。golden 记录生成器、依赖版本、权重身份、精度和容差；生产与参考权重处理不同时，差异必须随报告说明。

缺少 GPU 的 host CI 只报告协议和无设备测试通过。真实设备验收另行运行，保留未覆盖状态；不得以脚本化桩重新证明模型正确，也不得删除数值断言来让旧测试通过。

## 迁移顺序与完成条件

先逐个清点执行器消费者，给每个场景标记协议、数值或真实服务用途。先替换消费者，再删除 crate；同一轮依赖更新同步移除 feature、CLI 分支、IR 类型和构建门禁，不能长期保留两套测试 CLI。

先迁移 kernel registry、runtime 和 agent 的协议场景，再迁移 frontdoor/CLI。随后处理 CUDA examples 的 host 数值对照、CPU benchmark、Metal 验收中对测试版 CLI 的调用。正式 CLI 的帮助、doctor、错误路径仍可无 GPU 测试；成功推理和网络服务在真实设备 suite 中验证。

完成条件包括：

- 实现和 manifest 不再包含两个测试执行器、`test-backends` 或 `TestCpu`；旧值仅可出现在明确拒绝它们的回归测试和历史说明中。
- 发布 CLI 只有一种产品行为，不存在可运行 CPU fixture 的第二份测试构建。
- 原消费者的关键断言都有明确去向，尤其是超时后的 fence、物理 state 回收和数值对照。
- 无设备测试不依赖 CUDA Toolkit、Metal 设备或模型下载；真实设备 suite 缺少所需硬件时明确失败。
- CPU benchmark 继续测量正式 CPU 协议与分配，并明确排除模型执行，不依赖推理参考后端。

实现前的消费者入口包括 [runtime capacity 测试](../../../crates/engine/runtime/tests/capacity.rs)、[actor 测试](../../../crates/service/frontdoor/tests/unit/actor.rs)、[CLI smoke](../../../crates/service/cli/tests/smoke.rs)、[CPU benchmark](../../../tools/bench/cpu/src/engine/mod.rs) 和 [Metal 验收](../../../tools/check/metal.py)。这些位置是迁移清单，不是拟议保留的新目录。
