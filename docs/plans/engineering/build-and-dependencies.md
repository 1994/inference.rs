# 构建入口与依赖管理统一方案

状态：设计方案。基于 2026 年 10 月 9 日的 `7c79d99`，尚未调整 Makefile、Cargo 或 CI。

整体排期与任务状态见 [路线图](../README.md) 的 E2；消费者与测试迁移由 E1 交接。本文维护本项设计与验收。

统一 native/cross 构建的计划来源，明确 build、test、check、package、accept 的职责。主 workspace 集中管理依赖版本、最低 Rust 版本和 lint；测试通过 Cargo 的测试 target 与 dev-dependencies 接入，删除产品中的 `test-backends` 配置。

测试执行器的删除见 [测试推理后端方案](remove-test-backends.md)，文件归属和测试收集见 [测试组织方案](tests.md)。这三项按同一迁移顺序交付，不能只改目录而继续依赖第二份测试 CLI。

## 当前构建与依赖问题

[Makefile](../../../Makefile) 的 `build/test/package` 使用 Zig 打包驱动，`local-build` 则直接按 host 选择 Cargo features，并单独补 GCC include。两条入口共享部分目标，却各自承担平台和环境处理；`make test` 又只执行打包流程中的目标 CLI 检查，不代表工作区测试。

[gate.sh](../../../tools/check/gate.sh) 的 rust gate 同时执行 Python、lint、unit/integration、文档、release build、CPU benchmark 和测试 CLI 的 golden 推理。CLI/IR 无 feature 检查与启用 `test-backends` 的工作区检查交错，构建配置和覆盖范围需要靠阅读脚本才能确认。

根 manifest 已集中部分依赖，但 `tokenizers`、Minijinja、Metal、cuTile、tower 等版本仍在成员 manifest 声明。[CPU benchmark](../../../tools/bench/cpu/Cargo.toml) 与 [Attention 对照](../../../tools/bench/attention/Cargo.toml) 各有独立 workspace、lockfile 和完整 lint 副本。CPU benchmark 还通过 fixture 构造依赖 ReferenceModel/ReferenceKernels。

当前正常 CLI 依赖树没有两个 CPU 测试执行器；工作区 feature tree 则由这两个执行器和 dev 依赖启用 IR 的 `test-backends`。整理需要同时保持生产隔离并删除测试配置分叉，不能把现有默认发布配置描述为已经包含 CPU 后端。

## 一个构建计划来源

保留 [build.rs](../../../build.rs) 已有的 target 驱动合同，收敛 native、Zig、package 和 CI 的计划来源。公共 plan 包含 host、目标 triple、ABI、backend、生产 features、profile、工具链与 SDK 需求；执行驱动据此选择 native Cargo 或 cross Zig。

backend 由目标平台决定，host 用于判断工具运行和 SDK 准备。macOS native CLI 使用 Metal；Linux production CLI 显式启用 CUDA。Cargo features 仍是编译期选择，由驱动传入，不能让 build script 在运行时自动开启 feature 或递归调用 Cargo。

CUDA headers、libclang、GCC include、SDKROOT 和交叉 sysroot 的发现与校验复用一个 preflight 路径。报告实际采用的位置和来源，尊重明确配置，避免 native 与 package 的错误提示和补丁各自演化。

`build.rs` 只处理目标合同与编译期元数据；编译、测试、归档和验收由薄驱动按阶段执行。不再新增同时承担依赖安装、递归编译和发布的第二个总脚本。

## 命令职责

以下为拟议命令，不表示当前已实现：

| 入口 | 拟议职责 | 环境与结果 |
|---|---|---|
| `make build` | 构建本机正式 release CLI | 解析本机 target，检查对应 SDK/Toolkit；不隐式执行全部测试 |
| `make build TARGET=...` | 构建指定目标正式 CLI | native/cross 使用同一 plan，明确 backend 和 ABI |
| `make test` | 执行 host Rust/Python 单元与集成测试 | Rust 使用 release；不要求 GPU、Toolkit 或模型下载 |
| `make check` | 格式、lint、布局/依赖规则与 host 测试 | 与普通 CI 的源码检查范围相同 |
| `make test-cuda`、`make test-metal` | 执行对应真实设备 suite | release；所需硬件缺失时失败，报告实际收集与执行数量 |
| `make bench` | 运行显式选择的性能场景 | release、受控资源、指定基线，不混入普通 unit test |
| `make package TARGET=...` | 编译、归档、完整性和目标适用的 smoke | 跨目标只报告编译与文件校验，不冒充运行验收 |
| `make accept PACKAGE_FILE=... MODEL=... GOLDEN=...` | 在目标机器运行正式模型验收 | 验证 release 归档中的同一二进制与输入，输出设备验收结果 |

构建身份与正式测试的 release 要求由 [性能基线方案](../performance/baseline.md#release-与当前配置是必检条件) 统一维护；Rust 测试入口传入 `--locked --release`。Python 不使用 Cargo profile，调用的 Rust/GPU 程序仍须满足该要求。性能入口还须验证共同实验清单、当前 native/vLLM 的生效配置与固定 baseline ID，不能把命令执行成功当作比较条件已对齐。

`local-build`、旧 `check-cuda/check-metal/check-cpu` 可在迁移期作为薄别名，最终移除不再需要的同义入口。安全检查、MSRV 和大型数值对照仍有独立 CI 阶段，各阶段调用上述共同驱动或薄 gate。

普通测试与 Toolkit build check 分开。Linux CUDA package 的编译 job 安装 headers/工具链，但不需要 GPU；host test 不开启 `cuda`。普通测试通过、CUDA 编译通过、GPU 执行通过和性能通过分别记录。

## 主 workspace 的依赖规则

| 范围 | 拟议规则 |
|---|---|
| 版本 | 所有主 workspace 成员的直接依赖版本在根 `workspace.dependencies` 声明 |
| 成员 manifest | 指明使用的依赖类别、target 条件、optional 和实际 features，继承版本 |
| Normal | 仅产品运行所需；不存在对测试执行器、参考框架或 fixture 生成器的依赖 |
| Dev | 测试、examples、bench 所需依赖，不为测试修改产品 IR 或公开 API |
| Build | 仅构建工具和必要 FFI 生成，纳入目标依赖审计 |
| 可复现性 | build/test/CI 使用 `--locked`；升级通过显式 lock 更新单独评估 |
| Lint/MSRV | 主 workspace 成员统一继承；继续检查 Rust 1.90，不能只在当前 toolchain 通过 |

统一版本不等于给所有依赖加精确 `=`。保留必要的兼容范围和 CUDA 工具链精确约束，以 lockfile 固定具体解析结果；升级检查实际重复版本与 advisory，不机械消除上游需要的多版本。

根依赖只保留必要的共同 features。Tokio 等可选能力按成员实际用途声明，避免根表给每个消费者统一启用网络、线程池和 signal。`optional` 放在成员声明中；workspace 的 features 是加法合并，不能假设局部声明能撤销根配置。[Cargo workspace 依赖规则](https://doc.rust-lang.org/cargo/reference/workspaces.html#the-dependencies-table)。

最低版本为 Rust 1.90，因此 `default-features` 的关闭须在根依赖配置中明确表达并验证，不能依赖新工具链才支持的局部覆盖行为。不要为了整理格式一次性关闭所有默认 features；逐项核对用途并验收。

feature 仅表达真实产品或设备能力。测试样例、测试 backend、测试-only 序列化分支不进入产品 feature 列表。`cfg(test)` 只作用于正在作为测试编译的 crate，外部 integration test 不能依赖其被测库中的 test-only 导出。

resolver 2 会在测试/examples 所需场景启用 dev-dependencies 的 features，不能仅凭依赖写在 dev 区就认为测试构建与生产构建完全相同。生产审计以具体 package、target、features 和 normal/build edges 为准。[Cargo feature resolver 2](https://doc.rust-lang.org/cargo/reference/features.html#feature-resolver-version-2)。

## 开发工具与独立参考环境

CPU protocol benchmark 移入主 workspace 的开发工具成员，设为 `publish = false`，不进入默认产品构建集合，复用根 lock、版本和 lint。它继续使用现有局部协议 double，删除 ReferenceModel/ReferenceKernels 引入的依赖；布局门禁明确识别该工具成员。

Candle Attention 对照保留独立 workspace 和锁定依赖，重型参考框架不进入生产 lock 和解析图。它有明确的对照职责、运行环境和安全检查；独立 manifest 的 shared lint 由检查验证或生成同步，禁止手工演化成另一套标准。

官方 Python fixture 生成环境继续隔离。host 测试消费已登记的 fixture，不在运行测试时安装参考框架或下载模型。参考环境的版本、输入和生成命令随 golden 记录，便于重新生成。

最终保留主 Rust workspace 与必要的重型参考 workspace 两种有明确目的的边界。后续新增独立 workspace 必须说明依赖或工具链隔离原因，不能为了几行 helper、一个测试或复制 lint 而创建。

## CI 与依赖门禁

CI 分为 host check、MSRV/security、production target build/package 和真实设备验收。普通 CI 不构建第二份测试 CLI。package 不重复执行全部 host suite，而是验证构建合同、目标编译、归档与适用的二进制 smoke；用二进制/输入身份关联已有源码检查结果。

发布依赖检查覆盖 Linux CUDA、macOS Metal 的实际 feature 图，包含 normal 和 build edges。记录重复依赖、解析 features、lockfile 与目标身份；只检查当前 host 的无 feature CLI 依赖树不足以覆盖全部发布配置。

依赖更新统一检查 normal/dev/build 及 target-specific 声明，防止可选依赖绕过层级约束。以 Cargo metadata/feature tree 的结构化结果为依据，不只搜索 crate 名字符串。现有模型、编译器、runtime 和 backend 的依赖方向继续执行。

每个测试入口报告收集、执行、ignored 和失败数量。job 只完成 cross 编译时明确写 compiled；没有 GPU 的包保持未通过 GPU inference acceptance 的状态，不能由 host suite 代签。

## 迁移顺序与验收

首先清点现有 build/test/gate 调用、直接依赖和测试执行器消费者，建立场景映射。随后统一测试布局并替换协议消费者，移动数值对照，再删除两个 CPU backend、产品测试 feature 和 CLI 分支。

接着合并依赖版本声明、收敛 CPU benchmark workspace，统一 native/cross preflight 与命令职责，最后替换 CI 的重复命令。每阶段保持 Cargo.lock 与已有编译配置可解释，不在目录迁移中顺便升级第三方依赖。

依赖盘点、preflight 和构建计划设计可与 E1 分别推进；受影响 manifest/feature 的集成等待对应消费者迁移。公共 SPI 与 MTP 的接入需要其实际使用的构建合同稳定，不等待不相关 target 的打包与全部 CI 整理结束。共享 workspace/lockfile 的修改统一收口，详细并行边界见路线图。

验收检查同一 target/profile/features 的 native 与 package 构建计划一致，macOS/Linux 的 host tests 都不需要 GPU，正式 CLI 依赖和序列化不因测试改变，Rust 1.90 与安全规则继续通过。对照迁移前的编译时间、链接目标数、测试收集和场景覆盖，分别报告变化，不能仅用目录和脚本数量减少证明完成。
