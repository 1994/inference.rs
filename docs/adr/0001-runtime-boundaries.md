# ADR-0001：核心与执行边界

- 状态：采用
- 日期：2026-10-04
- 设计项：ARCH-001 / CORE-001 / SPI-001 / PROGRESS-001

## 背景

引擎需要同时支持多种 backend、workload 与调度策略，并允许在设备、状态所有权和协议上独立演进。如果把设备调用、字符串解析或策略分支放进热路径，后续每增加一种能力都会放大核心复杂度。

## 决策

- Foundation 定义 ID、生命周期、错误、固定事件与 IR；SPI 定义扩展合约。Compiler、State、Scheduler、Workloads 提供机制，Runtime 组合调用；Backend 通过 SPI 接收编译好的 program 与 step，Frontdoor 管理协议与连接。
- 加载/编译期解析扩展与字符串，热路径只执行静态 program。
- 设备能力按 backend payload 分组。NVIDIA 是首要目标，Metal 用于本地 GPU 验证，CPU 仅是显式测试依赖。
- 请求与状态单写。取消是意图，资源释放等待匹配的 fence/reader。
- poll 错误遵循 ticket 终结合约，timeout 不伪造 device completion。
- snapshot 与异步 ticket 分离；restore 校验 schema、weights/program/provider 与游标，只恢复可完整重建的 quiescent 状态。
- 观测热路径写固定 POD/L0，格式化与查询进入冷 owner；决策保留 Action/Reason/Evidence 与 Request→Step→Op→Kernel→Source 关联。

## 后果

依赖边界允许更换 backend、workload 与策略，静态执行减少热路径解析与动态查找。代价是新增状态或扩展必须提供资源与生命周期合约，并增加编译、验证与恢复的复杂度。

实现上先按职责建立可执行模块，具体依赖出现后再细拆 crate，避免为空能力创建占位工程。Python/WASM、协议与分布式编排留在外围。生产 Supported 的六项验收规则见[技术方案](../design/technical-plan.md)，当前完成度见[实现状态](../design/status.md)。
