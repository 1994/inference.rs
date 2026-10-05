# ADR-0001：核心与执行边界

日期：2026-10-04。状态：采用。设计项：ARCH-001 / CORE-001 / SPI-001 / PROGRESS-001。

## 决策

Foundation 定义 ID、生命周期、错误、固定事件与 IR；SPI 定义扩展合约。Compiler、State、Scheduler、Workloads 提供机制，Runtime 组合调用；Backend 通过 SPI 接收编译程序和 step，Frontdoor 管理协议与连接。

加载/编译期解析扩展与字符串，热路径执行静态 program。设备能力按 backend payload 分组；NVIDIA 为首要目标、Metal 用于本地 GPU 验证，CPU 仅是显式测试依赖。

请求/状态单写，取消是意图，释放等匹配 fence/reader。poll 错误遵循 ticket 终结合约，timeout 不伪造 device completion。snapshot 与异步 ticket 分离，restore 校验 schema、weights/program/provider 和游标，只恢复可完整重建的 quiescent 状态。

观测热路径写固定 POD/L0，格式化和查询进入冷 owner；决策保留 Action/Reason/Evidence 与 Request→Step→Op→Kernel→Source。

## 理由与取舍

依赖边界允许更换 backend、workload 和 policy；静态执行减少热路径解析与动态查找。新增状态/扩展必须提供资源和生命周期合约，增加编译、验证与恢复复杂度。

先按职责建立可执行模块，具体依赖出现后再细拆 crate，避免为空能力创建占位工程。Python/WASM、协议与分布式编排留在外围。生产 Supported 的六项验收规则见[技术方案](../design/technical-plan.md)，当前完成度见[状态表](../design/status.md)。
