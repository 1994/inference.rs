# 代码布局与调用链

目录按职责分组，`lib.rs` 负责模块声明和显式 API 导出，具体实现进入职责模块；测试保留在所属 crate 的 `tests/` 或 `src/**/tests.rs`。

## Crate 职责

| 目录 | Crate | 职责 |
| --- | --- | --- |
| `crates/foundation/core` | infer-core | ID、错误、生命周期、arena、credit、事件与有界容器 |
| `crates/foundation/ir` | infer-ir | 请求、模型、执行、设备能力、计划与共享输出 |
| `crates/foundation/spi` | infer-spi | backend、state、调度、模型、workload、资源和扩展接口 |
| `crates/backend/api` | infer-gpu-api | 设备描述、传输与批次槽协议 |
| `crates/backend/kernel-api` | infer-kernel-api | kernel 注册与选择 |
| `crates/backend/metal` | infer-backend-metal | Metal 设备、加载、forward、提交、完成与物理状态 |
| `crates/backend/cuda` | 暂无 crate | RTX 5090 接入约定，原生执行待迁移 |
| `crates/model/package` | infer-models | HF 包、索引、Safetensors、内存预检、tokenizer/template |
| `crates/model/compiler` | infer-compiler | IR 降低与 dataflow 编译 |
| `crates/engine/state` | infer-state | logical state、块池、prefix cache、KV manager 与物理 CPU 对照 |
| `crates/engine/scheduler` | infer-scheduler | 多队列、调度策略、公平性、预算与成本校准 |
| `crates/engine/workloads` | infer-workloads | 采样、embedding、rerank、decision 与 projection |
| `crates/engine/runtime` | infer-runtime | 请求所有权、阶段推进、资源交接与故障处理 |
| `crates/diagnostics/observe` | infer-observe | 指标、trace、事件、源码映射与诊断视图 |
| `crates/diagnostics/quality` | infer-quality | golden、进展检测、校准与性能测量 |
| `crates/service/frontdoor` | infer-frontdoor | 输入准备、单所有者 actor、HTTP/SSE 与交付 |
| `crates/service/agent` | infer-agent | 诊断工具、证据查询、实验与建议 |
| `crates/service/cli` | infer-cli | 参数、命令分发、backend 选择与运行支持 |
| `crates/testing/cpu/*` | infer-backend-host/reference | feature gate 下的正确性对照 |

## 依赖方向

基础类型与 IR 位于依赖底层；SPI 描述接口，模型编译、state、scheduler 和 workload 实现各自的职责。runtime 组合这些模块并通过 SPI 驱动 backend；frontdoor、agent、cli 位于入口层。runtime 不依赖 Metal 或 CUDA 实现，backend 不依赖服务入口。默认生产 CLI 不依赖 CPU 测试执行器。

调度策略及队列归 `scheduler`；runtime 的 `pipeline/scheduling` 负责把请求、就绪视图和资源结果接到策略，不能在这里另建一套队列策略。KV 的逻辑分配与所有权归 `state`；backend 负责设备内存和复制执行；runtime 负责资源票据与请求生命周期的协调。

## 推理调用链

```text
CLI / HTTP / SSE
  → frontdoor::ingress / cpu            有界接入与异步输入准备
  → frontdoor::actor                   单所有者驱动、命令与输出交付
  → runtime::preparation / pipeline::admission
  → runtime::pipeline::scheduling      就绪视图、阶段与资源状态衔接
  → scheduler::queue / policy         队列选择、公平性、预算与计划
  → runtime::pipeline::dispatch        计划、resource、任务与 backend 交接
  → BackendProvider                    submit / poll / 资源操作
  → runtime::pipeline::completion      完成反馈与 CPU 输出阶段
  → runtime::pipeline::lifecycle       终止、取消、超时与回收
  → frontdoor::actor::delivery          HTTP/SSE 输出与背压
```

prefill/decode 适配保留在 `runtime/src/stages`；runner 管理 backend 线程与交接。观测由 `runtime/src/observation` 收集，通过 `infer-observe` 输出；checkpoint、snapshot 与 replay 归 `runtime/src/persistence`。PD 扩展目标见[状态表](../design/status.md)。

## 模块内组织

- `runtime/src/engine`：类型、构造、检查、诊断与不变量；不混入阶段执行。
- `runtime/src/pipeline`：admission、scheduling、dispatch、completion、lifecycle。
- `runtime/src/requests`：请求 record、冷热存储与租户统计。
- `state/src/logical`：reservation、prefix、lifecycle、inspection；`blocks/cache/kv` 保持独立。
- `metal/src/executor`：loading、setup、state、prefix、forward、submission、execution、completion、checkpoint；`provider` 汇集 SPI 适配。
- `spi/src`：按 provider、model、workload、scheduling、state、backend、kernel、protocol、observer 和扩展合约拆分。
- `frontdoor/src/actor`：启动、驱动、命令、准入与交付；`http` 按 native、text、protocol、observability、server、error 拆分。
- `cli/src/commands`：按命令领域组织；`support` 按 engine、JSON、fixtures、execution、server、agent 组织。

公开 API 由 facade 显式导出，依赖在使用处导入，helper 使用最窄可见性。

## 文档与工具

`docs/guides` 放操作说明，`docs/architecture` 放当前实现，`docs/design` 放目标与差距，`docs/validation` 放带边界的证据，`docs/adr` 放决策。模型样例与测试资产归 `examples`；生成日志、报告与负载结果归 `artifacts`。

`tools/check` 是统一门禁与架构规则，`tools/validation` 是服务和负载验证，`tools/fixtures` 是独立参考导出，`tools/bench/cpu` 是隔离的分配测量 crate。`Makefile` 与 CI 调用同一门禁入口。

新增 crate 必须落在职责分组下；新增 backend 必须落在 `backend/<name>`，CPU 对照只能位于 `testing/cpu`。目录门禁校验分组、路径依赖、入口大小、实现层反向依赖和文档链接；Rust 门禁继续校验公开 API、feature 隔离及全部 lint。
