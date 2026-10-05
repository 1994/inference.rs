# 开发与运行

使用固定 Rust 1.99.0，最低版本 1.90、edition 2024。CLI 默认 mimalloc，构建需要 C 编译器；macOS 安装 Xcode Command Line Tools，Linux 使用 GCC/Clang。Rust 运行时不依赖 Python/Torch，Python 仅用于参考资产与验证工具。

## 构建与测试

```sh
cargo build --locked --release -p infer-cli
make check
```

分项及工具版本见[质量门禁](quality-gates.md)。CPU 对照显式构建：

```sh
cargo build --locked -p infer-cli --features test-backends
target/debug/infer --backend test-cpu verify --package examples/qwen-hybrid-tiny \
  --golden examples/qwen-hybrid-tiny/golden.json
```

默认发布版只提供 auto/cuda/metal，auto 不回退到 CPU。设置 `INFER_QWEN_TEXT_PACKAGE` 为官方 tokenizer 包绝对路径可执行额外文本 parity；普通 CI 使用仓库内微型资产。

## 模型包与请求

```sh
target/release/infer inspect-package --package examples/qwen-hybrid-tiny
target/release/infer --backend metal run --package examples/qwen-hybrid-tiny \
  --requests examples/requests.json --config examples/runtime.json
target/release/infer --backend metal verify --package examples/qwen-hybrid-tiny \
  --golden examples/qwen-hybrid-tiny/golden.json --atol 0.000002 --rtol 0.00002
target/release/infer --backend metal serve --package examples/qwen-hybrid-tiny \
  --listen 127.0.0.1:8080
target/release/infer --backend metal agent --package examples/qwen-hybrid-tiny \
  --probe-memory-mib 4
```

正式执行需要真实 package。`--device-memory-mib` 默认 512；`--kv-page-tokens`、`--kv-cache-blocks`、`--prefill-chunk-tokens`、`--upload-staging-mib` 为全局选项。auto 当前在可用 Mac 上选择 Metal，CUDA 未实现时显式选择返回 Unsupported。`doctor` 报告可用性与 catalog。

请求格式见[请求示例](../../examples/requests.json)，调度/配额见[配置示例](../../examples/scheduling.json)。加载、readout 和文本资产见[模型执行](model-execution.md)；Linux 放置配置见[Linux 指南](linux.md)。

## HTTP/SSE

| 路径 | 功能 |
|---|---|
| GET /health | ready、fault、pending release；隔离时 503 |
| GET /native/v1/runtime | runtime 与状态检查 |
| GET /metrics | Prometheus |
| GET /native/v1/observability | 指标、资源、session、trace 完整性 |
| GET /native/v1/events | after/limit/request cursor 查询 |
| GET /native/v1/diagnostics | 故障与准入证据 |
| GET /native/v1/timeline | Chrome Trace |
| GET /native/v1/traces/otlp | OTLP JSON |
| POST /native/v1/requests | CanonicalRequest → 完成结果 |
| POST /native/v1/stream | token/finished/error SSE |
| POST /native/v1/text | prompt/messages → 文本或 workload 结果与 fingerprint |
| DELETE /native/v1/requests/{id} | 取消 |

HTTP body≤2 MiB，队列与 worker 有界；准备池默认按可用逻辑核减去 owner 预算计算，限制为 1..32，而非固定 8。嵌入配置入口见[CpuConfig](../../crates/service/frontdoor/src/cpu.rs)。断连或输出背压取消相关请求，在途资源按 reader/fence 回收。

领域错误为 code/message JSON：schema 422、业务输入 400、不支持 501、重复 ID 409、容量/配额 429；框架处理 JSON/body/Content-Type 错误。未知字段拒绝。accepted RequestId 在同一 Engine 不复用，建议单调递增。

## Checkpoint、回放与测量

```sh
target/release/infer --backend metal run --package examples/qwen-hybrid-tiny \
  --requests examples/requests.json --config examples/runtime.json \
  --journal artifacts/journal.json --snapshot artifacts/checkpoint.json
target/release/infer --backend metal replay --package examples/qwen-hybrid-tiny \
  --snapshot artifacts/checkpoint.json
target/release/infer --backend metal profile --package examples/qwen-hybrid-tiny \
  --journal artifacts/journal.json --config examples/runtime.json \
  --output artifacts/profile.json
target/release/infer --backend metal benchmark --package examples/qwen-hybrid-tiny \
  --requests 8 --input-tokens 6 --output-tokens 5 --output artifacts/benchmark.json
```

`run --op-trace` 另导出算子编码记录；`replay --journal` 重放控制动作。checkpoint 要求 quiescent，restore 校验当前 schema、weights/program/provider 与物理数据。历史截断不可视作完整 journal，协议见[生命周期](../architecture/request-lifecycle.md)。

benchmark 为 closed 测量；`compare` 校验同 workload/SLO、外部 correctness、goodput 与 P99。固定到达 HTTP 验证由 `make check-metal` 执行，结果范围见[验证记录](../validation/index.md)。Agent 命令与实际输出验收实验见[Agent](../architecture/agent.md)。

## 提交文件

提交 `crates/`、`tools/`、`docs/`、`examples/`、CI 与根配置；主 workspace 和 CPU benchmark 的两个 Cargo.lock 均保留，微型模型权重/tokenizer/golden 是必要测试资产。完整目标模型下载、本机环境、所有层级 target、Python 缓存与 artifacts 由[.gitignore](../../.gitignore)排除。`.env.example` 可提交，真实 `.env` 不提交。

`make clean-artifacts` 保留文档引用证据和模型资产，删除其他生成输出。提交前执行相关门禁，并用 `git status --short` / `git diff --cached --stat` 检查实际文件；完整质量规则见[质量门禁](quality-gates.md)。
