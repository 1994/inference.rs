# 工具导航

统一入口是仓库根目录的 `make check`，各分项见[质量门禁](../docs/guides/quality-gates.md)。工具输出归 `artifacts`，不进入产品运行依赖。

| 目录 | 入口 | 用途 |
| --- | --- | --- |
| `check` | [gate.sh](check/gate.sh) | 本地与 CI 共用的 fail-fast 门禁 |
| `check` | [layout.py](check/layout.py)、[policy.py](check/policy.py) | 目录/依赖边界、导航、源码与样例路径、lint/feature 规则 |
| `check` | [clean-artifacts.py](check/clean-artifacts.py) | 删除生成输出，保留文档引用及关联日志、模型/tokenizer 资产 |
| `check` | [metal.py](check/metal.py) | 强制真实 Metal 的 CLI、Agent、HTTP 与 KV 验收 |
| `validation` | [smoke-host.py](validation/smoke-host.py) | 所选 backend 的命令行、Agent、HTTP/SSE 与观测检查 |
| `validation` | [smoke-paged-kv.py](validation/smoke-paged-kv.py) | 页池压力、抢占、checkpoint、golden 与所有权排空 |
| `validation` | [cpu-open-loop.py](validation/cpu-open-loop.py) | 固定到达 HTTP 负载、控制延迟与显式容量拒绝 |
| `fixtures` | [export-qwen-golden.py](fixtures/export-qwen-golden.py)、[export-text-golden.py](fixtures/export-text-golden.py) | 独立参考的模型与文本 golden 导出 |
| `fixtures` | [requirements.txt](fixtures/requirements.txt) | 参考导出环境的固定依赖 |
| `bench/cpu` | [Cargo.toml](bench/cpu/Cargo.toml) | 隔离的 release 分配计数；通过 `make check-cpu` 运行 |

服务检查依赖已构建的 CLI，具体参数通过各脚本的 `--help` 查看。参考模型导出示例见[微型模型说明](../examples/qwen-hybrid-tiny/README.md)。CPU 基准的计数范围见[CPU 验证](../docs/validation/cpu.md)；包含协议循环、KV primitive 与真实 Engine 主线程热路径；各自范围分开记录，不包含整个服务或原生模型执行。

`make clean-artifacts` 清理历史验收输出、调试日志和可重建的 golden Python 环境。可先运行 `python3 tools/check/clean-artifacts.py --dry-run` 查看清单；保留文档链接的证据及其 summary 日志、顶层模型文件和模型/tokenizer 包。参考导出环境按样例说明重新创建，普通 Rust/Metal 验证直接使用已提交的 golden。
