# 工具导航

统一入口是仓库根目录的 `make check`，各分项见[质量门禁](../docs/guides/quality-gates.md)。工具输出写入 `artifacts/`，不进入产品运行依赖。

| 目录 | 入口 | 用途 |
|---|---|---|
| `check` | [gate.sh](check/gate.sh) | 本地与 CI 共用的 fail-fast 门禁 |
| `check` | [layout.py](check/layout.py)、[policy.py](check/policy.py) | 目录与依赖边界、文档链接、lint/feature 与闲置依赖规则 |
| `check` | [clean-artifacts.py](check/clean-artifacts.py) | 删除可重建输出，保留文档引用的证据与模型资产 |
| `check` | [metal.py](check/metal.py) | 真实 Metal 的 CLI、Agent、HTTP 与 KV 验收 |
| `validation` | [smoke-host.py](validation/smoke-host.py) | 所选 backend 的 CLI、Agent、HTTP/SSE 与观测检查 |
| `validation` | [smoke-paged-kv.py](validation/smoke-paged-kv.py) | 页池压力、抢占、checkpoint、golden 与所有权排空 |
| `validation` | [cpu-open-loop.py](validation/cpu-open-loop.py) | 固定到达 HTTP 负载、控制延迟与显式容量拒绝 |
| `fixtures` | [export-qwen-golden.py](fixtures/export-qwen-golden.py)、[export-text-golden.py](fixtures/export-text-golden.py) | 独立参考的模型与文本 golden 导出 |
| `fixtures` | [requirements.txt](fixtures/requirements.txt) | 参考导出环境的固定依赖 |
| `bench/cpu` | [Cargo.toml](bench/cpu/Cargo.toml) | 隔离的 release 分配计数与协议测量，通过 `make check-cpu` 运行 |
| `bench` | [cuda-baseline.sh](bench/cuda-baseline.sh) | CUDA 投影基线采集，用法见 [CUDA 性能指南](../docs/guides/cuda-performance.md) |

服务检查依赖已构建的 CLI，参数通过各脚本的 `--help` 查看。参考模型导出示例见[微型模型说明](../examples/qwen-hybrid-tiny/README.md)；CPU 基准的计数范围见 [CPU 验证](../docs/validation/cpu.md)。

```sh
python3 tools/check/clean-artifacts.py --dry-run   # 预览将被删除的生成输出
make clean-artifacts                              # 实际删除
```

清理会移除历史验收输出、调试日志与可重建的 golden Python 环境，但保留文档链接的证据及其 summary 日志、顶层模型文件与模型 / tokenizer 包。参考导出环境按样例说明重新创建；普通 Rust / Metal 验证直接使用已提交的 golden。
