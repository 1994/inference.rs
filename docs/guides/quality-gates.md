# 质量门禁

统一入口 `make check`，任一步失败退出；CI 复用脚本，`quality-gate` 汇总 Linux/macOS Rust、工具、MSRV 与依赖安全 jobs，接入仓库后设为 required check。

## 检查规则

| 范围 | 要求 |
|---|---|
| 目录/依赖 | 职责分组、crate/path 注册、小型 facade、禁止反向 service/具体 backend 依赖、文档与源码链接 |
| Rust lint | fmt；Clippy all/pedantic/nursery deny；所有警告失败 |
| 长度/复杂度 | too_many_lines≤100、cognitive_complexity≤25；禁止 allow/expect 或调高阈值 |
| 错误路径 | 生产拒绝 unwrap/expect/panic/todo/unimplemented/dbg、忽略 Result |
| unsafe | 默认 deny；仅 OS placement FFI、Metal device FFI、隔离 benchmark allocator 开放并说明安全条件 |
| 局部例外 | 禁止关闭 lint group；allow/expect 有具体原因，失效 expect 失败；长度/复杂度无例外 |
| Features | 独立默认 CLI/IR/workspace 与 all-features；生产依赖图排除 CPU executor |
| 编译/测试/文档 | 发布构建、unit/integration/doctest、独立 golden；Rustdoc 警告失败 |
| CPU | release 协议、KV primitives、Engine tick 分配门禁；[计数范围](../validation/cpu.md) |
| MSRV | Rust 1.90 生产、全 workspace 与隔离 CPU benchmark |
| 依赖 | frozen lockfile；advisory/yanked、license/source allowlist、duplicate/wildcard 与闲置声明检查 |
| 凭证 | Gitleaks 脱敏扫描源码；有 Git 时另扫历史 |
| 工具 | Ruff lint/format、actionlint、workspace lint 继承 |
| Linux | 原生 affinity/cpuset/恢复与启动测试；Mac cross compile；NUMA 硬件专项见[Linux 指南](linux.md) |
| Metal | 实际 CLI/Agent/HTTP/SSE、golden、页压力、checkpoint、fixed-arrival HTTP 与资源排空 |

Clippy restriction 逐项启用。测试断言可用 unwrap/expect/panic，helper 返回 Result；数值/接口例外局部说明，独立 golden 保留运算顺序。

## 执行入口

`make check-rust`、`check-tools`、`check-security`、`check-msrv`、`check-cpu`、`check-metal`、`check-linux`、`check-linux-numa` 可独立运行。CPU benchmark 的 manifest/lockfile 同样受 MSRV/安全门禁。

macOS 的完整本地门禁包含真实 Metal，缺少设备时专项失败；公共 CI 的编译不能替代硬件验收。Linux 原生测试在 required Rust job 执行，NUMA 专项要求相应 syscall 权限。CUDA 硬件门禁随 5090 原生执行器接入。

## 工具与依赖例外

Rust 1.99.0/rustfmt/Clippy 由 rust-toolchain.toml 固定；另需 Rust 1.90.0、Python 3.12+、uv、Go、cargo-deny 0.20.2、cargo-audit 0.22.2。Ruff 0.15.7、actionlint 1.7.7、Gitleaks 8.24.3。CI actions 固定 commit，权限 contents/read。

```sh
rustup toolchain install 1.90.0 --profile minimal
cargo install --locked cargo-deny --version 0.20.2
cargo install --locked cargo-audit --version 0.22.2
make check
```

[deny.toml](../../deny.toml)记录 paste 1.0.15 的 RUSTSEC-2024-0436 例外与两条精确重复版本例外；原因和移除条件以配置为准。[policy.py](../../tools/check/policy.py)另校验 workspace 继承与闲置依赖。

Metal 间接依赖 block 0.1.6 有已知 future-incompatibility 报告，作为上游迁移项保留；不通过全局 RUSTFLAGS/allow 隐藏。实际通过的门禁和证据见[验证记录](../validation/index.md)。
