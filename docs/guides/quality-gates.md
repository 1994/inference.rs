# 质量门禁

所有检查共用统一入口 `make check`，任一步失败即退出。CI 与本地使用同一脚本，`quality-gate` job 汇总 Rust、工具、MSRV 与依赖安全结果。

## 检查项

| 范围 | 要求 |
|---|---|
| 目录与依赖 | crate 分组、路径注册、小型 facade、禁止实现层反向依赖服务或具体 backend、本地文档链接有效 |
| Rust lint | `cargo fmt --check`；Clippy all / pedantic / nursery 全部 `-D warnings` |
| 长度与复杂度 | `too_many_lines` ≤ 100、`cognitive_complexity` ≤ 25，不允许调高阈值 |
| 错误路径 | 生产代码禁止 `unwrap` / `expect` / `panic` / `todo` / `unimplemented` / `dbg` 与忽略 `Result` |
| unsafe | 默认 `deny`；仅 OS 放置 FFI、Metal device FFI 与隔离的 benchmark allocator 开放，并说明安全条件 |
| Feature 隔离 | 默认 CLI / IR / workspace 与 `--all-features` 分别构建；生产依赖图不含 CPU 执行器 |
| 编译与文档 | release 构建、unit / integration / doctest、独立 golden、Rustdoc 警告失败 |
| CPU | release 协议、KV primitives、Engine tick 分配门禁，范围见 [CPU 验证](../validation/cpu.md) |
| MSRV | Rust 1.90 检查 workspace（不含 CUDA）与隔离的 CPU benchmark |
| 依赖 | frozen lockfile；advisory / yanked、license / source allowlist、重复与通配版本、闲置声明 |
| 凭证 | Gitleaks 扫描源码；在 Git 仓库中另扫历史 |
| 工具 | Ruff 检查与格式化、actionlint、workspace lint 继承 |
| Linux | 原生 affinity / cpuset / 失败恢复与 owner 启动；Mac 上交叉编译检查 |
| Metal | 真实 CLI / Agent / HTTP / SSE、golden、页压力、checkpoint、固定到达负载与资源排空 |

## 执行入口

```sh
make check            # 全部（macOS 上包含 Metal 实机）
make check-rust       # 目录、lint、测试、文档、release、CPU 分配与 golden
make check-tools      # Ruff 与 actionlint
make check-security   # cargo-deny、cargo-audit、Gitleaks
make check-msrv       # Rust 1.90
make check-cpu        # 隔离的 CPU 分配计数
make check-metal      # Metal 实机验收（仅 macOS）
make check-cuda       # CUDA 实机 kernel 验收（需 Toolkit 与 GPU）
make check-linux      # Linux 原生放置测试
make check-linux-numa # Linux NUMA 硬件验收，见 Linux 指南
```

`check-rust` 会单独构建 CUDA crate 的无 feature 版本，因此托管 CI 不需要 CUDA Toolkit；设备检查集中在 `check-cuda`。缺少 GPU 或相应 syscall 权限时，硬件门禁明确失败，不会跳过。

## 工具链与版本

| 工具 | 版本 |
|---|---|
| Rust（固定） | 1.99.0，由 `rust-toolchain.toml` 提供 |
| Rust（MSRV） | 1.90.0 |
| Python | 3.12+ |
| uv | 任意近期版本 |
| Go | 用于 `go run` 执行 actionlint 与 gitleaks |
| cargo-deny / cargo-audit | 0.20.2 / 0.22.2 |
| Ruff / actionlint / Gitleaks | 0.15.7 / 1.7.7 / 8.24.3 |

```sh
rustup toolchain install 1.90.0 --profile minimal
cargo install --locked cargo-deny --version 0.20.2
cargo install --locked cargo-audit --version 0.22.2
make check
```

## 例外

- [deny.toml](../../deny.toml) 记录 `paste 1.0.15`（RUSTSEC-2024-0436）与两条精确重复版本例外，原因和移除条件以配置为准。
- `tools/check/policy.py` 另外校验 workspace lint 继承与闲置依赖。
- Metal 间接依赖 `block 0.1.6` 有已知 future-incompatibility 报告，作为上游迁移项保留，不用全局 `RUSTFLAGS` 隐藏。
- 测试代码允许 `unwrap` / `expect` / `panic` 用于断言，helper 仍返回 `Result`；其他数值或接口例外需在局部说明。
- 可选的官方 tokenizer parity 测试通过 `INFER_QWEN_TEXT_PACKAGE` 指向外部包，普通 CI 使用仓库内微型资产。
