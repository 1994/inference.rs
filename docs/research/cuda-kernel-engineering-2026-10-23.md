# CUDA kernel 工程整理与 45 重建记录

日期：2026-10-23（America/Los_Angeles）。源码基线为 `8747225bbf4da4725e6b12f511e348cc007eecf1`。
45 的活跃 checkout 已有后续提交并正在运行服务基准，因此工作使用独立分支
`codex/kernel-engineering`，目录 `/home/r/code/inference-kernel-engineering-20261023`。
本轮没有修改运行时 Rust、kernel 源码、嵌入 PTX/SO、数值 profile 或 tuning fingerprint。

## 发现与落实

- 8 组 AOT 制品分别维护脚本，原 CI 漏掉 oxide attention/GDN，F32 bridge 校验还被错误地归入 Linux-only 部分。
- 5 份重复脚本收敛为同一校验器；FA4 特有的 ABI/ELF 验证继续委托冻结脚本，避免改动其哈希身份。
- `tools/cuda/kernels.json` 成为目录/目标/符号/生成器清单；新增 PTX 未登记会被拒绝。
- 新入口提供 `list`、`verify`、`rebuild`。重建只写新目录，记录输入哈希、真实命令、日志、成功/失败结果。
- FP16 AOT 生成器从 45 的 staging 收回仓库，去除机器路径依赖；固定上游源码、SDK、依赖版本与全部生成产物哈希。
- 45 原有 quack/tvm-ffi 的 uv 缓存目录已消失；本次在独立工作目录安装固定版本构建依赖，完整重建成功。
- 两个 oxide 原 provenance 指定的编译器后端 `.so` 已消失。统一入口校验哈希并明确失败，不自动换工具链。

技术边界及后续准入方法见 [Kernel 制品工程](../../crates/backend/cuda/kernels/README.md)。
新增 tile 算子继续采用 cuTile，新增精细 SIMT 统一 nvcc，FA4 作为第三方离线 AOT 库。
两个现有 oxide kernel 保留，后续逐算子迁移需要独立数值、CUDA Graph、质量和性能证据。

## 验证结果

45：RTX 5090，driver 615.78.08，nvcc 13.4.92。AOT host 编译使用现有 GCC 16.2.1。
以下均为 CPU 编译/完整性验证；未执行 GPU kernel，也未占用正在运行的服务基准显存。

| 项目 | 结果 |
|---|---|
| 全部 8 组源文件、依赖、PTX 和 AOT bundle | 统一完整性校验通过 |
| Linux FA4 指令、host ABI 与动态依赖 | 通过 |
| shared / vector / combined attention | 三组 nvcc PTX 逐字节重建一致 |
| F32 KV bridge | PTX 逐字节重建一致 |
| FP16 FA4 SM120 | PTX、cubin、host header/object、SO、bridge 全部逐字节一致 |
| BF16 FA4 六目标 | 六组 PTX、cubin、host header/object、SO 及 bridge 全部逐字节一致 |
| oxide attention / GDN | 完整性通过；固定编译器后端缺失，未完成源码重建 |
| 新增完整性/失效测试 | 8 项通过（本地及 45） |
| 本地 tools/check 测试集 | 89 项，88 通过、1 跳过 |
| 45 目录、文档链接、架构策略检查 | 通过 |
| 新工具 Ruff 与变更空白检查 | 通过 |

完整最终构建记录保存在 45 的工作目录 `artifacts/kernel-engineering-final/<kernel>/result.json`，
其旁是日志和真实执行命令。首次调试和明确失败的记录保留在 `artifacts/kernel-rebuild/`。
独立依赖保存在 `artifacts/fa4-build-deps/`，不是运行时依赖。

本轮证明工程入口能复现现有产物。没有新的 GPU 数值、模型质量或服务吞吐结论；
其他 GPU 架构仍仅为交叉编译，不因本轮重建而获得硬件资格。
