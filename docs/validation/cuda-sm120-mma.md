# RTX 5090：NVFP4 scaled MMA 能力验证

2026-10-06，实测 RTX 5090 / sm_120，驱动 615.71.09，CUDA 13.4.92，cuTile Rust 0.4.0。

## 结果

**原生 NVFP4 block-scaled MMA 路径存在，但不能只用架构编号或数值通过作为快路径判据。**

| Tile M×N×K | 矩阵 M×N×K | 数值最大绝对误差 | release lowering |
|---|---|---:|---|
| 16×16×64 | 16×16×128、17×19×128 | 0 | 未出现 MMA 指令 |
| 32×64×128 | 32×64×256 | 0 | 原生 scaled MMA |
| 64×64×128 | 64×64×256 | 0 | 原生 scaled MMA |
| 128×128×128 | 128×128×256 | 0 | 原生 scaled MMA |

较大 tile 的 sm_120 cubin 经 CUDA `cuobjdump --dump-sass` 检查包含：

```text
OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
```

输入包含全部 E2M1 编码、正负值、每 16 元素变化的 E4M3 scale（0.5/1/2），与独立 CPU F64 矩阵乘法比较。覆盖 K 多 tile 累积、非整齐 M/N 边界。debug 小 tile 同样数值正确，但没有 MMA 指令；release 小 tile 也没有。因此不能把本次结果外推到任意 tile、工具链版本或完整模型吞吐。

`tcgen05` 与 `mmaf_scaled` 属于不同抽象层：前者不受 sm_120 支持，不能推出后者无法由编译器降低为适用该架构的指令。[CUTLASS #2800](https://github.com/NVIDIA/cutlass/issues/2800) 描述特定 CuTe DSL 操作的架构限制，查阅时已关闭；它不能替代 cuTile 实测。[cuTile NVFP4 文档](https://nvlabs.github.io/cutile-rs/main/tutorials/11-nvfp4-inference.html) 描述 Tile IR 路径。

## 复现与证据

- 探针：`crates/backend/cuda/examples/cuda-mma-probe.rs`。
- 结果及 cubin SHA256/指令计数：`benchmarks/capabilities/rtx5090-nvfp4-mma.json`。
- 本机原始 Tile IR、JSON、cubin/SASS：`artifacts/mma-probe-tiles*`。
- 受保护进程组运行约 2.58 秒，峰值主机内存 181.3 MiB，swap 0；没有加载大模型，没有测量吞吐。

```sh
CARGO_BUILD_JOBS=1 CUDA_TOOLKIT_PATH=/opt/cuda \
  BINDGEN_EXTRA_CLANG_ARGS='-isystem /usr/lib/gcc/x86_64-pc-linux-gnu/16/include' \
  cargo +stable build --release -p infer-backend-cuda --features cuda --example cuda-mma-probe
bash tools/bench/safe-run.sh /usr/bin/env \
  XDG_CACHE_HOME="$PWD/artifacts/mma-probe-cache" CUTILE_DUMP=ir \
  "$PWD/target/release/examples/cuda-mma-probe" \
  > artifacts/mma-probe.json 2> artifacts/mma-probe.log
python3 tools/bench/inspect-cutile-cache.py --cache artifacts/mma-probe-cache \
  --output artifacts/mma-probe-disassembly
```

## 策略影响

5090 的后续 prefill/批量 verify 可以继续验证 Rust NVFP4 MMA；候选 tile 必须经过 lowering、数值和性能三项验证。当前 weight-only GEMV 并未因此自动变成 W4A4 Tensor Core kernel，激活动态量化仍待实现。H200 的 FP8/BF16 策略保持独立，本结果不授权 H200 使用原生 FP4。
