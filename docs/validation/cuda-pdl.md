# MLP 的可选 PDL 依赖边

2026-10-06：RTX 5090 / CUDA 13.4.92 / cuTile Rust 0.4.0。

## 实现范围

`MlpConfig.pdl` 控制 `SiLU(gate) × up → down projection` 一条边，默认关闭。模型诊断入口通过 `--mlp-graph --mlp-pdl` 显式开启；报告记录两个开关。其余 kernel 使用普通有序提交，尚未给整个模型启用 PDL。

同步契约：

1. 生产者每个 CTA 将 `store` 返回的 token 交给 `gdc_launch_dependents_tko`。
2. 消费者先取 `gdc_wait_tko` 的 token，再通过 `input.set_token` 约束所有激活读取。仅把 wait 写在 load 前面不够。
3. BF16/FP8/NVFP4 消费者的权重和 scale 不可变；图持有固定缓冲区直到最后一次使用完成。
4. host 只给这些专用消费者设置 programmatic launch 属性，生产者与消费者位于同一捕获 stream。两者均不依赖并发执行才能完成。

unsafe 仅位于 `mlp/pdl.rs` 与 `mlp/pdl_consumers.rs`，附有上述安全证明；工作区其它代码仍拒绝 unsafe。普通路径不引用 PDL 指令，保留旧工具链兼容性。显式请求 PDL 时，launcher 校验 Tile IR 13.4 和架构；**不承诺整个 API 在不支持的平台自动 no-op**。

## 已验证

- 三种存储精度，hidden=48、intermediate=80/560，包含多 CTA 和尾块。
- 每种配置同一图重放 32 次，输入轮换正数、负数和零，共 192 次；结果通过独立公式参考检查。
- 调用方释放 weight handles 后仍可重放，检查图保留资源生命周期。
- Tile IR 明确显示消费者 `load_view_tko` 依赖 `gdc_wait_tko` 返回 token；生产者 signal 依赖 store token。
- 所有 GPU 检查在 `safe-run.sh` 的内存隔离内运行；未加载大模型。

```sh
bash tools/bench/safe-run.sh "$PWD/target/debug/examples/cuda-mlp-check" --pdl
```

本机证据：`artifacts/mlp-pdl-check.log`（含 IR）、`artifacts/mlp-pdl-multicta-check.log`。

## 尚未验证

这只是正确性和依赖顺序验证。尚无 launch gap trace 或独立 PDL on/off 性能数据，因此不默认开启、不声明加速。PDL 不会减少 CPU launch 数量，也不保证 CTA 并发；本实现选择写完再 signal，能重叠的尾部工作可能很少。后续在常驻执行器上用独立 profiler run 检查间隙和资源竞争，再决定是否将 signal 提前并加载独立权重。
