# Linux 与 CPU 放置

Linux 是首要生产平台。本页说明线程放置、NUMA 配置和硬件门禁；CUDA 执行器的实现范围见[能力边界](../architecture/backends.md)。

## 放置配置

`RuntimeConfig.cpu.placement` 分别配置 `scheduler`、`device` 和 `output` 三类 owner，Frontdoor 的准备 worker 使用 `CpuConfig.placement`。默认不指定核号，不猜测测试机拓扑。

```json
{
  "cpu": {
    "placement": {
      "scheduler": {"cpus": [2], "numa": {"Bind": 0}},
      "device": {"cpus": [4], "numa": {"Bind": 0}},
      "output": {"cpus": [6, 8], "numa": {"Prefer": 0}}
    }
  }
}
```

上例中的核号、PCI 地址与 NUMA 节点均为示意值，必须替换为目标机器实际允许的值。

Rust 嵌入入口可用 `OwnerPlacement::for_gpu("0000:01:00.0")` 解析 PCI 设备的 NUMA 节点，并在继承的 cpuset 内为 scheduler / device 选择两个不同物理核。output 与 preparation 的核预算单独配置，避免与关键 owner 争用同一物理核。

## 发现与约束

- 读取线程实际 affinity、`/proc/self/status` 的 `Mems_allowed_list`、SMT sibling 和 GPU PCI `numa_node`。
- CPU bitmap 按内核要求动态扩展，不受 libc `cpu_set_t` 1024 位限制；NUMA bitmap 按系统 possible node 列表确定大小，并遵循内核 maxnode 边界约定。
- GPU 节点为 -1、节点不在 memory cpuset 内、可用核不足或配置超出继承 CPU mask 时，启动直接报错；不会擅自扩大容器 / cgroup 的资源集合。
- `ThreadPlacement::scope` 在冷初始化前设置 affinity 与 memory policy，结束时或初始化失败时恢复调用线程；`spawn` 在线程应用策略成功后才完成启动握手。
- NUMA 策略只影响之后首次触碰的页面。已驻留页和 mimalloc 复用的页面不会自动迁移，因此“配置生效”与“页面实际位于目标节点”需要分别验证。

## 门禁

```sh
make check-linux       # Linux 原生测试；Mac 上只检查 Linux 专属代码的编译与 lint
make check-linux-numa  # Linux 硬件测试，需要允许 NUMA 策略与 move_pages 查询
```

`check-linux` 覆盖原生绑核、非法 cpuset 拒绝、失败恢复与 owner 启动，要求安装 `x86_64-unknown-linux-gnu` target（跨编译不执行 Linux 测试）。`check-linux-numa` 验证 Bind/Prefer 的实际线程策略、scope 失败恢复、worker 策略，以及匿名新映射首次触碰后每页的实际 NUMA 节点；它用 `move_pages` 的查询模式读取驻留位置，不迁移页面。容器 seccomp / 权限禁止相关 syscall、内存不足或驻留不符都会使专项门禁失败。

这两个硬件测试在普通 CI 中标记为 `ignored`，必须由目标机器上的专项门禁执行；跨编译成功不代表 NUMA 验收通过。验收应保存日志、进程 cpuset、GPU PCI/NUMA 与 CPU 拓扑，并核对真实引擎池的页面驻留、迁核次数、CPU P99 与 goodput。当前本机证据范围见 [CPU 性能测量](cpu-performance.md)。
