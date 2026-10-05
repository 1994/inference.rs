# Linux 生产平台与 CPU 放置

Linux 是首要生产平台；本页说明 CPU 放置与硬件验收。CUDA 执行器和目标设备的实现范围见[状态表](../design/status.md)。

## 启动配置

`RuntimeConfig.cpu.placement` 分别配置 `scheduler`、`device` 和 `output`。Frontdoor 的准备 worker 使用 `CpuConfig.placement`。默认不指定核号，不猜测测试机拓扑。

Rust 嵌入入口可用 `OwnerPlacement::for_gpu("0000:01:00.0")` 解析 PCI 设备的 NUMA 节点，并在继承 cpuset 内选择两个不同物理核给 scheduler/device。地址仅示意，必须使用测试机的实际 PCI 地址。输出和 preparation 核预算单独配置，避免与关键 owner 争用同一物理核。

显式配置的结构如下；核号与 NUMA 节点同样必须替换为目标机器实际允许的值：

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

发现代码读取线程实际 affinity、`/proc/self/status` 的 `Mems_allowed_list`、SMT sibling 和 GPU PCI `numa_node`。CPU bitmap 按内核要求动态扩展，不限制在 libc `cpu_set_t` 的 1024 位；NUMA bitmap 按系统 possible node 列表确定大小，并遵循内核 maxnode 的边界约定，保留 word 末位节点。[Linux nodemask ABI 源码](https://github.com/torvalds/linux/blob/master/mm/mempolicy.c)。GPU 节点为 -1、节点不在 memory cpuset 内、核不足或配置超出继承 CPU mask 时返回启动错误。不会擅自扩大容器/cgroup 的资源集合。

`ThreadPlacement::scope` 在冷初始化前设置 affinity/memory policy，结束或初始化失败时恢复调用线程。`spawn` 在线程应用策略成功后才完成启动握手。Engine、device owner、output scratch 与 preparation pool 的冷初始化使用对应 scope，长期 worker 使用对应 spawn；策略失败时不接受请求。

NUMA 策略影响之后首次触碰的页面，已驻留页和 mimalloc 重用的页面不会自动迁移。因此设置成功、配置报告和物理页面驻留分别验证，不能把配置报告当作全部池都已位于目标节点的证据。[Linux 内核 NUMA 策略说明](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)

## 固定门禁与硬件验收

```sh
make check-linux       # Linux 原生测试；Mac 上检查 Linux 专属代码和测试的编译/lint
make check-linux-numa  # Linux 硬件测试，必须允许 NUMA 策略及 move_pages 查询
```

Linux CI 的 required Rust job 包含原生绑核、非法 cpuset 拒绝、失败恢复及 owner 启动测试。Mac 开发机需要安装 `x86_64-unknown-linux-gnu` Rust target；跨编译不执行 Linux 测试。

NUMA 硬件门禁验证 Bind/Prefer 的实际线程策略、scope 失败恢复、worker 策略，以及匿名新映射首次触碰后每页的实际 NUMA 节点。它通过 `move_pages` 的查询模式读取驻留，不迁移页面。容器 seccomp/权限禁止相关 syscall、内存不足或驻留不符均使专项门禁失败。普通 CI 中这两个硬件测试显式标为 ignored，必须由目标机器上的专项门禁执行；跨编译成功不代表 NUMA 验收成功。

验收保存日志、进程 cpuset、GPU PCI/NUMA 与 CPU 拓扑，并核对真实引擎池驻留、迁核次数、CPU P99 与 goodput。本机证据见[CPU 验证](../validation/cpu.md)。
