# KV Cache Manager 与页缓存

KV 管理分为 scheduler 的逻辑页预算和 backend owner 的物理页 / 私有状态。共享实现 `KvCacheManager<P>` 组合 BlockPool 与 PrefixCache，设备通过 copy plan 与 lease 接入；manager 本身不调用 CUDA / Metal API。

## 所有权与操作

| 操作 | 合约 |
|---|---|
| `page_growth` / `prepare_append` | O(1) growth 报价；原子预留新增页与共享尾页 COW，返回复制计划 |
| `fork_pages` | 保留共享 lease，不复制完整 KV |
| `attach_prefix` | 匹配 namespace 与最长完整块前缀，原子取得引用 |
| `publish_prefix` | cache 取得独立引用，LRU 淘汰只释放自身所有权 |
| `pin` / `release` | GPU copy / reader 期间保留；最后一个 fence 与引用归零后可回收 |
| `inspect` / `check_owners` | 冷路径核对 unique block、refcount、generation、owner 与 cache |

lease 为 `(OwnerId, index, generation)`，跨池、旧 generation 和重复确认都会被拒绝。pool owner 不可 clone，页表使用设备 `u32` index。逻辑增页是批次事务：物理编码失败时回滚新增 lease 与页表；已发布的设备操作保留其 pin 直到 fence。引用计数校验使用固定 marker，append / COW 使用常驻 lease scratch。分配证据范围见 [CPU 验证](../validation/cpu.md)。

## StateRecipe 与预算

加载期从编译图生成 checked `StateRecipe`，分别布局私有 conv / delta、hidden / logits、tokens / page table / probes、host mirror / readback / lease 与共享 KV block。Metal 用 recipe 分配和报价，reset 复用已分配 buffer。

准入保留私有状态与最大 token / page descriptor 容量，KV 按实际计算增页。单请求最大计算长度必须能独占页池，tenant 配额按最坏 token / page 数计算；只分配零 KV 的准入不能被当作零状态内存。

```text
available_blocks = free_blocks + unpinned_cache_only_blocks
```

active 与 cache 可以共享同一块，容量按 unique block 统计；pinned cache-only 不可回收。压力下先淘汰 cache-only，再由调度策略选择安全的重算 victim。固定驻留字节、私有状态、cache snapshot 与 host mirror 分别统计。

## Metal 布局、COW 与 prefix

- 每层 K/V plane 形状为 `[blocks, page_tokens, kv_width]`，sequence 用持久页表定位块；conv / delta / hidden / logits 为私有设备 buffer。attention 的 KV 写入与读取分 pass。
- 写入共享尾页时用 GPU blit 复制各层 K/V，保留源页 pin，然后原子切换写入方页表。失败时恢复旧页表与新增 lease；host 侧发布顺序不能替代 GPU 复制依赖。
- Prefix namespace 绑定精确 weights、backend、precision、layout 与 token hash-chain。只发布完成 GPU 边界的完整块，并保存对应 conv / delta / readout 状态；命中时共享 KV，并在 GPU 内恢复私有状态，普通路径不读回完整张量。最后 prompt token 需要重算以取得最终 logits。
- 缓存受条数 / 字节与 LRU 约束。Full snapshot 可用于 compact Generate，compact snapshot 不支持 Full readout。取消与淘汰独立于活动引用，不能释放仍被 flight 或 CPU reader 使用的状态。

## Checkpoint 与扩展边界

quiescent checkpoint 冷读回实际状态并记录共享页关联；恢复时先校验身份、shape、有限值、容量、generation 与 payload 一致性，再创建新池，同一共享页只分配一次。失败保留现有状态；prefix cache 与 debug history 不作为恢复缓存。

当前物理分配与淘汰的 owner 位于 backend。统一到 scheduler 决策 ledger、异构状态组、分层 offload 与远程 PD transfer 的目标见[实现状态](../design/status.md)。
