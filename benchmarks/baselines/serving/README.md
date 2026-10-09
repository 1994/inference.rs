# 服务基线证据

本目录保存已验收的服务基线。每个基线是一个**不可覆盖**的目录，由
[freeze-baseline.py](../../../tools/bench/freeze-baseline.py) 写出：

```sh
python3 tools/bench/freeze-baseline.py --baseline-id <id> --profile <profile> \
    --baseline <native-report.json> --candidate <reference-report.json> \
    --evidence <workload.json> <native.server.log> <telemetry.jsonl>
python3 tools/bench/freeze-baseline.py --verify benchmarks/baselines/serving/<id>
```

## 签发条件

冻结前由 [compare-results.py](../../../tools/bench/compare-results.py) 校验**有效性**：两边报告都带可验证的 release 构建身份、同一模型制品与硬件、一致的 cache 状态与资源约束，矩阵完整，且没有失败或截断的请求。有效性不通过就不签发。

性能结论按实测记录：测得退化仍是有效基线，`manifest.json` 的 `gate.passed` 如实写 `false`。任务质量不在本目录判定。

## 目录内容

| 文件 | 内容 |
|---|---|
| `manifest.json` | baseline/profile ID、生成时间、源码与硬件身份、被引用文件的哈希、gate 结论 |
| 两份报告 | 固定侧与候选侧的逐请求样本 |
| 证据 | workload、服务器日志、遥测等，按原文件名保存 |

manifest 最后写入：缺少它的目录是未完成的冻结。`--verify` 会重算所有哈希，任一文件缺失或被改动都会失败。

## 统计基础

方案要求正式测量至少五个独立配对单位。`manifest.json` 的 `statistical_basis` 如实记录本次冻结**实际**包含几个配对单位、声明的漂移容差，以及每个复测单元相对主配对的逐指标漂移：

```sh
python3 tools/bench/freeze-baseline.py --baseline-id <id> --profile <profile>     --baseline A B --reproduction A2 B2 [--reproduction A3 B3 ...] --max-drift 0.2
```

- 主配对与每个复测单元都必须各自通过有效性门禁；
- 漂移是「复测单元的配对比值」相对「主配对比值」的变化，逐指标记录，`worst_drift` 为最大值；
- 超出声明容差不会被拒绝，而是记 `within_declared_tolerance: false` 并保留测量——测得波动也是结果；
- 容差必须在测量前声明。若某次冻结的容差是事后选定的，必须在结论里说明，不能反过来当作已满足的统计要求。

### 源码必须是干净的

冻结要求两份报告的 `identity.source.dirty` 为 `false`。脏工作区意味着记录下来的 revision 不能唯一确定被测二进制，基线无法从它记录的版本重建。二进制哈希仍标识那次测量，但不足以签发基线。

## 已签发的服务基线

| baseline ID | profile | 模型 | MTP | 配对单位 |
|---|---|---|---|---|
| `2b-mtp0-serving-v1` | 2b-mtp0 | qwen3vl-2b | 0 | 待重建 |
| `27b-mtp0-serving-v1` | 27b-mtp0 | Qwen3.8-27B-NVFP4 | 0 | 待重建 |
| `27b-mtp2-serving-v1` | 27b-mtp2 | Qwen3.8-27B-NVFP4 | 2 | 待重建 |

## 与投影基线的区别

[上一层目录](../README.md) 的 cuTile 投影记录保持自己的测量层级，不用来补齐服务矩阵。服务对比的 release、配置对齐、计时统计与证据规则统一见[性能基线方案](../../../docs/plans/performance/baseline.md)。
