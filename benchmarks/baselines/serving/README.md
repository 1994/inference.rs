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

## 与投影基线的区别

[上一层目录](../README.md) 的 cuTile 投影记录保持自己的测量层级，不用来补齐服务矩阵。服务对比的 release、配置对齐、计时统计与证据规则统一见[性能基线方案](../../../docs/plans/performance/baseline.md)。
