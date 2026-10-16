# 并行 WY 完整质量回归与解码成本（2026-10-13）

新并行准备方案把长提示首 token 从同轮 178.948 ms 降到 145.741 ms，降低 18.557%；
固定 vLLM 为 62.744 ms，候选仍是其 2.323 倍。完整 256 题回归答对 241，旧 WY 为 243。
相对 scalar 的逐题回退率单侧 95% 上界为 3.001%，超过原 2% 门槛。
该候选不采用，也未合并到 main。计时通过不能替代质量通过。

## 固定工作量的服务计时

同一台验证 GPU 上两轮反序进程对比，顺序 old/prepared/vLLM 和 vLLM/prepared/old。
所有六项启动/捕获时后台 GPU 进程一致性检查通过，保留其他进程。
实际输入为 51/511 token，四路并发各约 52 token；MTP=0，每路输出恰好 64 token，
prefix reuse=0，测量进程禁用新调参，所有请求完成。
每轮每类工作量三次测量，以下取两轮进程中位数的中位数，单位 ms。
这是两进程轮次的实验，未达到五进程统计验收，也不外推其他模型或 MTP 配置。

| 指标 | 旧 WY | 并行候选 | vLLM | 相对旧 WY 延迟降低 |
|---|---:|---:|---:|---:|---:|
| 短提示 TTFT | 36.858 | 31.721 | 45.554 | 13.938% |
| 短提示 TPOT | 16.161 | 16.133 | 14.778 | 0.174% |
| 短提示 wall | 1055.829 | 1048.702 | 977.527 | 0.675% |
| 长提示 TTFT | 178.948 | 145.741 | 62.744 | 18.557% |
| 长提示 TPOT | 16.123 | 16.108 | 14.826 | 0.092% |
| 长提示 wall | 1195.013 | 1161.727 | 996.631 | 2.785% |
| 四路并发 TTFT | 149.247 | 138.921 | 83.694 | 6.919% |
| 四路并发 TPOT | 19.382 | 19.497 | 16.419 | -0.594% |
| 四路并发 wall | 1371.600 | 1374.451 | 1120.827 | -0.208% |

decode 基本未改变；长提示 wall 仅降低 2.785%，四路并发 wall 增加 0.208%。
候选相对 vLLM 的逐 token 延迟仍高约 9%，四路并发 wall 高 22.628%。
不能将这个结果描述成整体吞吐已经接近 vLLM。

旧 native CLI SHA256 `4be5d78d71c995e773c9571b0bf23fadb57796d8d44439c96ec1d8f5602d8440`；
并行 CLI SHA256 `ce9107738c2e2e6a824e6cc3de9cc73185d6192baa7ecd1d670a0a7e7af282df`。
候选 cache SHA256 `c2c7a620e06df0d3fd7bfb1d309ae7fcbb6adb7c9528ce2ae50e0315a59d2729`。
保留原 GEMM cache，只追加已测量的新 GDN 策略。vLLM 运行时仍冻结为 3fe917abb，
研究源码为已拉取的 main，二者不能混称同一版本。
实验根目录仍是 `/home/r/benchmarks/inference.rs/vllm-parity/c5aec9e/first-principles/`；
原始服务与资格记录在 `gdn-wy-parallel-perf-v1`。

## 完整质量结论

使用原始 256 题、原始 tokenized 输入、自然 EOS、max512、temperature=0、thinking off、
MTP=0；不增加输出预算，不删除 Length 或缺答案的请求。
256 请求 prefix reuse 均为零，binary/cache 前后不变，后台 GPU compute 保留且不变。
独立 `score-task-quality.py` 从完整原始文本复算，原 2% 和单侧 95% 门槛不变。

| 基线 | 基线答对 | 候选答对 | 回退 | 新增答对 | 回退率上界 | 验收 |
|---|---:|---:|---:|---:|---:|---|
| scalar | 240 | 241 | 3 | 4 | 3.001% | not_demonstrated |
| 旧 WY | 243 | 241 | 5 | 3 | 4.063% | not_demonstrated |
| vLLM | 243 | 241 | 6 | 4 | 4.573% | not_demonstrated |

候选短提示 117/128、八例长提示 124/128。相对 scalar 的回退为
`gsm8k:806:zero_shot`、`gsm8k:154:zero_shot`、`gsm8k:293:zero_shot`。
此前六道诊断题中没有 scalar 回退，不能代表完整题集。
总体多答对一题不能掩盖其他题的回退，也不能按诊断题调好参数后直接宣称整体通过。

完整原始结果 SHA256 `1ca1dd6d8d467d805574d7037395b051f24c54b59aebe26d117530a9d418a800`。
输入 suite SHA256 `adb429cdc275400fef3c3e6331b942a38237b2d5ca07f4aad5948ebf06fb94e6`，
质量 plan SHA256 `991ac5b21d7862270477e561aaa9e6fd1a5803626e014870890fbad60f416c83`。
原始与独立评分目录为 `gdn-wy-parallel-quality-v1`。
完整可编译候选源码、patch 和哈希保存在
`gdn-wy-parallel-prepare-f32-v1/failed-full-model-prototype-source`，未丢弃失败记录。

## 原生产解码图的已有成本证据

复用先前原生产版本的 CUDA event JSONL，不额外运行模型。
排除四路并发第一次预热的所有 slot 事件后，保留 191 个测量 replay；
其中四路都活跃的 184 个 replay，图耗时中位数为 21.714 ms。

| segment 类别 | 中位数 ms |
|---|---:|
| linear | 17.189 |
| delta | 1.180 |
| norm | 0.795 |
| add | 0.510 |
| attention | 0.491 |
| conv | 0.475 |

linear / 全图两个中位数之比为 79.164%。linear segment 包含量化与矩阵计算，
不等于裸 GEMM；节点间事件也增加了开销，不能拿 21.714 ms 替换无探针 TPOT。
这里的 tokens 是活跃槽数，物理 capture 宽度另由代码及调参 shape 核实。

原 JSONL SHA256 `a268b1e90f3cf228fb1717a50cdcb121bb20c185f81c168eb373c48530c76f0b`，
汇总 `gdn-wy-current-profile-v1/summary-decode-measured.json`。
现有 M=4 调参中，两类 NVFP4 仍选择原 baseline；多数 FP8 也选择 baseline。
扩大现有 tile/预取搜索并未普遍解决小批量投影。

## 后续需要验证的方向

针对小 M、大 K 的投影，先测更细的输出列分块：它增加互不重叠输出块的并行度，
保持 K 累加顺序。再评估沿 K 分块和归约，它会改变累加顺序，需要独立数值和模型质量。
全部候选必须进入硬件调参，完整量化、计算及必要归约图参与冷/暖 AB/BA 测量。
硬件名称只作实验标识，不作算法分支。

长提示方面仍需减少重复遍历权重，并测量候选的实际 GEMM/GDN/Attention 成本。
单纯把 graph 宽度加大，会同时增加工作区和改变状态处理路径，不能跳过资源与质量验收。
本次同时发现原量化 opt-in recurrent output scratch 没有纳入 admission：
现有 48-head、D=128、64/256-row 两张图合计 7.5 MiB，应按每张实际保留图计入，
并按 value geometry 在层间共享。这项资源修复不改变 CUDA 数学，也不声称提升性能。
