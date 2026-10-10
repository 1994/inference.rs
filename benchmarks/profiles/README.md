# 实验 profile 清单

每个 profile 是一份实验条件的单一来源，由 [experiment-checklist.py](../../tools/bench/experiment-checklist.py)
生成与校验。采集端用它验证一次运行确实符合声明，门禁要求对照的两份报告来自**同一**清单。

## 为什么需要它

[性能基线方案](../docs/plans/performance/baseline.md) 要求两边在实际工作量、质量条件与资源约束上对齐，
并且明确禁止沿用脚本里的旧默认值。清单把模型制品指纹、资源上限、cache 状态、workload 哈希、矩阵与
统计规则固定下来；运行开始前校验，缺失或不相符直接失败，而不是等到比较时才发现两边不是同一个实验。

## 生成

```sh
python3 tools/bench/serve-workloads.py --native-binary <release-bin> \
    --package /home/r/models/Qwen3.8-27B-NVFP4 --output artifacts/workloads/27b-inputs.json

python3 tools/bench/experiment-checklist.py --init --profile-id 27b-mtp2 \
    --model /home/r/models/Qwen3.8-27B-NVFP4 \
    --workload artifacts/workloads/27b-inputs.json --out benchmarks/profiles/27b-mtp2.json \
    --mtp 2 --tokens 64 --gpu-memory-utilization 0.88 \
    --max-model-len 262144 --max-num-seqs 16
```

制品指纹与 workload 哈希由工具计算，不接受手填；`--init` 的结果与签入文件受同一套结构校验。

## 使用

```sh
# 采集时校验，不相符则这一次运行失败
python3 tools/bench/serve-compare.py --engine native --checklist benchmarks/profiles/27b-mtp2.json ...

# 采集后独立复核某份报告
python3 tools/bench/experiment-checklist.py --check benchmarks/profiles/27b-mtp2.json <report.json>

# 输出两边的配置对照表
python3 tools/bench/experiment-checklist.py --compare <native.json> <vllm.json>
```

对照表逐项列出两边读数与是否一致，覆盖模型制品、上下文上限、cache、显存比例、MTP 深度、输出预算、
采样、EOS、workload 哈希与矩阵。引擎版本本来就不同，不对齐只作为信息列出。

## 已签入的清单

| 文件 | 模型 | MTP |
|---|---|---|
| [27b-mtp0.json](27b-mtp0.json) | Qwen3.8-27B-NVFP4 | 0 |
| [27b-mtp2.json](27b-mtp2.json) | Qwen3.8-27B-NVFP4 | 2 |
| [2b-mtp0.json](2b-mtp0.json) | qwen3vl-2b | 0 |

## 能力限制（未签入清单的那些档位）

方案要求"每个模型覆盖普通 decode 与当前已支持的 MTP 档位；**不支持的算法明确登记能力限制**，不能把
回退成普通 decode 的结果当作该算法验收"。据此登记：

| 模型 | 可测 MTP 档位 | 依据 |
|---|---|---|
| Qwen3.8-27B-NVFP4 | 0、2（`27b-mtp0`、`27b-mtp2`） | 配置声明 `text_config.mtp_num_hidden_layers = 1`，即带 MTP head；档位由提议深度给出 |
| qwen3vl-2b | **仅 0**（`2b-mtp0`） | 该 checkpoint 的配置**没有任何 `mtp_*` 键**，即不附带 MTP head；任何正深度都只会落到普通 decode，因此不签发该档位 |

这条限制现在**由工具强制**：`experiment-checklist.py` 在读取清单时，若 `workload.mtp_depth > 0`
而模型配置未声明 MTP head，直接拒绝该清单（模型路径不可读时不作判断，因此离线环境仍可校验结构）。
测试覆盖：声明了 head 的模型通过、没有 head 的正深度被拒、深度 0 无需 head、配置不可读时不判定，以及
已签入的三份清单彼此一致。

清单里的资源与 workload 条件对应 `artifacts/workloads/` 下同名输入；重新生成输入会改变
`inputs_sha256`，必须重新签发清单，否则采集端会拒绝运行。

`numeric` 与 `quality` 字段声明本 profile 的数值与质量归属；质量由各自的独立门禁判定，本清单只记录
引用，不代替它。
