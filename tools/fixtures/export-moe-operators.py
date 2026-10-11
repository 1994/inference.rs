"""Generate independent PyTorch goldens for selected-expert FP8 projection contracts."""

import argparse
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
experts, rows, columns, active = 3, 8, 256, 2
weights = (
    (torch.arange(experts * rows * columns).reshape(experts, rows, columns) % 17) - 8
).float()
packed = weights.to(torch.float8_e4m3fn)
scales_channel = (torch.arange(experts * rows).reshape(experts, rows, 1) % 7 + 1).float() / 64
scales_block = (torch.arange(experts * rows * 2).reshape(experts, rows, 2) % 11 + 1).float() / 64
routes = [2.0, 0.0, 0.625, 0.375]
cases = []
for routed in [False, True]:
    lanes = active if routed else 1
    x = ((torch.arange(lanes * columns).reshape(lanes, columns) % 53) - 26).float() / 19
    for channel in [False, True]:
        scaling = scales_channel if channel else scales_block
        for quantized in [False, True]:
            expected = []
            for rank in range(active):
                expert = int(routes[rank])
                source = x[rank if routed else 0]
                result = torch.zeros(rows)
                for block in range(2):
                    a = source[block * 128 : (block + 1) * 128]
                    scale = torch.tensor(1.0)
                    if quantized:
                        maximum = source.abs().max() if channel else a.abs().max()
                        scale = (maximum / 448).clamp(min=1e-12)
                        a = (a / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                    b = packed[expert, :, block * 128 : (block + 1) * 128].float()
                    result += torch.mv(b, a) * scaling[expert, :, 0 if channel else block] * scale
                expected.append(result)
            cases.append(
                {
                    "routed": routed,
                    "channel": channel,
                    "quantized": quantized,
                    "input": x.flatten().tolist(),
                    "scales": scaling.flatten().tolist(),
                    "expected": torch.stack(expected).flatten().tolist(),
                }
            )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "producer": "independent PyTorch selected-expert FP8",
            "torch": torch.__version__,
            "generator": "tools/fixtures/export-moe-operators.py",
            "experts": experts,
            "rows": rows,
            "columns": columns,
            "active": active,
            "routes": routes,
            "weights": packed.view(torch.uint8).flatten().tolist(),
            "cases": cases,
        }
    )
    + "\n"
)
print(json.dumps({"output": str(args.output), "cases": len(cases)}))
