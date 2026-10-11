"""Export a reduced official Qwen MoE checkpoint and independent full-prefix goldens."""
import argparse
import json
from pathlib import Path
import torch
import transformers
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3NextConfig, Qwen3NextForCausalLM
from tokenizers import Tokenizer, models, pre_tokenizers

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--family", choices=["qwen3", "next"], default="qwen3")
parser.add_argument("--production-heads", action="store_true")
parser.add_argument("--packed", action="store_true")
parser.add_argument("--no-normalize", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(20261011)
config = Qwen3MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
    moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=1, head_dim=8, num_experts=4, num_experts_per_tok=2,
    max_position_embeddings=64, norm_topk_prob=True, tie_word_embeddings=False)
config.norm_topk_prob = not args.no_normalize
model_class = Qwen3MoeForCausalLM
if args.family == "next":
    config = Qwen3NextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_conv_kernel_dim=4, full_attention_interval=2,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
        norm_topk_prob=not args.no_normalize, tie_word_embeddings=False,
        partial_rotary_factor=0.5)
    model_class = Qwen3NextForCausalLM
if args.production_heads:
    config.hidden_size = 256
    config.head_dim = 128
    config.moe_intermediate_size = 128
    config.shared_expert_intermediate_size = 128
    config.linear_key_head_dim = 128
    config.linear_value_head_dim = 128
config._attn_implementation = "eager"
model = model_class(config).float().eval()
# Store BF16 exactly, then run F32 arithmetic on those exact weight values.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(torch.linspace(0.8, 1.2, p.numel()).view_as(p))
        p.copy_(p.bfloat16().float())
weights = {name: value.bfloat16().contiguous() for name,value in model.state_dict().items()}
if args.packed:
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp.experts"
        gates, ups, downs = [], [], []
        for expert in range(config.num_experts):
            gates.append(weights.pop(f"{prefix}.{expert}.gate_proj.weight"))
            ups.append(weights.pop(f"{prefix}.{expert}.up_proj.weight"))
            downs.append(weights.pop(f"{prefix}.{expert}.down_proj.weight"))
        weights[f"{prefix}.gate_up_proj"] = torch.stack([torch.cat([gate, up]) for gate, up in zip(gates, ups)])
        weights[f"{prefix}.down_proj"] = torch.stack(downs)
save_file(weights, str(args.output / "model.safetensors"))
config.save_pretrained(args.output)
tokens = [1, 2, 3, 5, 8, 13]
prefixes = []
with torch.no_grad():
    for end in range(1, len(tokens) + 1):
        result = model(torch.tensor([tokens[:end]]), use_cache=False, output_hidden_states=True)
        prefixes.append({"tokens": tokens[:end], "logits": result.logits[0, -1].tolist(),
            "hidden": result.hidden_states[-1][0].tolist()})
golden = {"producer": f"Hugging Face {model_class.__name__}, eager F32 with exact BF16 weights",
    "torch": torch.__version__, "transformers": transformers.__version__,
    "seed": 20261011, "prefixes": prefixes, "generator": {
        "script": "tools/fixtures/export-moe-golden.py",
        "command": f"python3 tools/fixtures/export-moe-golden.py --output {args.output} --family {args.family}"\
            + (" --production-heads" if args.production_heads else "") + (" --packed" if args.packed else "") + (" --no-normalize" if args.no_normalize else "")}}
(args.output / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
tokenizer = Tokenizer(models.WordLevel({f"token{i}": i for i in range(32)}, unk_token="token0"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.save(str(args.output / "tokenizer.json"))
print(json.dumps({"package": str(args.output), "parameters": sum(p.numel() for p in model.parameters()),
    "prefixes": len(prefixes)}))
