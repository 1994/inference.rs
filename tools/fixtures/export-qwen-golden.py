"""Cold-path official Transformers golden export; never needed by the Rust runtime."""

import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
parser.add_argument(
    "--grouped", action="store_true", help="2 key / 6 value heads and partial-RoPE GQA"
)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.manual_seed(20261004)
torch.set_num_threads(1)
config = Qwen3_5TextConfig(
    vocab_size=32,
    hidden_size=8,
    intermediate_size=12,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    max_position_embeddings=64,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=2,
    linear_value_head_dim=2,
    linear_conv_kernel_dim=3,
    attn_output_gate=True,
    tie_word_embeddings=False,
    eos_token_id=31,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [1, 0, 0],
        "mrope_interleaved": True,
    },
)
config._attn_implementation = "eager"
if args.grouped:
    config.hidden_size = 16
    config.intermediate_size = 24
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 6
    config.linear_key_head_dim = 4
    config.linear_value_head_dim = 4
    config.rope_parameters["partial_rotary_factor"] = 0.25
model = Qwen3_5ForCausalLM(config).float().eval()
# Nontrivial norms, decay and gates: initialization alone leaves several parameters
# at zero and would miss offset/gating binding mistakes.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "layernorm" in name or name == "model.norm.weight" or "_norm.weight" in name:
            p.copy_(torch.linspace(-0.15, 0.2, p.numel()).view_as(p))
        elif "linear_attn.norm.weight" in name:
            p.copy_(torch.linspace(0.8, 1.2, p.numel()).view_as(p))
        elif name.endswith("A_log"):
            p.copy_(torch.linspace(-0.3, 0.3, p.numel()))
        elif name.endswith("dt_bias"):
            p.copy_(torch.linspace(-0.4, 0.1, p.numel()))
save_file(
    {k: v.contiguous() for k, v in model.state_dict().items()},
    str(args.output / "model.safetensors"),
)
(args.output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
tokens = [1, 2, 3, 5, 8, 13]
prefixes = []
layer_outputs = {}
hooks = [
    layer.register_forward_hook(
        lambda module, inputs, output, index=index: layer_outputs.__setitem__(
            index, (output[0] if isinstance(output, tuple) else output).detach().clone()
        )
    )
    for index, layer in enumerate(model.model.layers)
]
with torch.no_grad():
    for n in range(1, len(tokens) + 1):
        out = model(torch.tensor([tokens[:n]]), use_cache=False, output_hidden_states=True)
        prefixes.append(
            {
                "tokens": tokens[:n],
                "logits": out.logits[0, -1].tolist(),
                "hidden": out.hidden_states[-1][0].tolist(),
                "layers": [layer_outputs[index][0].tolist() for index in range(len(hooks))],
            }
        )
    for hook in hooks:
        hook.remove()
    generated = model.generate(
        torch.tensor([tokens]),
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        use_cache=True,
    )[0].tolist()[len(tokens) :]
golden = {
    "producer": "Hugging Face Transformers Qwen3_5ForCausalLM, eager F32 CPU",
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "seed": 20261004,
    "prefixes": prefixes,
    "greedy_tokens": generated,
}
(args.output / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")

vocab = {f"token{i}": i for i in range(32)}
for i, text in enumerate(["[UNK]", "hello", "world", "system", "user", "assistant"]):
    del vocab[f"token{i}"]
    vocab[text] = i
tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.save(str(args.output / "tokenizer.json"))
template = (
    "{% for message in messages %}{{ message['role'] }} "
    "{{ message['content'].strip() }} {% endfor %}"
    "{% if add_generation_prompt %}assistant{% endif %}"
)
(args.output / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}) + "\n")
torch.manual_seed(20261005)
readouts = {
    "embedding.weight": torch.randn(4, config.hidden_size) * 0.1,
    "rank.weight": torch.randn(1, config.hidden_size) * 0.1,
    "rank.bias": torch.tensor([0.05]),
    "decision.weight": torch.randn(32, config.hidden_size) * 0.1,
    "decision.bias": torch.linspace(-0.1, 0.1, 32),
}
save_file(readouts, str(args.output / "readouts.safetensors"))
with torch.no_grad():
    h = model(
        torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False, output_hidden_states=True
    ).hidden_states[-1][0]
    embedding = torch.nn.functional.linear(h, readouts["embedding.weight"]).mean(dim=0)
    embedding = torch.nn.functional.normalize(embedding, dim=0)
    decisions = torch.nn.functional.linear(
        h[-1], readouts["decision.weight"], readouts["decision.bias"]
    )
    rank_scores = []
    for tokens in [[1, 2, 3, 4, 5], [1, 2, 6, 7], [1, 2, 8]]:
        h_pair = model(
            torch.tensor([tokens]), use_cache=False, output_hidden_states=True
        ).hidden_states[-1][0, -1]
        rank_scores.append(
            torch.nn.functional.linear(
                h_pair, readouts["rank.weight"], readouts["rank.bias"]
            ).item()
        )
(args.output / "readout-golden.json").write_text(
    json.dumps(
        {
            "embedding": embedding.tolist(),
            "rank_scores": rank_scores,
            "decision_logits": decisions.tolist(),
            "decision_probabilities": [
                torch.softmax(decisions[[0, 1]], dim=0).tolist(),
                torch.softmax(decisions[[2, 3, 4]], dim=0).tolist(),
                torch.softmax(decisions[[1, 2, 3]], dim=0).tolist(),
                [],
            ],
            "continuous_expected": torch.sigmoid(decisions[0]).item(),
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps({"output": str(args.output), "prefixes": len(prefixes), "greedy_tokens": generated})
)
