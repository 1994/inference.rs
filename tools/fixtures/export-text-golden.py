"""Export C0 reference from local official tokenizer/template assets."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--package", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--revision", required=True)
args = p.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.package, local_files_only=True)
messages = [
    {"role": "system", "content": "你是一个助手。"},
    {"role": "user", "content": "解释 Rust 的 ownership。"},
]
cases = []
for thinking in [False, True]:
    options = {"add_generation_prompt": True, "enable_thinking": thinking}
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, **options)
    tokens = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    cases.append(
        {"messages": messages, "enable_thinking": thinking, "rendered": rendered, "tokens": tokens}
    )
args.output.write_text(
    json.dumps({"revision": args.revision, "cases": cases}, ensure_ascii=False, indent=2) + "\n"
)
