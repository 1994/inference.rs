"""External vLLM reference only; the Rust inference implementation does not import this."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from check_limits import limits


def main():
    limits()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Rust report containing input_tokens")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text())
    from vllm import LLM, SamplingParams

    configuration = {
        "model": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": False,
        "generation_config": "vllm",
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    if args.mtp:
        configuration["speculative_config"] = {"method": "mtp", "num_speculative_tokens": args.mtp}
    started = time.perf_counter()
    llm = LLM(**configuration)
    load_seconds = time.perf_counter() - started
    parameters = report["generation"]["sampling"]
    sampling = SamplingParams(
        temperature=parameters["temperature"],
        top_k=parameters["top_k"] or -1,
        top_p=parameters["top_p"],
        min_p=parameters["min_p"],
        presence_penalty=parameters["presence_penalty"],
        repetition_penalty=parameters["repetition_penalty"],
        seed=parameters["seed"],
        max_tokens=args.tokens,
        stop_token_ids=parameters["eos_tokens"],
    )
    prompt = {"prompt_token_ids": report["input_tokens"]}
    llm.generate([prompt], sampling, use_tqdm=False)
    trials = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = llm.generate([prompt], sampling, use_tqdm=False)[0]
        elapsed = time.perf_counter() - started
        output = result.outputs[0]
        trials.append(
            {
                "wall_seconds": elapsed,
                "token_ids": list(output.token_ids),
                "text": output.text,
                "output_tokens": len(output.token_ids),
                "finish_reason": output.finish_reason,
            }
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "framework": "vllm",
                "version": importlib.metadata.version("vllm"),
                "torch_version": importlib.metadata.version("torch"),
                "configuration": configuration,
                "sampling": parameters,
                "input_tokens": report["input_tokens"],
                "load_seconds": load_seconds,
                "warmup_runs": 1,
                "trials": trials,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
