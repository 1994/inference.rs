"""External comparison: replay exactly the Rust suite's tokenized requests."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from check_limits import limits


def main():
    limits()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mtp", type=int, default=0)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    from vllm import LLM, SamplingParams

    configuration = {
        "model": report["model"],
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 512,
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
        max_tokens=report["max_new_tokens"],
        stop_token_ids=parameters["eos_tokens"],
    )
    prompts = [{"prompt_token_ids": row["input_tokens"]} for row in report["results"]]
    llm.generate([prompts[0]], sampling, use_tqdm=False)
    results = []
    measured_start_unix = time.time()
    suite_started = time.perf_counter()
    for row, prompt in zip(report["results"], prompts, strict=False):
        started = time.perf_counter()
        output = llm.generate([prompt], sampling, use_tqdm=False)[0].outputs[0]
        results.append(
            {
                "id": row["id"],
                "input_tokens": prompt["prompt_token_ids"],
                "wall_seconds": time.perf_counter() - started,
                "token_ids": list(output.token_ids),
                "text": output.text,
                "finish_reason": output.finish_reason,
            }
        )
    elapsed = time.perf_counter() - suite_started
    args.output.write_text(
        json.dumps(
            {
                "framework": "vllm",
                "version": importlib.metadata.version("vllm"),
                "torch_version": importlib.metadata.version("torch"),
                "configuration": configuration,
                "model": report["model"],
                "dataset": report["dataset"],
                "generation": report["generation"],
                "max_new_tokens": report["max_new_tokens"],
                "mtp_depth": args.mtp,
                "concurrency": 1,
                "warmup_requests": 1,
                "load_seconds": load_seconds,
                "wall_seconds": elapsed,
                "measured_start_unix": measured_start_unix,
                "results": results,
                "completed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
