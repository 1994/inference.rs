"""Create a reproducible real-data workload; pyarrow is only a benchmark dependency."""

import argparse
import hashlib
import json
import random
from pathlib import Path

SOURCES = [
    {
        "dataset": "AI-ModelScope/gsm8k",
        "split": "main/test",
        "revision": "2680164407bc7fd6c04d1ad609399a6e62a3b21e",
        "file": "main/test-00000-of-00001.parquet",
        "sha256": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
    },
    {
        "dataset": "AI-ModelScope/sharegpt_gpt4",
        "split": "first-human-turn",
        "revision": "c5ef4f29d7a927846af2c36013755c1a04e16c41",
        "file": "sharegpt_zh_38K_format.jsonl",
        "sha256": "a87dbcfa92ba53dfccaad97270189c377ca670a8992d632ab02ec073b1b615b7",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", required=True, type=Path)
    parser.add_argument("--sharegpt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-dataset", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()
    if args.per_dataset < 1 or args.max_chars < 1:
        parser.error("counts must be positive")
    from pyarrow import parquet

    for source, path in zip(SOURCES, [args.gsm8k, args.sharegpt], strict=False):
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        source["url"] = (
            f"https://modelscope.cn/datasets/{source['dataset']}"
            f"/resolve/{source['revision']}/{source['file']}"
        )
    groups = [
        parquet.read_table(args.gsm8k).to_pylist(),
        [json.loads(line) for line in args.sharegpt.read_text().split("\n") if line.strip()],
    ]
    samples = []
    for number, rows in enumerate(groups):
        eligible = []
        for index, row in enumerate(rows):
            if number == 0:
                prompt, answer = row["question"], row["answer"]
            else:
                turns = row.get("conversations", [])
                if not turns or turns[0].get("from") != "human":
                    continue
                prompt, answer = turns[0]["value"], None
            if 1 <= len(prompt) <= args.max_chars:
                eligible.append(
                    {"id": f"{number}:{index}", "prompt": prompt, "reference_answer": answer}
                )
        random.Random(args.seed + number).shuffle(eligible)
        if len(eligible) < args.per_dataset:
            raise ValueError("Not enough eligible rows")
        SOURCES[number].update(total_rows=len(rows), eligible_rows=len(eligible))
        samples.extend(eligible[: args.per_dataset])
    args.output.write_text(
        json.dumps(
            {
                "schema": 1,
                "seed": args.seed,
                "sources": SOURCES,
                "samples": samples,
                "selection": (
                    f"seeded shuffle, first human turn, 1..{args.max_chars} "
                    "Unicode characters; no truncation"
                ),
                "scope": (
                    "fixed subset, not full dataset; ShareGPT references are not correctness labels"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
