#!/usr/bin/env python3
"""Declare one experiment profile once, then verify every run and print the comparison table.

The baseline plan requires the two engines to be aligned on real work, quality conditions and
resource constraints, and it forbids reusing a script's old defaults: the checklist is the single
source for a profile, and a run has to prove it matched it.

A checklist is a JSON document with the following shape; every field is required because a
missing condition is the failure mode this exists to prevent:

    {
      "schema": 1,
      "profile_id": "27b-mtp2-nvfp4",
      "model": {"path": "/models/...", "files": {"config.json": "<sha256>", ...}},
      "numeric": {"weights": "...", "activation": "...", "kv": "..."},
      "quality": {"gate": "<reference to the quality check>", "required": false},
      "resources": {"gpu_memory_utilization": 0.88, "max_model_len": 262144,
                    "max_num_seqs": 16},
      "cache": {"prefix_cache": true},
      "workload": {"inputs_sha256": "...", "max_new_tokens": 64, "temperature": 0,
                   "mtp_depth": 2, "eos_tokens": [151645],
                   "matrix": {"short": {"concurrency": 1, "repeats": 3}, ...}},
      "engines": {"native": {}, "vllm": {"version": "0.31.0"}},
      "measurement": {"repeats": 3, "max_ratio": 1.1, "require_identical_tokens": false}
    }

Usage:
    python3 tools/bench/experiment-checklist.py --check profile.json reports/native.json
    python3 tools/bench/experiment-checklist.py --compare reports/native.json reports/vllm.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Every condition a profile declares. A missing one is reported rather than defaulted.
REQUIRED_SECTIONS = (
    "profile_id",
    "model",
    "numeric",
    "quality",
    "resources",
    "cache",
    "workload",
    "engines",
    "measurement",
)


def load(path):
    """Read and structurally validate a checklist."""
    return load_document(json.loads(Path(path).read_text()))


def load_document(checklist):
    """Structurally validate a checklist document."""
    if checklist.get("schema") != 1:
        raise ValueError("checklist schema must be 1")
    missing = [section for section in REQUIRED_SECTIONS if section not in checklist]
    if missing:
        raise ValueError(f"checklist is missing sections: {', '.join(missing)}")
    for field in ("path", "files"):
        if field not in checklist["model"]:
            raise ValueError(f"checklist model is missing {field}")
    for field in ("gpu_memory_utilization", "max_model_len", "max_num_seqs"):
        if field not in checklist["resources"]:
            raise ValueError(f"checklist resources are missing {field}")
    for field in ("prefix_cache",):
        if field not in checklist["cache"]:
            raise ValueError(f"checklist cache is missing {field}")
    for field in (
        "inputs_sha256",
        "max_new_tokens",
        "temperature",
        "mtp_depth",
        "eos_tokens",
        "matrix",
    ):
        if field not in checklist["workload"]:
            raise ValueError(f"checklist workload is missing {field}")
    return checklist


def _mismatch(mismatches, item, expected, actual):
    if expected != actual:
        mismatches.append(f"{item}: checklist declares {expected!r}, run has {actual!r}")


def check_run(checklist, report):
    """Return the ways a measured run failed to match its checklist.

    The report's recorded values are the ones the harness observed, including the effective
    configuration read back from the server, so this compares declared conditions against what
    actually happened rather than against the command line that was passed.
    """
    mismatches = []
    identity = report.get("identity") or {}
    engine = identity.get("engine")
    _mismatch(mismatches, "model path", checklist["model"]["path"], report.get("model"))
    files = (identity.get("model") or {}).get("files") or {}
    for name, digest in checklist["model"]["files"].items():
        _mismatch(mismatches, f"model artifact {name}", digest, files.get(name))
    _mismatch(
        mismatches,
        "gpu_memory_utilization",
        checklist["resources"]["gpu_memory_utilization"],
        report.get("gpu_memory_utilization"),
    )
    _mismatch(
        mismatches,
        "prefix cache",
        checklist["cache"]["prefix_cache"],
        report.get("prefix_cache_enabled"),
    )
    for item, field in (
        ("max_new_tokens", "max_new_tokens"),
        ("temperature", "temperature"),
        ("mtp_depth", "mtp_depth"),
        ("eos_tokens", "eos_tokens"),
    ):
        _mismatch(mismatches, item, checklist["workload"][field], report.get(field))
    _mismatch(
        mismatches,
        "workload inputs",
        checklist["workload"]["inputs_sha256"],
        report.get("inputs_sha256"),
    )
    _mismatch(mismatches, "matrix", checklist["workload"]["matrix"], report.get("matrix"))
    _mismatch(
        mismatches,
        "measured repeats",
        checklist["measurement"]["repeats"],
        next(iter(report.get("matrix", {}).values()), {}).get("repeats"),
    )
    # The context ceiling is read back from the server on native and declared on the reference
    # side; both have to match the profile, not just each other.
    limits = identity.get("limits") or {}
    if engine == "native":
        _mismatch(
            mismatches,
            "effective max model len",
            checklist["resources"]["max_model_len"],
            limits.get("effective_max_model_len"),
        )
    else:
        _mismatch(
            mismatches,
            "max_model_len",
            checklist["resources"]["max_model_len"],
            limits.get("max_model_len"),
        )
        _mismatch(
            mismatches,
            "max_num_seqs",
            checklist["resources"]["max_num_seqs"],
            limits.get("max_num_seqs"),
        )
    expected_version = (checklist["engines"].get(engine) or {}).get("version")
    if expected_version is not None:
        _mismatch(
            mismatches,
            f"{engine} version",
            expected_version,
            identity.get("engine_version"),
        )
    return mismatches


def comparison_table(reports):
    """The per-item native/reference configuration table the baseline plan asks for."""
    by_engine = {report["identity"]["engine"]: report for report in reports}
    if len(by_engine) != len(reports):
        raise ValueError("the comparison table needs one report per engine")
    items = {
        "model": lambda r: r.get("model"),
        "model artifacts": lambda r: sorted(
            ((r.get("identity") or {}).get("model") or {}).get("files", {}).items()
        ),
        "context ceiling": lambda r: (
            ((r.get("identity") or {}).get("limits") or {}).get("effective_max_model_len")
            or ((r.get("identity") or {}).get("limits") or {}).get("max_model_len")
        ),
        "prefix cache": lambda r: r.get("prefix_cache_enabled"),
        "gpu_memory_utilization": lambda r: r.get("gpu_memory_utilization"),
        "mtp depth": lambda r: r.get("mtp_depth"),
        "output tokens": lambda r: r.get("max_new_tokens"),
        "temperature": lambda r: r.get("temperature"),
        "eos tokens": lambda r: r.get("eos_tokens"),
        "workload inputs": lambda r: r.get("inputs_sha256"),
        "matrix": lambda r: r.get("matrix"),
        "engine version": lambda r: (r.get("identity") or {}).get("engine_version"),
    }
    rows = []
    for item, extract in items.items():
        values = {engine: extract(report) for engine, report in by_engine.items()}
        rendered = {json.dumps(value, sort_keys=True) for value in values.values()}
        aligned = len(rendered) == 1
        rows.append({"item": item, "values": values, "aligned": aligned})
    return rows


def render(rows):
    engines = sorted({engine for row in rows for engine in row["values"]})
    width = max(len(row["item"]) for row in rows)
    lines = [
        f"{'item'.ljust(width)}  " + "  ".join(engine.ljust(24) for engine in engines) + "  aligned"
    ]
    for row in rows:
        cells = "  ".join(str(row["values"].get(engine, "-"))[:24].ljust(24) for engine in engines)
        lines.append(f"{row['item'].ljust(width)}  {cells}  {'yes' if row['aligned'] else 'NO'}")
    return "\n".join(lines)


def draft(arguments):
    """Build a checklist from a generated workload and the model it was tokenized for.

    The workload hash and the artifact fingerprints are computed here rather than typed in, so a
    profile cannot claim a condition nobody verified.
    """
    workload = Path(arguments.workload)
    rows = json.loads(workload.read_text())
    cases = {}
    for row in rows:
        entry = cases.setdefault(row["case"], {"concurrency": row["concurrency"], "repeats": 0})
        if entry["concurrency"] != row["concurrency"]:
            raise ValueError(f"{row['case']} declares inconsistent concurrency")
        entry["repeats"] = max(entry["repeats"], row["repeat"] + 1)
    model = Path(arguments.model)
    files = {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest()
        for name in arguments.artifacts
        if (model / name).is_file()
    }
    if not files:
        raise ValueError(f"no known artifact found in {model}")
    eos = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
    engines = {"native": {"version": None}} if arguments.engine == "native" else {}
    for engine in ("native", "vllm", "sglang"):
        if engine != arguments.engine:
            engines.setdefault(engine, {"version": None})
    engines[arguments.engine] = {"version": arguments.engine_version}
    return {
        "schema": 1,
        "profile_id": arguments.profile_id,
        "model": {"path": str(model), "files": files},
        "numeric": {
            "weights": arguments.weights,
            "activation": arguments.activation,
            "kv": arguments.kv,
        },
        "quality": {"gate": arguments.quality_gate, "required": arguments.quality_required},
        "resources": {
            "gpu_memory_utilization": arguments.gpu_memory_utilization,
            "max_model_len": arguments.max_model_len,
            "max_num_seqs": arguments.max_num_seqs,
        },
        "cache": {"prefix_cache": not arguments.no_prefix_cache},
        "workload": {
            "inputs_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "max_new_tokens": arguments.tokens,
            "temperature": arguments.temperature,
            "mtp_depth": arguments.mtp,
            "eos_tokens": eos if isinstance(eos, list) else [eos],
            "matrix": cases,
        },
        "engines": engines,
        "measurement": {
            "repeats": arguments.repeats,
            "max_ratio": arguments.max_ratio,
            "require_identical_tokens": arguments.require_identical_tokens,
        },
    }


ARTIFACTS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "chat_template.jinja",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, metavar="CHECKLIST", help="checklist to verify")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    parser.add_argument("report", type=Path, nargs="?", help="report to check")
    parser.add_argument("--init", action="store_true", help="draft a profile from a workload")
    parser.add_argument("--out", type=Path, help="where --init writes the checklist")
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--workload", type=Path, default=None)
    parser.add_argument("--engine", default="native", choices=["native", "vllm", "sglang"])
    parser.add_argument("--engine-version", default=None)
    parser.add_argument("--weights", default="unspecified")
    parser.add_argument("--activation", default="unspecified")
    parser.add_argument("--kv", default="model default")
    parser.add_argument("--quality-gate", default="not declared")
    parser.add_argument("--quality-required", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-ratio", type=float, default=1.1)
    parser.add_argument("--require-identical-tokens", action="store_true")
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()
    if args.init:
        if not (args.profile_id and args.model and args.workload and args.out):
            parser.error("--init needs --profile-id, --model, --workload and --out")
        args.artifacts = ARTIFACTS
        document = draft(args)
        load_document(document)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2) + "\n")
        print(f"{args.out}: profile {document['profile_id']}")
        return 0
    if args.compare is not None:
        reports = [json.loads(path.read_text()) for path in args.compare]
        rows = comparison_table(reports)
        print(render(rows))
        return 0 if all(row["aligned"] for row in rows) else 1
    if args.check is None or args.report is None:
        parser.error("provide --check CHECKLIST REPORT, or --compare A B")
    checklist = load(args.check)
    report = json.loads(args.report.read_text())
    mismatches = check_run(checklist, report)
    for mismatch in mismatches:
        print(mismatch, file=sys.stderr)
    if mismatches:
        print(f"{len(mismatches)} condition(s) differ from {args.check}", file=sys.stderr)
        return 1
    print(f"{args.report} matches {args.check} ({checklist['profile_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
