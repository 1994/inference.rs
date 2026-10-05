"""Check lint inheritance, exceptions, and deployment feature isolation."""

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def dependency_tables(manifest):
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        yield manifest.get(section, {})
    for target in manifest.get("target", {}).values():
        yield from dependency_tables(target)


def check_manifests():
    config = tomllib.loads((ROOT / "clippy.toml").read_text())
    require(
        config.get("too-many-lines-threshold", 100) <= 100,
        "Function length limit must stay at 100 or below",
    )
    require(
        config.get("cognitive-complexity-threshold", 25) <= 25,
        "Cognitive complexity limit must stay at 25 or below",
    )
    benchmark = tomllib.loads((ROOT / "tools/bench/cpu/Cargo.toml").read_text())
    require(
        benchmark["lints"] == WORKSPACE["lints"],
        "CPU allocator harness must keep all strict lint gates",
    )
    inherited = set()
    for member in WORKSPACE["members"]:
        manifest = tomllib.loads((ROOT / member / "Cargo.toml").read_text())
        require(manifest.get("lints") == {"workspace": True}, f"{member}: inherit workspace lints")
        require(
            "test-backends" not in manifest.get("features", {}).get("default", []),
            f"{member}: CPU test features must be opt-in",
        )
        for table in dependency_tables(manifest):
            inherited.update(
                name
                for name, value in table.items()
                if isinstance(value, dict) and value.get("workspace")
            )
    unused = set(WORKSPACE["dependencies"]) - inherited
    require(not unused, f"Unused workspace dependencies: {sorted(unused)}")
    for group in ("all", "pedantic", "nursery"):
        require(WORKSPACE["lints"]["clippy"][group]["level"] == "deny", f"Keep {group} denied")
    for lint in (
        "unwrap_used",
        "expect_used",
        "panic",
        "todo",
        "unimplemented",
        "dbg_macro",
        "undocumented_unsafe_blocks",
        "allow_attributes_without_reason",
        "mem_forget",
        "multiple_unsafe_ops_per_block",
        "unused_result_ok",
        "get_unwrap",
        "rc_mutex",
        "too_many_lines",
        "cognitive_complexity",
    ):
        require(WORKSPACE["lints"]["clippy"][lint] == "deny", f"Keep {lint} denied")
    require(WORKSPACE["lints"]["rust"]["warnings"] == "deny", "Keep Rust warnings denied")
    require(WORKSPACE["lints"]["rust"]["unsafe_code"] == "deny", "Keep unsafe code denied")
    require(
        WORKSPACE["lints"]["rust"]["unfulfilled_lint_expectations"] == "deny",
        "Expired lint expectations must fail the build",
    )


def check_exceptions():
    pattern = re.compile(r"#!?\[(.*?)\]", re.DOTALL)
    forbidden = {
        "warnings",
        "clippy::all",
        "clippy::pedantic",
        "clippy::nursery",
        "clippy::too_many_lines",
        "clippy::cognitive_complexity",
    }
    paths = [*(ROOT / "crates").rglob("*.rs"), *(ROOT / "tools/bench/cpu/src").rglob("*.rs")]
    for path in paths:
        for attribute in pattern.finditer(path.read_text()):
            text = attribute[1]
            match = re.search(r"\b(?:allow|expect)\s*\((.*?)\)", text, re.DOTALL)
            if match is None:
                continue
            reason = re.search(r'reason\s*=\s*"([^"]+)"', text)
            require(reason is not None and len(reason[1]) >= 20, f"{path}: explain lint exceptions")
            lints = {part.strip() for part in match[1].split(",") if "reason" not in part}
            require(not lints & forbidden, f"{path}: strict lint checks cannot be waived")
            if "unsafe_code" in lints:
                require(
                    path
                    in (
                        ROOT / "crates/backend/metal/src/device.rs",
                        ROOT / "crates/backend/cuda/src/device.rs",
                        ROOT / "crates/foundation/core/src/placement/mod.rs",
                        ROOT / "tools/bench/cpu/src/main.rs",
                    ),
                    f"{path}: unsafe code belongs only at audited device or measurement boundaries",
                )
    deny = tomllib.loads((ROOT / "deny.toml").read_text())
    for exception in deny["advisories"].get("ignore", []) + deny["bans"].get("skip", []):
        require(
            isinstance(exception, dict) and exception.get("reason"), "Explain dependency exceptions"
        )


def check_production_dependencies():
    tree = subprocess.run(
        [
            "cargo",
            "tree",
            "--locked",
            "-p",
            "infer-cli",
            "--no-default-features",
            "--edges",
            "normal",
            "--prefix",
            "none",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(
        not any(name in tree for name in ("infer-backend-host", "infer-backend-reference")),
        "Production CLI must not depend on CPU testing executors",
    )


if __name__ == "__main__":
    check_manifests()
    check_exceptions()
    check_production_dependencies()
    print("Lint inheritance, exception policy, and GPU-only dependency checks passed")
