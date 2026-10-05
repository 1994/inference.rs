"""Keep crate ownership, API facades, dependency direction, and local links coherent."""

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"foundation", "backend", "model", "engine", "diagnostics", "service", "testing"}
SERVICE = {"infer-frontdoor", "infer-agent", "infer-cli"}
BACKENDS = {"infer-backend-metal", "infer-backend-cuda", "infer-backend-host", "infer-backend-reference"}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def dependency_tables(manifest):
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        yield manifest.get(section, {})
    for target in manifest.get("target", {}).values():
        yield from dependency_tables(target)


def check_crates():
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]
    members = set(workspace["members"])
    found = {str(path.parent.relative_to(ROOT)) for path in (ROOT / "crates").rglob("Cargo.toml")}
    require(found == members, f"Unregistered or missing crates: {sorted(found ^ members)}")
    for member in sorted(members):
        directory = ROOT / member
        parts = Path(member).parts
        require(len(parts) >= 3 and parts[1] in GROUPS, f"{member}: choose a responsibility group")
        require(
            len(parts) == (4 if parts[1] == "testing" else 3),
            f"{member}: use group/crate, or testing/cpu/crate",
        )
        if parts[1] == "testing":
            require(parts[2] == "cpu", f"{member}: CPU reference backends belong in testing/cpu")
        manifest = tomllib.loads((directory / "Cargo.toml").read_text())
        dependencies = set()
        for table in dependency_tables(manifest):
            dependencies.update(table)
            for name, value in table.items():
                if isinstance(value, dict) and "path" in value:
                    target = directory / value["path"] / "Cargo.toml"
                    require(target.is_file(), f"{member}: broken path dependency {name}")
        if parts[1] != "service":
            require(
                not dependencies & SERVICE, f"{member}: implementation must not depend on services"
            )
        if parts[1] == "engine":
            require(
                not set(manifest.get("dependencies", {})) & BACKENDS,
                f"{member}: depend on backend contracts, not concrete executors",
            )
        facade = directory / "src/lib.rs"
        if facade.is_file():
            source = facade.read_text()
            require(len(source.splitlines()) <= 100, f"{member}: keep lib.rs a small API facade")
            require(
                not re.search(
                    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s|^impl\b", source, re.M
                ),
                f"{member}: place implementation in responsibility modules",
            )
    for name, value in workspace["dependencies"].items():
        if isinstance(value, dict) and "path" in value:
            require(
                (ROOT / value["path"] / "Cargo.toml").is_file(), f"Broken workspace path: {name}"
            )
    print(f"Layout: {len(members)} grouped crates, thin facades and dependency boundaries passed")


def check_links():
    paths = [
        ROOT / "README.md",
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "crates").rglob("*.md"),
        *(ROOT / "tools").rglob("*.md"),
        *(ROOT / "examples").rglob("*.md"),
    ]
    for path in paths:
        # Generated validation artifacts may be absent in a fresh checkout.
        for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            link = unquote(match[1].split("#", 1)[0].strip("<>"))
            if not link or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link):
                continue
            target = (path.parent / link).resolve()
            if target.is_relative_to(ROOT / "artifacts"):
                continue
            require(target.exists(), f"{path.relative_to(ROOT)}: broken local link {link}")
    print("Documentation: local navigation links passed")


def check_sources():
    for path in (ROOT / "crates").rglob("*.rs"):
        for source in re.findall(r'file:\s*"(crates/[^"\n]+)"', path.read_text()):
            require(
                (ROOT / source).is_file(), f"{path.relative_to(ROOT)}: stale source map {source}"
            )
        for fixture in re.findall(
            r'env!\("CARGO_MANIFEST_DIR"\)[^;]{0,80}?\.join\("([^"\n]+)"\)',
            path.read_text(),
        ):
            directory = next(parent for parent in path.parents if (parent / "Cargo.toml").is_file())
            require(
                (directory / fixture).exists(),
                f"{path.relative_to(ROOT)}: broken manifest-relative fixture {fixture}",
            )
    print("Observability and fixtures: source paths passed")


if __name__ == "__main__":
    check_crates()
    check_links()
    check_sources()
