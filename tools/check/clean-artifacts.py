"""Remove generated output while keeping documented evidence and model assets."""

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
MODEL_SUFFIXES = {".safetensors", ".gguf", ".bin", ".pt", ".pth", ".onnx"}


def protected_paths(root):
    root = root.resolve()
    artifacts = root / "artifacts"
    retained = set()
    documents = [root / "README.md"]
    for directory in ("docs", "crates", "tools", "examples"):
        documents.extend((root / directory).rglob("*.md"))
    for document in documents:
        if not document.is_file():
            continue
        for link in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", document.read_text()):
            link = unquote(link.split("#", 1)[0].strip("<>"))
            if not link or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link):
                continue
            target = (document.parent / link).resolve()
            if target.is_relative_to(artifacts) and target.exists():
                retained.add(target)
    for child in artifacts.iterdir():
        if child.is_symlink():
            continue
        if (child.is_file() and child.suffix in MODEL_SUFFIXES) or (
            child.is_dir()
            and (
                (child / "tokenizer.json").is_file()
                or (child / "config.json").is_file()
                or any(child.glob("*.safetensors"))
            )
        ):
            retained.add(child)
    for report in tuple(retained):
        if report.name != "summary.json" or not report.is_file():
            continue
        for name in json.loads(report.read_text()).get("logs", {}):
            log = (report.parent / name).resolve()
            if log.is_relative_to(artifacts) and log.is_file():
                retained.add(log)
    return retained


def candidates(directory, retained):
    for child in sorted(directory.iterdir()):
        if child in retained:
            continue
        if child.is_symlink() or not any(p.is_relative_to(child) for p in retained):
            yield child
        elif child.is_dir():
            yield from candidates(child, retained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List removals without deleting")
    options = parser.parse_args()
    artifacts = ROOT / "artifacts"
    if artifacts.is_symlink():
        raise SystemExit("Refusing a symlinked artifacts directory")
    if not artifacts.exists():
        return
    retained = protected_paths(ROOT)
    removals = list(candidates(artifacts, retained))
    for entry in removals:
        print(f"{'Would remove' if options.dry_run else 'Remove'} {entry.relative_to(ROOT)}")
        if not options.dry_run:
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            else:
                shutil.rmtree(entry)
    print(f"{'Previewed' if options.dry_run else 'Removed'} {len(removals)} entries")


if __name__ == "__main__":
    main()
