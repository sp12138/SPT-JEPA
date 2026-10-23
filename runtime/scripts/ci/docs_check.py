#!/usr/bin/env python3
"""Validate the local links and navigation of the MkDocs source tree."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import markdown
import yaml

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
HEADING_ID = re.compile(r'<h[1-6][^>]*\sid="([^"]+)"')


def nav_targets(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return set().union(*(nav_targets(item) for item in value))
    if isinstance(value, dict):
        return set().union(*(nav_targets(item) for item in value.values()))
    return set()


def local_target(source: Path, raw: str) -> Path | None:
    target = raw.strip().split("#", 1)[0]
    if not target or "://" in target or target.startswith(("mailto:", "data:")):
        return None
    return (source.parent / target).resolve()


def markdown_anchors(source: Path) -> set[str]:
    rendered = markdown.Markdown(extensions=["attr_list", "toc"]).convert(source.read_text())
    return set(HEADING_ID.findall(rendered))


def check(config_path: Path) -> list[str]:
    errors: list[str] = []
    config = yaml.safe_load(config_path.read_text())
    docs = config_path.parent / config["docs_dir"]
    configured = nav_targets(config.get("nav", []))
    pages = {path.relative_to(docs).as_posix() for path in docs.rglob("*.md")}

    for missing in sorted(configured - pages):
        errors.append(f"{config_path.name}: nav target does not exist: {missing}")
    for orphan in sorted(pages - configured):
        errors.append(f"{config_path.name}: markdown page is absent from nav: {orphan}")

    anchors: dict[Path, set[str]] = {}
    for source in sorted(docs.rglob("*.md")):
        for match in LINK.finditer(source.read_text()):
            raw = match.group(1).strip()
            target = local_target(source, raw)
            if target is not None and not target.exists():
                errors.append(f"{source.relative_to(ROOT)}: local target does not exist: {raw}")
                continue

            path_part, separator, fragment = raw.partition("#")
            if not separator or not fragment or "://" in raw:
                continue
            anchor_target = source if not path_part else target
            if anchor_target is None or anchor_target.suffix.lower() != ".md":
                continue
            available = anchors.setdefault(anchor_target, markdown_anchors(anchor_target))
            if unquote(fragment) not in available:
                errors.append(f"{source.relative_to(ROOT)}: local anchor does not exist: {raw}")

    if not errors:
        print(f"docs check ok: {docs.name}, {len(pages)} pages, all navigated and locally linked")
    return errors


def main() -> int:
    errors = check(ROOT / "mkdocs-next.yml")
    if errors:
        print("docs check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
