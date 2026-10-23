#!/usr/bin/env python3
"""Write or verify the checksum manifest for a vendored source tree.

Used by scripts/vendor_mcore.sh to enforce that vforge/mcore stays byte-identical
to what the vendoring script produced. Kept standalone rather than inlined in the
shell script so it can be exercised directly against a throwaway tree.

    vendor_manifest.py write  <tree> <manifest>
    vendor_manifest.py verify <tree> <manifest>

verify exits 0 when the tree matches, 1 on any difference, 2 when the manifest is
missing. Differences are reported on stderr, one file per line.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

# VENDOR.md is ours, and a manifest cannot meaningfully contain its own hash.
SKIP = frozenset({"VENDOR.md", "CHECKSUMS.sha256"})


def _digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(tree: pathlib.Path) -> dict[str, str]:
    """Hash every file in the tree, keyed by POSIX-style relative path."""
    found: dict[str, str] = {}
    for path in tree.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(tree).as_posix()
        if rel not in SKIP:
            found[rel] = _digest(path)
    return found


def read_manifest(manifest: pathlib.Path) -> dict[str, str]:
    recorded: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            recorded[rel] = digest
    return recorded


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in {"write", "verify"}:
        print(__doc__, file=sys.stderr)
        return 2

    mode, tree, manifest = argv[1], pathlib.Path(argv[2]), pathlib.Path(argv[3])
    current = scan(tree)

    if mode == "write":
        manifest.write_text("".join(f"{current[rel]}  {rel}\n" for rel in sorted(current)))
        print(f"{len(current)} files")
        return 0

    if not manifest.exists():
        print(f"missing manifest: {manifest}", file=sys.stderr)
        return 2

    recorded = read_manifest(manifest)
    modified = sorted(
        rel for rel in current.keys() & recorded.keys() if current[rel] != recorded[rel]
    )
    added = sorted(current.keys() - recorded.keys())
    removed = sorted(recorded.keys() - current.keys())

    if not (modified or added or removed):
        print(f"{len(current)} files match")
        return 0

    for rel in modified:
        print(f"  modified: {rel}", file=sys.stderr)
    for rel in added:
        print(f"  added:    {rel}", file=sys.stderr)
    for rel in removed:
        print(f"  removed:  {rel}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
