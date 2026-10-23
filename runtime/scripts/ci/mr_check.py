#!/usr/bin/env python3
"""Check the MR title and every source-only commit subject using one convention."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCOPES = "ci|pt|ct|ckpt|loader|rl|data|infra|train|ops|model|compat"
TYPES = "feat|fix|refactor|perf|docs|test|chore|build|ci"
PATTERN = re.compile(rf"\[({SCOPES})\] ({TYPES}):[^\S\n]+\S[^\n\r]*")


def check(metadata):
    if metadata.get("mode") != "mr" or not metadata.get("commits"):
        raise ValueError("MR metadata must contain source-only commits")
    titles = [("MR", metadata.get("mr_title", ""))]
    titles.extend((commit["sha"], commit["subject"]) for commit in metadata["commits"])
    errors = [
        f"{label}: {title!r}" for label, title in titles if not PATTERN.fullmatch(title.strip())
    ]
    if errors:
        raise ValueError("Expected [scope] type: description\n" + "\n".join(errors))
    print(f"[mr_check] PASS: MR and {len(metadata['commits'])} commits", flush=True)


if __name__ == "__main__":
    check(json.loads(Path(sys.argv[1]).read_text()))
