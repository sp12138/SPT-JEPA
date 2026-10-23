#!/usr/bin/env python3
"""Run pytest, preserving its exit status; JUnit is diagnostic output only."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    args, tests = parser.parse_known_args()
    args.junit.parent.mkdir(parents=True, exist_ok=True)
    args.junit.unlink(missing_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "--durations=10",
            "-p",
            "no:cacheprovider",
            f"--junitxml={args.junit}",
            *tests,
        ],
        env=dict(os.environ, PYTEST_ADDOPTS=""),
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
