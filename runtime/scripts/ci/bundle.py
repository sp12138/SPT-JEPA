"""Archive one source revision and the metadata needed by checks without Git history."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def _archive_submodules(root, checkout, source, paths):
    revisions = {}
    for name in paths:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError(f"submodule must be a repository-relative path: {name!r}")
        entry = git(root, "ls-tree", source, "--", name).decode().strip()
        fields, _, recorded_path = entry.partition("\t")
        if recorded_path != name or not fields.startswith("160000 commit "):
            raise ValueError(f"{name!r} is not a submodule at source revision {source}")
        revision = fields.split()[2]
        repository = (Path(root) / name).resolve()
        if (
            not repository.is_dir()
            or Path(git(repository, "rev-parse", "--show-toplevel").decode().strip()).resolve()
            != repository
        ):
            raise ValueError(
                f"initialize {name} before bundling: git submodule update --init --checkout {name}"
            )
        # Archive the source commit's gitlink, never the worktree or its current HEAD.
        destination = checkout / name
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(git(repository, "archive", revision))) as archive:
            archive.extractall(destination, filter="data")
        revisions[name] = revision
    return revisions


def prepare(root, output, *, source, target=None, mr_title=None, submodules=()):
    source = git(root, "rev-parse", "--verify", f"{source}^{{commit}}").decode().strip()
    metadata = {"mode": "mr" if target else "publish", "source_sha": source}
    changed = b""
    if target:
        target = git(root, "rev-parse", "--verify", f"{target}^{{commit}}").decode().strip()
        base = git(root, "merge-base", source, target).decode().strip()
        commits = []
        for sha in git(root, "rev-list", "--reverse", f"{target}..{source}").decode().splitlines():
            subject = git(root, "show", "-s", "--format=%s", sha).decode().rstrip("\n")
            commits.append({"sha": sha, "subject": subject})
        metadata.update(target_sha=target, merge_base=base, mr_title=mr_title, commits=commits)
        if not mr_title or not commits:
            raise ValueError("MR title and source-only commits are required")
        changed = git(
            root, "diff", "--name-only", "-z", "--diff-filter=ACMR", base, source, "--", "*.py"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkout = output / "vforge"
    checkout.mkdir()
    with tarfile.open(fileobj=io.BytesIO(git(root, "archive", source))) as archive:
        archive.extractall(checkout, filter="data")
    metadata["submodules"] = _archive_submodules(root, checkout, source, submodules)
    (checkout / ".ci").mkdir(exist_ok=True)
    (checkout / ".ci/metadata.json").write_text(json.dumps(metadata, ensure_ascii=False))
    (checkout / ".ci/lint-files").write_bytes(changed)
    return metadata
