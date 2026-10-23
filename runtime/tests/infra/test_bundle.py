"""A model bundle must contain the source gitlink, never a local dirty checkout."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "vforge_ci_bundle", Path(__file__).resolve().parents[2] / "scripts/ci/bundle.py"
)
bundle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bundle)


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Bundle Test",
            "-c",
            "user.email=bundle-test@example.invalid",
            *args,
        ],
        text=True,
    ).strip()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    child = root / "private_model"
    child.mkdir()
    git(child, "init", "-q")
    (child / "model.py").write_text("VERSION = 1\n")
    git(child, "add", "model.py")
    git(child, "commit", "-qm", "first")
    pinned = git(child, "rev-parse", "HEAD")
    git(root, "update-index", "--add", "--cacheinfo", f"160000,{pinned},private_model")
    git(root, "commit", "-qm", "pin model")
    return root, child, pinned


def test_bundle_uses_pinned_revision_and_records_provenance(source, tmp_path):
    root, child, pinned = source
    (child / "model.py").write_text("VERSION = 2\n")
    git(child, "commit", "-qam", "second")
    (child / "local_only.py").write_text("DO_NOT_SHIP = True\n")
    output = tmp_path / "bundle"
    metadata = bundle.prepare(root, output, source="HEAD", submodules=["private_model"])
    assert (output / "vforge/private_model/model.py").read_text() == "VERSION = 1\n"
    assert not (output / "vforge/private_model/local_only.py").exists()
    assert metadata["submodules"] == {"private_model": pinned}
    assert json.loads((output / "vforge/.ci/metadata.json").read_text()) == metadata


def test_base_bundle_does_not_require_private_checkout(source, tmp_path):
    root, child, _ = source
    child.rename(tmp_path / "saved_model")
    metadata = bundle.prepare(root, tmp_path / "bundle", source="HEAD")
    assert metadata["submodules"] == {}


def test_requested_missing_checkout_fails(source, tmp_path):
    root, child, _ = source
    child.rename(tmp_path / "saved_model")
    child.mkdir()
    with pytest.raises(ValueError, match="initialize private_model"):
        bundle.prepare(root, tmp_path / "bundle", source="HEAD", submodules=["private_model"])


@pytest.mark.parametrize("path", ["../outside", "/absolute", "unknown"])
def test_bundle_rejects_non_gitlink_paths(source, tmp_path, path):
    root, _, _ = source
    with pytest.raises(ValueError):
        bundle.prepare(root, tmp_path / "bundle", source="HEAD", submodules=[path])
