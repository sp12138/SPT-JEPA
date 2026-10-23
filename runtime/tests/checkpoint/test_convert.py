import sys

import pytest

from vforge.training.checkpoint import CheckpointManifest
from vforge.training.checkpoint.manifest import write_manifest
from vforge.training.checkpoint.storage import (
    convert_dcp_weights_to_pt,
    estimate_dcp_conversion,
)


def test_convert_reports_missing_format_utils(tmp_path, monkeypatch):
    source = tmp_path / "00001.dcp"
    source.mkdir()
    write_manifest(
        str(source),
        CheckpointManifest.create(state="weights", completed_steps=1),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.checkpoint.format_utils",
        None,
    )

    with pytest.raises(ImportError, match="dcp_to_torch_save"):
        convert_dcp_weights_to_pt(str(source), str(tmp_path / "weights.pt"))


def test_conversion_memory_dry_run_does_not_load_tensors(tmp_path):
    source = tmp_path / "00001.dcp"
    weights = source / "weights"
    weights.mkdir(parents=True)
    (weights / "__0_0.distcp").write_bytes(b"x" * 1024)
    write_manifest(
        str(source),
        CheckpointManifest.create(state="weights", completed_steps=1),
    )

    estimate = estimate_dcp_conversion(str(source))

    assert estimate["estimated_peak_bytes"] == 2560
    assert "available_bytes" in estimate
