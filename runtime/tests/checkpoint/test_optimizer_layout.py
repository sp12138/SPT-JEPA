import pytest
import torch
import torch.distributed.checkpoint as dcp

from vforge.training.checkpoint.manifest import CheckpointManifest
from vforge.training.checkpoint.optimizer_layout import (
    DEFAULT_OPTIMIZER_AUX_THRESHOLD_BYTES,
    OptimizerFileSystemWriter,
)
from vforge.training.checkpoint.storage import (
    _dcp_file_system_writer,
    _dcp_optimizer_file_system_writer,
    _dcp_save,
    save_dcp,
)


@pytest.mark.parametrize("factory", [_dcp_file_system_writer, _dcp_optimizer_file_system_writer])
def test_cpu_staged_writer_does_not_touch_cuda(tmp_path, monkeypatch, factory):
    # A CPU-only test can still detect the writer selecting its CUDA copy loader.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def no_cuda_stream(*args, **kwargs):
        raise AssertionError("writing a CPU snapshot must not access a CUDA stream")

    monkeypatch.setattr(torch.cuda, "current_stream", no_cuda_stream)
    state = {"value": torch.arange(6, dtype=torch.float32)}
    _dcp_save(state, None, storage_writer=factory(str(tmp_path), cpu_staged=True))
    restored = {"value": torch.zeros_like(state["value"])}
    dcp.load(restored, checkpoint_id=tmp_path)
    torch.testing.assert_close(restored["value"], state["value"], rtol=0, atol=0)


def test_optimizer_writer_separates_payload_and_auxiliary_items(tmp_path):
    checkpoint = tmp_path / "optimizer"
    large = torch.arange(
        DEFAULT_OPTIMIZER_AUX_THRESHOLD_BYTES // torch.float32.itemsize + 1,
        dtype=torch.float32,
    )
    small = torch.tensor(7.0)
    state = {"optimizer": {"large": large, "step": small}}

    _dcp_save(
        state,
        None,
        storage_writer=OptimizerFileSystemWriter(checkpoint),
    )

    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    locations = {
        index.fqn: storage.relative_path for index, storage in metadata.storage_data.items()
    }
    assert locations["optimizer.large"].endswith("_payload.distcp")
    assert locations["optimizer.step"].endswith("_aux.distcp")
    assert sorted(path.name for path in checkpoint.glob("*.distcp")) == [
        "__0_aux.distcp",
        "__0_payload.distcp",
    ]

    loaded = {
        "optimizer": {
            "large": torch.zeros_like(large),
            "step": torch.tensor(0.0),
        }
    }
    dcp.load(loaded, checkpoint_id=checkpoint)
    assert torch.equal(loaded["optimizer"]["large"], large)
    assert torch.equal(loaded["optimizer"]["step"], small)


def test_optimizer_writer_omits_empty_payload_bucket(tmp_path):
    checkpoint = tmp_path / "optimizer"
    state = {"optimizer": {"step": torch.tensor(3.0)}}

    _dcp_save(
        state,
        None,
        storage_writer=OptimizerFileSystemWriter(checkpoint),
    )

    assert [path.name for path in checkpoint.glob("*.distcp")] == ["__0_aux.distcp"]


def test_full_dcp_save_uses_optimizer_payload_layout(tmp_path):
    model = torch.nn.Linear(600, 600, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 600)).sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "00001.dcp"

    save_dcp(
        model,
        optimizer,
        str(checkpoint),
        manifest=CheckpointManifest.create(state="full", completed_steps=1),
    )

    optimizer_files = sorted(path.name for path in (checkpoint / "optimizer").glob("*.distcp"))
    assert optimizer_files == ["__0_aux.distcp", "__0_payload.distcp"]


def test_optimizer_writer_rejects_invalid_layout_options(tmp_path):
    try:
        OptimizerFileSystemWriter(tmp_path, aux_threshold_bytes=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("expected a negative threshold to fail")

    try:
        OptimizerFileSystemWriter(tmp_path, single_file_per_rank=False)
    except ValueError as exc:
        assert "single_file_per_rank=True" in str(exc)
    else:
        raise AssertionError("expected per-item file mode to fail")
