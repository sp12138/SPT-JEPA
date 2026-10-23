from __future__ import annotations

from vforge.training.checkpoint import CheckpointManifest, prepare_staged_dcp_path
from vforge.training.checkpoint.manifest import write_manifest
from vforge.training.checkpoint.planner_staging import (
    PlannerPreselection,
    planner_selective_path_enabled,
)
from vforge.utils.node_staging import StagingResult


def _source(tmp_path):
    source = tmp_path / "remote" / "00002.dcp"
    source.mkdir(parents=True)
    write_manifest(
        str(source),
        CheckpointManifest.create(state="full", completed_steps=2),
    )
    return source


def _staged(path, destination, *, files=20, bytes_=1234, backend="s3-crt"):
    return StagingResult(
        source=path,
        destination=str(destination),
        copied=True,
        files=files,
        bytes=bytes_,
        backend=backend,
    )


def test_prepare_staged_dcp_path_is_disabled_by_default(tmp_path) -> None:
    path = str(tmp_path / "missing.dcp")

    assert prepare_staged_dcp_path(path, enabled=False) == path


def test_prepare_staged_dcp_path_uses_node_leader_staging(
    monkeypatch,
    tmp_path,
) -> None:
    import vforge.utils.node_staging as node_staging

    source = _source(tmp_path)
    calls = []

    def fake_stage(path, *, config, overlap_callback):
        calls.append(
            (
                path,
                config.root,
                config.copy_workers,
                config.s3_transfer_concurrency,
                overlap_callback,
            )
        )
        return _staged(path, "/local-ssd/checkpoints/00002.dcp")

    monkeypatch.setenv("VFORGE_CKPT_LOCAL_ROOT", "/local-ssd/checkpoints")
    monkeypatch.setenv("VFORGE_STAGE_WORKERS", "7")
    monkeypatch.setenv("VFORGE_S3_TRANSFER_CONCURRENCY", "31")
    monkeypatch.setattr(node_staging, "stage_on_node", fake_stage)

    staged = prepare_staged_dcp_path(str(source), enabled=True)

    assert staged == "/local-ssd/checkpoints/00002.dcp"
    assert calls == [
        (
            str(source),
            "/local-ssd/checkpoints",
            7,
            31,
            None,
        )
    ]


def test_prepare_staged_dcp_path_warms_distributed_during_copy(
    monkeypatch,
    tmp_path,
) -> None:
    from vforge.training.checkpoint import storage as checkpoint_storage
    from vforge.utils import node_staging

    source = _source(tmp_path)
    barriers = []

    def fake_stage(path, *, config, overlap_callback):
        assert config.root
        assert overlap_callback is not None
        overlap_callback()
        return _staged(path, "/local-ssd/checkpoints/00002.dcp")

    monkeypatch.setattr(checkpoint_storage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoint_storage.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(checkpoint_storage.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(checkpoint_storage.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        checkpoint_storage.dist,
        "barrier",
        lambda: barriers.append("barrier"),
    )
    monkeypatch.setattr(node_staging, "stage_on_node", fake_stage)

    staged = prepare_staged_dcp_path(str(source), enabled=True)

    assert staged == "/local-ssd/checkpoints/00002.dcp"
    assert barriers == ["barrier"]


def test_prepare_staged_dcp_path_activates_selective_reader(
    monkeypatch,
    tmp_path,
) -> None:
    from vforge.training.checkpoint import planner_staging
    from vforge.training.checkpoint import storage as checkpoint_storage
    from vforge.utils import node_staging

    source = _source(tmp_path)
    monkeypatch.setenv("VFORGE_DCP_PLANNER_SELECTIVE", "1")
    local = tmp_path / "local" / "00002.dcp"

    monkeypatch.setattr(
        planner_staging,
        "build_node_preselection",
        lambda _path: PlannerPreselection(
            relative_paths=("manifest.json", "weights/.metadata"),
            component_file_sizes={"weights": {".metadata": 12}},
        ),
    )

    def fake_stage(path, *, config, overlap_callback, relative_paths):
        assert path == str(source)
        assert config.root
        assert overlap_callback is None
        assert relative_paths == ("manifest.json", "weights/.metadata")
        return _staged(
            path,
            local,
            files=2,
            bytes_=12,
            backend="filesystem",
        )

    monkeypatch.setattr(node_staging, "stage_on_node", fake_stage)
    monkeypatch.setattr(
        checkpoint_storage,
        "build_node_preselection",
        planner_staging.build_node_preselection,
        raising=False,
    )

    staged = prepare_staged_dcp_path(str(source), enabled=True)

    assert staged == str(local)
    assert planner_selective_path_enabled(local / "weights")
