from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint.metadata import MetadataIndex

from vforge.training.checkpoint.manifest import (
    CheckpointManifest,
    write_manifest,
)
from vforge.training.checkpoint.planner_staging import (
    ObservingFileSystemReader,
    PlannerStagingError,
    ReplayLoadPlanner,
    build_node_preselection,
    compute_node_unions,
    enrich_node_unions_with_file_sizes,
    map_plan_items,
    node_id_for_rank,
    observe_rank_plan,
    parse_distcp_owner_rank,
    planner_observe_enabled,
    pop_replay_plan,
    register_replay_plan,
    reset_planner_staging_state,
    summarize_component,
)


def test_parse_distcp_owner_rank() -> None:
    assert parse_distcp_owner_rank("__3_0.distcp") == 3
    assert parse_distcp_owner_rank("weights/__12_0.distcp") == 12
    assert parse_distcp_owner_rank(".metadata") is None


def test_replay_plan_cache_is_single_use(tmp_path) -> None:
    checkpoint_id = tmp_path / "weights"
    plan = SimpleNamespace(items=["read"])

    register_replay_plan(checkpoint_id, plan)
    with pytest.raises(PlannerStagingError, match="already registered"):
        register_replay_plan(checkpoint_id, plan)

    assert pop_replay_plan(checkpoint_id) is plan
    assert pop_replay_plan(checkpoint_id) is None


def test_reset_planner_staging_state_clears_replay_cache(tmp_path) -> None:
    checkpoint_id = tmp_path / "weights"
    register_replay_plan(checkpoint_id, SimpleNamespace(items=["read"]))

    reset_planner_staging_state()

    assert pop_replay_plan(checkpoint_id) is None
    register_replay_plan(checkpoint_id, SimpleNamespace(items=["next"]))
    assert pop_replay_plan(checkpoint_id) is not None


def test_replay_load_planner_reuses_exact_items() -> None:
    calls = []

    class Delegate:
        def set_up_planner(self, state_dict, metadata, is_coordinator):
            calls.append(("setup", state_dict, metadata, is_coordinator))

        def finish_plan(self, plan):
            calls.append(("finish", plan))
            return plan

        def load_bytes(self, read_item, value):
            calls.append(("bytes", read_item, value))

        def resolve_tensor(self, read_item):
            calls.append(("resolve", read_item))
            return "tensor"

        def commit_tensor(self, read_item, tensor):
            calls.append(("commit", read_item, tensor))

    final_plan = SimpleNamespace(items=["exact-final-read"])
    replay = ReplayLoadPlanner(Delegate(), final_plan)

    replay.set_up_planner({"model": 1}, "metadata", True)
    assert replay.create_local_plan() is final_plan
    assert replay.create_global_plan([final_plan]) == [final_plan]
    assert replay.finish_plan(final_plan) is final_plan
    assert replay.resolve_tensor("read") == "tensor"
    replay.commit_tensor("read", "tensor")
    replay.load_bytes("read", "bytes")
    assert calls[0] == ("setup", {"model": 1}, "metadata", True)


def test_build_node_preselection_uses_payload_aux_layout(
    monkeypatch,
    tmp_path,
) -> None:
    root = tmp_path / "00001.dcp"
    root.mkdir()
    manifest = replace(
        CheckpointManifest.create(state="full", completed_steps=1),
        world_size=8,
        components={
            "weights": "weights",
            "optimizer": "optimizer",
            "trainer": "trainer",
        },
        components_present={
            "weights": True,
            "optimizer": True,
            "trainer": True,
        },
    )
    write_manifest(str(root), manifest)
    for component in ("weights", "optimizer", "trainer"):
        (root / component).mkdir()
    (root / "weights" / ".metadata").write_bytes(b"metadata")
    (root / "optimizer" / ".metadata").write_bytes(b"metadata")
    for rank in range(8):
        (root / "weights" / f"__{rank}_0.distcp").write_bytes(b"w")
        (root / "optimizer" / f"__{rank}_0_payload.distcp").write_bytes(b"p")
        (root / "optimizer" / f"__{rank}_1_aux.distcp").write_bytes(b"a")
        (root / "trainer" / f"rank-{rank:05d}.pt").write_bytes(b"t")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(dist, "get_rank", lambda: 4)

    selection = build_node_preselection(root)

    assert "weights/__4_0.distcp" in selection.relative_paths
    assert "weights/__0_0.distcp" not in selection.relative_paths
    assert "optimizer/__4_0_payload.distcp" in selection.relative_paths
    assert "optimizer/__0_0_payload.distcp" not in selection.relative_paths
    assert "optimizer/__0_1_aux.distcp" in selection.relative_paths
    assert "trainer/rank-00004.pt" in selection.relative_paths
    assert "trainer/rank-00000.pt" not in selection.relative_paths


def test_map_plan_items_requires_complete_coverage() -> None:
    storage_index = MetadataIndex("model.w")
    missing = MetadataIndex("model.missing")
    storage_data = {
        storage_index: SimpleNamespace(
            relative_path="__0_0.distcp",
            offset=0,
            length=16,
        )
    }
    items = [
        SimpleNamespace(storage_index=storage_index, lengths=torch.Size([4])),
        SimpleNamespace(storage_index=missing, lengths=torch.Size([2])),
    ]

    with pytest.raises(PlannerStagingError, match="mapping incomplete"):
        map_plan_items(items, storage_data, require_complete=True)

    mapped, unmapped = map_plan_items(items, storage_data, require_complete=False)
    assert len(mapped) == 1
    assert unmapped == ["model.missing|None|None"]


def test_observe_rank_plan_and_node_union_metrics() -> None:
    idx0 = MetadataIndex("w0")
    idx1 = MetadataIndex("w1")
    idx2 = MetadataIndex("w2")
    storage_data = {
        idx0: SimpleNamespace(relative_path="__0_0.distcp", offset=0, length=10),
        idx1: SimpleNamespace(relative_path="__1_0.distcp", offset=0, length=20),
        idx2: SimpleNamespace(relative_path="__0_0.distcp", offset=10, length=5),
    }
    file_sizes = {"__0_0.distcp": 100, "__1_0.distcp": 200}
    rank0 = observe_rank_plan(
        rank=0,
        component="weights",
        plan_items=[
            SimpleNamespace(storage_index=idx0, lengths=torch.Size([10])),
            SimpleNamespace(storage_index=idx2, lengths=torch.Size([5])),
        ],
        storage_data=storage_data,
        file_sizes=file_sizes,
    )
    rank1 = observe_rank_plan(
        rank=1,
        component="weights",
        plan_items=[SimpleNamespace(storage_index=idx1, lengths=torch.Size([20]))],
        storage_data=storage_data,
        file_sizes=file_sizes,
    )
    # Rank 1 reading a file named for rank 0 counts as foreign.
    rank1_foreign = observe_rank_plan(
        rank=1,
        component="optimizer",
        plan_items=[SimpleNamespace(storage_index=idx0, lengths=torch.Size([10]))],
        storage_data=storage_data,
        file_sizes=file_sizes,
    )

    assert rank0.relative_paths == ["__0_0.distcp"]
    assert rank0.selected_file_bytes == 100
    assert rank0.logical_requested_bytes == 15
    assert rank0.foreign_file_count == 0
    assert rank1_foreign.foreign_file_count == 1
    assert rank1_foreign.foreign_file_bytes == 100

    unions = enrich_node_unions_with_file_sizes(
        compute_node_unions(
            [rank0, rank1],
            ranks_per_node=1,
            full_component_bytes=300,
        ),
        file_sizes,
        full_component_bytes=300,
    )
    assert [union.node for union in unions] == [0, 1]
    assert unions[0].selected_file_bytes == 100
    assert unions[1].selected_file_bytes == 200
    assert unions[0].selection_ratio == pytest.approx(100 / 300)

    shared = observe_rank_plan(
        rank=1,
        component="weights",
        plan_items=[SimpleNamespace(storage_index=idx0, lengths=torch.Size([10]))],
        storage_data=storage_data,
        file_sizes=file_sizes,
    )
    two_rank_node = enrich_node_unions_with_file_sizes(
        compute_node_unions(
            [rank0, shared],
            ranks_per_node=2,
            full_component_bytes=300,
        ),
        file_sizes,
        full_component_bytes=300,
    )
    assert len(two_rank_node) == 1
    assert two_rank_node[0].relative_paths == ["__0_0.distcp"]
    assert two_rank_node[0].selected_file_bytes == 100

    summary = summarize_component(
        component="weights",
        rank_observations=[rank0, rank1],
        node_unions=unions,
        full_component_bytes=300,
    )
    assert summary["mapping_coverage"] == 1.0
    assert summary["remote_duplication_ratio"] == pytest.approx(1.0)


def test_node_id_for_rank() -> None:
    assert node_id_for_rank(0, 8) == 0
    assert node_id_for_rank(7, 8) == 0
    assert node_id_for_rank(8, 8) == 1
    with pytest.raises(PlannerStagingError):
        node_id_for_rank(0, 0)


def test_planner_observe_enabled(monkeypatch) -> None:
    monkeypatch.delenv("VFORGE_DCP_PLANNER_OBSERVE", raising=False)
    assert planner_observe_enabled() is False
    monkeypatch.setenv("VFORGE_DCP_PLANNER_OBSERVE", "1")
    assert planner_observe_enabled() is True


def _reader_roundtrip_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    output_dir,
    staging_root,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        import os

        import torch.distributed.checkpoint as dcp

        ranks_per_node = 2
        selective = staging_root is not None
        if selective:
            node = rank // ranks_per_node
            os.environ["LOCAL_RANK"] = str(rank % ranks_per_node)
            os.environ["LOCAL_WORLD_SIZE"] = str(ranks_per_node)
            os.environ["VFORGE_CKPT_LOCAL_ROOT"] = os.path.join(
                staging_root,
                f"node-{node}",
            )
            os.environ["VFORGE_NODE_IO_INLINE"] = "1"
        torch.manual_seed(0)
        width = 16 if selective else 4
        model = torch.nn.Linear(width, width, bias=False)
        expected = model.weight.detach().clone()
        dcp.save(
            state_dict={"model": model.state_dict()},
            checkpoint_id=checkpoint_dir,
        )
        with torch.no_grad():
            model.weight.zero_()

        reader = ObservingFileSystemReader(
            checkpoint_dir,
            component="weights",
            output_dir=output_dir,
            ranks_per_node=ranks_per_node,
            selective=selective,
        )
        dcp.load(
            state_dict={"model": model.state_dict()},
            storage_reader=reader,
        )
        assert torch.equal(model.weight, expected)
    finally:
        dist.destroy_process_group()


def test_observing_and_selective_reader_roundtrips(tmp_path) -> None:
    world_size = 4
    for name, staging_root in (
        ("observe", None),
        ("selective", str(tmp_path / "staging")),
    ):
        init_file = tmp_path / f"{name}-init"
        init_file.write_text("")
        mp.spawn(
            _reader_roundtrip_worker,
            args=(
                world_size,
                str(init_file),
                str(tmp_path / f"{name}-ckpt"),
                str(tmp_path / f"{name}-output"),
                staging_root,
            ),
            nprocs=world_size,
            join=True,
        )

    for node in range(2):
        node_root = tmp_path / "staging" / f"node-{node}"
        staged = [path for path in node_root.iterdir() if path.is_dir()]
        assert len(staged) == 1
        assert (staged[0] / ".metadata").is_file()
        assert (staged[0] / "READY.json").is_file()
    for name in ("observe", "selective"):
        output = tmp_path / f"{name}-output"
        summary = (output / "summary-weights.json").read_text(encoding="utf-8")
        assert '"component": "weights"' in summary
        assert '"mapping_coverage": 1.0' in summary
        assert (output / "summary-weights.md").exists()
        assert (output / "rank-000-weights.json").exists()
