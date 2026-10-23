from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vforge.training.checkpoint.backend import (
    MCoreCheckpointBackend,
    PreparedMCoreSave,
    _validate_mcore_topology,
    build_mcore_model_sharded_state_dict,
)
from vforge.training.checkpoint.config import CheckpointConfig
from vforge.training.checkpoint.manager import CheckpointManager
from vforge.training.checkpoint.manifest import read_manifest
from vforge.training.mcore_optimizer import MCoreDistributedOptimizer


class _NativeLeaf(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))

    def sharded_state_dict(self, *, prefix, sharded_offsets, metadata):
        assert sharded_offsets == ()
        return {
            prefix + "weight": {
                "kind": "native",
                "data": self.weight,
                "dp_cp_group": metadata["dp_cp_group"],
            }
        }


def test_mixed_model_state_recurses_without_flattening_native_children(monkeypatch) -> None:
    from vforge.training.checkpoint import backend

    root = torch.nn.Module()
    root.register_parameter("replicated", torch.nn.Parameter(torch.ones(3)))
    root.sharded = torch.nn.Linear(2, 2, bias=False)
    root.native = _NativeLeaf()
    wrapper = SimpleNamespace(module=root, _vforge_data_parallel_backend="mcore")
    groups = SimpleNamespace(tp="tp-group", dp_cp="dp-cp-group")

    monkeypatch.setattr(backend, "mcore_process_group_collection", lambda _model: groups)
    monkeypatch.setattr(
        backend,
        "tensor_parallel_shard_axes",
        lambda module: {"weight": 0} if module is root.sharded else {},
    )

    transformer_utils = ModuleType("vforge.mcore.transformer.utils")

    def make_sharded(state, prefix, axes, **groups):
        assert groups == {"tp_group": "tp-group", "dp_cp_group": "dp-cp-group"}
        return {
            prefix + key: {
                "kind": "vforge",
                "data": value,
                "axis": axes.get(key),
            }
            for key, value in state.items()
        }

    transformer_utils.make_sharded_tensors_for_checkpoint = make_sharded
    monkeypatch.setitem(sys.modules, "vforge.mcore.transformer.utils", transformer_utils)

    state = build_mcore_model_sharded_state_dict(wrapper)

    assert state["replicated"]["axis"] is None
    assert state["sharded.weight"]["axis"] == 0
    assert state["native.weight"]["kind"] == "native"
    assert set(state) == {"replicated", "sharded.weight", "native.weight"}


def test_model_state_uses_parallel_state_without_allocating_mcore_ddp(monkeypatch) -> None:
    from vforge.training.checkpoint import backend

    model = torch.nn.Linear(2, 2, bias=False)
    monkeypatch.setattr(
        backend,
        "get_parallel_state",
        lambda: SimpleNamespace(
            attention_tp_group="eval-tp-group",
            dp_cp_group="eval-dp-cp-group",
        ),
    )
    monkeypatch.setattr(
        backend,
        "mcore_process_group_collection",
        lambda _model: (_ for _ in ()).throw(
            AssertionError("eval model must not require an MCore DDP wrapper")
        ),
    )
    monkeypatch.setattr(backend, "tensor_parallel_shard_axes", lambda _module: {})
    transformer_utils = ModuleType("vforge.mcore.transformer.utils")

    def make_sharded(state, prefix, _axes, **groups):
        assert groups == {
            "tp_group": "eval-tp-group",
            "dp_cp_group": "eval-dp-cp-group",
        }
        return {prefix + key: value for key, value in state.items()}

    transformer_utils.make_sharded_tensors_for_checkpoint = make_sharded
    monkeypatch.setitem(sys.modules, "vforge.mcore.transformer.utils", transformer_utils)

    state = build_mcore_model_sharded_state_dict(model)

    assert state == {"weight": model.weight}


def test_native_state_builder_temporarily_suspends_tp_export_hook(monkeypatch) -> None:
    from vforge.distributed.tensor_parallel import (
        suspend_tensor_parallel_state_dict_post_hooks,
    )
    from vforge.training.checkpoint import backend

    module = torch.nn.Linear(2, 2, bias=False)

    def export_hook(_module, state, _prefix, _metadata):
        state["weight"] = state["weight"] + 1

    handle = module.register_state_dict_post_hook(export_hook)
    module._vforge_tp_state_dict_post_hook = export_hook
    module._vforge_tp_state_dict_post_hook_handle = handle

    assert torch.equal(module.state_dict()["weight"], module.weight + 1)
    with suspend_tensor_parallel_state_dict_post_hooks(module):
        assert torch.equal(module.state_dict()["weight"], module.weight)
    assert torch.equal(module.state_dict()["weight"], module.weight + 1)

    wrapper = SimpleNamespace(module=module, _vforge_data_parallel_backend="mcore")
    monkeypatch.setattr(
        backend,
        "mcore_process_group_collection",
        lambda _model: SimpleNamespace(tp="tp", dp_cp="dp"),
    )
    monkeypatch.setattr(backend, "tensor_parallel_shard_axes", lambda _module: {})
    transformer_utils = ModuleType("vforge.mcore.transformer.utils")
    transformer_utils.make_sharded_tensors_for_checkpoint = (
        lambda state, _prefix, _axes, **_groups: state
    )
    monkeypatch.setitem(sys.modules, "vforge.mcore.transformer.utils", transformer_utils)

    native = build_mcore_model_sharded_state_dict(wrapper)

    assert torch.equal(native["weight"], module.weight)


def test_mcore_optimizer_delegates_native_dp_reshardable_state() -> None:
    calls = []

    class Inner:
        param_groups = []

        def sharded_state_dict(self, model_state, *, is_loading, metadata):
            calls.append((model_state, is_loading, metadata))
            return {"native": is_loading}

        def load_state_dict(self, state):
            assert not torch.is_grad_enabled()
            calls.append(("load", state))

    optimizer = MCoreDistributedOptimizer(torch.nn.Linear(1, 1), Inner(), clip_grad=0.0)

    assert optimizer.checkpoint_sharded_state_dict({"model": 1}, is_loading=True) == {
        "native": True
    }
    optimizer.load_checkpoint_sharded_state_dict({"optimizer": 2})

    assert calls == [
        (
            {"model": 1},
            True,
            {"distrib_optim_sharding_type": "dp_reshardable"},
        ),
        ("load", {"optimizer": 2}),
    ]


def test_mcore_topology_allows_only_dp_to_change(monkeypatch) -> None:
    from vforge.distributed import parallel_state

    current = SimpleNamespace(config=SimpleNamespace(dp_size=4, cp_size=2, tp_size=2, pp_size=1))
    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: current)
    manifest = SimpleNamespace(topology={"dp_size": 2, "cp_size": 2, "tp_size": 2, "pp_size": 1})

    assert _validate_mcore_topology(manifest) is False

    manifest.topology["tp_size"] = 4
    with pytest.raises(ValueError, match="only reshards the DP dimension"):
        _validate_mcore_topology(manifest)


def test_mcore_load_validates_native_and_trainer_components(monkeypatch) -> None:
    from vforge.training.checkpoint import backend
    from vforge.training.checkpoint.config import ResumeConfig

    manifest = SimpleNamespace(
        schema_version=2,
        format="mcore",
        state="full",
        topology={"dp_size": 1, "cp_size": 1, "tp_size": 1, "pp_size": 1},
        components={"weights": "mcore", "optimizer": "mcore", "trainer": "trainer"},
        components_present={"weights": True, "optimizer": True, "trainer": True},
    )
    validated = []
    monkeypatch.setattr(backend, "_validate_mcore_topology", lambda _manifest: True)
    monkeypatch.setattr(
        backend,
        "build_mcore_model_sharded_state_dict",
        lambda _model: {"weight": object()},
    )
    monkeypatch.setattr(
        backend,
        "unwrap_replicated_data_parallel",
        lambda model: model,
    )

    from vforge.training.checkpoint import storage

    monkeypatch.setattr(
        storage,
        "_validate_component_integrity",
        lambda path, _manifest, component, **kwargs: validated.append((path, component, kwargs)),
    )
    monkeypatch.setattr(storage, "_trainer_sidecar_path", lambda *_args: "/tmp/rank.pt")
    monkeypatch.setattr(storage.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(
        backend.torch,
        "load",
        lambda *_args, **_kwargs: {"scheduler": None},
    )

    dist_checkpointing = ModuleType("vforge.mcore.dist_checkpointing")
    dist_checkpointing.load = lambda target, *_args, **_kwargs: {
        "model": {},
        "optimizer": {"native": True},
    }
    mcore = ModuleType("vforge.mcore")
    mcore.dist_checkpointing = dist_checkpointing
    monkeypatch.setitem(sys.modules, "vforge.mcore", mcore)
    monkeypatch.setitem(sys.modules, "vforge.mcore.dist_checkpointing", dist_checkpointing)
    optimizer = SimpleNamespace(
        checkpoint_sharded_state_dict=lambda *_args, **_kwargs: {"native": object()},
        load_checkpoint_sharded_state_dict=lambda _state: None,
    )
    monkeypatch.setattr(backend, "_require_mcore_optimizer", lambda _optimizer: optimizer)

    model = torch.nn.Module()
    backend.load_mcore_training_state(
        model,
        optimizer,
        "/checkpoint/00001.mcore",
        manifest=manifest,
        resume=ResumeConfig(
            weights=True,
            optimizer=True,
            training_state=True,
            strict=False,
        ),
        weights_strict=False,
        integrity="full",
    )

    assert validated == [
        (
            "/checkpoint/00001.mcore",
            "weights",
            {"distributed": True, "mode": "full"},
        ),
        (
            "/checkpoint/00001.mcore",
            "trainer",
            {"distributed": True, "mode": "full"},
        ),
    ]


def test_mcore_async_requires_nvrx(monkeypatch) -> None:
    def missing_nvrx() -> None:
        raise ModuleNotFoundError(
            "checkpoint.format='mcore' with async_save=true requires "
            "nvidia-resiliency-ext>=0.6,<0.7"
        )

    monkeypatch.setattr(
        MCoreCheckpointBackend,
        "_require_nvrx",
        staticmethod(missing_nvrx),
    )

    with pytest.raises(ModuleNotFoundError, match="nvidia-resiliency-ext"):
        MCoreCheckpointBackend(async_save=True)


def test_mcore_wait_accepts_request_finalized_by_concurrent_poll() -> None:
    calls = []
    queue = SimpleNamespace(
        maybe_finalize_async_calls=lambda *, blocking: calls.append(blocking) or [],
    )
    backend = MCoreCheckpointBackend(async_save=False)
    backend._async_queue = queue

    backend.wait(17)

    assert calls == [True]


def test_mcore_manager_polls_native_request_before_atomic_commit(tmp_path, monkeypatch) -> None:
    from vforge.training.checkpoint import manager as checkpoint_manager
    from vforge.training.checkpoint import storage

    class FakeBackend:
        def __init__(self, *, async_save):
            assert async_save
            self.finalize = None
            self.closed = False

        def prepare_save(self, _model, _optimizer, path, *, manifest, **_kwargs):
            incomplete = storage._prepare_dcp_directory(path, manifest)
            payload = tmp_path / "00001.mcore.incomplete" / "mcore"
            payload.mkdir()
            (payload / "native.distcp").write_bytes(b"native")
            return PreparedMCoreSave(
                path=path,
                incomplete=incomplete,
                payload_path=str(payload),
                manifest=replace(
                    manifest,
                    components={"weights": "mcore"},
                    components_present={
                        "weights": True,
                        "optimizer": False,
                        "trainer": False,
                    },
                ),
                state={"model": {}},
                state_dict_s=0.01,
                sidecar_s=0.0,
            )

        def schedule_async(self, _prepared, *, finalize):
            self.finalize = finalize
            return 17

        def poll(self, call_index):
            assert call_index == 17
            assert self.finalize is not None
            finalize, self.finalize = self.finalize, None
            finalize()
            return True

        def wait(self, call_index):
            self.poll(call_index)

        def close(self):
            self.closed = True

    monkeypatch.setattr(checkpoint_manager, "MCoreCheckpointBackend", FakeBackend)
    manager = CheckpointManager(
        CheckpointConfig(
            enabled=True,
            every_n_steps=1,
            dirpath=str(tmp_path),
            state="weights",
            format="mcore",
            async_save=True,
        )
    )
    result = manager.save(
        torch.nn.Linear(1, 1),
        None,
        None,
        1,
        dataloader_state={"contract": {}},
        execution_policy={},
        topology={"dp_size": 1, "cp_size": 1, "tp_size": 1, "pp_size": 1},
    )

    assert result.pending is not None
    assert not result.pending.done()
    assert not (tmp_path / "00001.mcore").exists()
    completed = manager.poll()

    assert completed is not None
    assert completed.completed_steps == 1
    assert read_manifest(str(tmp_path / "00001.mcore")).format == "mcore"
    assert not (tmp_path / "00001.mcore.incomplete").exists()
    backend = manager._mcore_backend
    manager.close()
    assert backend.closed
