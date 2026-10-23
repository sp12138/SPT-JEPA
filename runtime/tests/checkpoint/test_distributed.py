import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vforge.training.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointManifest,
    ResumeConfig,
    load_dcp,
    load_training_state,
    save_dcp,
)


class _SparseGradientModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(3, 2)
        self.unused = torch.nn.Linear(3, 2)

    def forward(self, inputs):
        return self.used(inputs)


def _distributed_dcp_worker(
    rank,
    world_size,
    init_file,
    checkpoint_path,
    async_save,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(17)
        model = _SparseGradientModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = model(torch.ones(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        assert model.unused.weight not in optimizer.state
        assert model.unused.bias not in optimizer.state
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        manifest = CheckpointManifest.create(
            state="full",
            completed_steps=1,
            execution_policy={"mode": "performance", "seed": 42},
            topology={"dp_size": world_size},
        )
        rank_state = {"rank": rank, "token": torch.tensor([rank])}
        if async_save:
            manager = CheckpointManager(
                CheckpointConfig(
                    enabled=True,
                    every_n_steps=1,
                    dirpath=os.path.dirname(checkpoint_path),
                    state="full",
                    format="dcp",
                    async_save=True,
                )
            )
            manager.save(
                model,
                optimizer,
                checkpoint_path,
                manifest=manifest,
                rank_state=rank_state,
            )
            manager.close()
        else:
            save_dcp(
                model,
                optimizer,
                checkpoint_path,
                manifest=manifest,
                rank_state=rank_state,
            )

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        loaded_manifest, rank_state = load_dcp(
            model,
            optimizer,
            checkpoint_path,
        )

        assert loaded_manifest.completed_steps == 1
        assert rank_state["rank"] == rank
        assert rank_state["token"].item() == rank
        assert model.unused.weight not in optimizer.state
        assert model.unused.bias not in optimizer.state
        for name, value in model.state_dict().items():
            assert torch.equal(value, expected[name])
    finally:
        dist.destroy_process_group()


def _distributed_async_phase_failure_worker(
    rank,
    world_size,
    init_file,
    checkpoint_path,
    phase,
):
    from vforge.training.checkpoint import storage as checkpoint_storage

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    manager = None
    original_save = torch.save
    original_stage = checkpoint_storage._stage_dcp_state
    original_writer = checkpoint_storage._dcp_file_system_writer
    messages = {
        "sidecar": "injected sidecar failure",
        "staging": "injected staging failure",
        "thread_submit": "injected thread submit failure",
        "writer_setup": "injected writer setup failure",
    }
    try:
        model = _SparseGradientModel()
        uses_sidecar = phase == "sidecar"
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) if uses_sidecar else None
        manager = CheckpointManager(
            CheckpointConfig(
                enabled=True,
                every_n_steps=1,
                dirpath=os.path.dirname(checkpoint_path),
                state="full" if uses_sidecar else "weights",
                format="dcp",
                async_save=True,
            )
        )
        if rank == 1:
            if phase == "sidecar":

                def fail_sidecar(payload, path):
                    if str(path).endswith("rank-00001.pt.tmp"):
                        raise OSError(messages[phase])
                    return original_save(payload, path)

                torch.save = fail_sidecar
            elif phase == "staging":

                def fail_staging(state):
                    raise OSError(messages[phase])

                checkpoint_storage._stage_dcp_state = fail_staging
            elif phase == "thread_submit":

                def fail_submit(*args, **kwargs):
                    raise RuntimeError(messages[phase])

                manager._executor.submit = fail_submit
            elif phase == "writer_setup":

                def fail_writer(checkpoint_id, **kwargs):
                    raise OSError(messages[phase])

                checkpoint_storage._dcp_file_system_writer = fail_writer
            else:
                raise ValueError(f"unsupported failure phase: {phase}")
        try:
            manager.save(
                model,
                optimizer,
                checkpoint_path,
                manifest=CheckpointManifest.create(
                    state="full" if uses_sidecar else "weights",
                    completed_steps=1,
                ),
                rank_state={"rank": rank} if uses_sidecar else None,
            )
        except RuntimeError as exc:
            assert messages[phase] in str(exc)
        else:
            raise AssertionError(f"expected distributed {phase} failure")
    finally:
        torch.save = original_save
        checkpoint_storage._stage_dcp_state = original_stage
        checkpoint_storage._dcp_file_system_writer = original_writer
        if manager is not None:
            manager.close()
        dist.destroy_process_group()


def _optimizer_reshard_save_worker(
    rank,
    world_size,
    init_file,
    checkpoint_path,
    expected_path,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(17)
        model = _SparseGradientModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
        save_dcp(
            model,
            optimizer,
            checkpoint_path,
            manifest=CheckpointManifest.create(
                state="full",
                completed_steps=1,
                topology={"dp_size": world_size},
            ),
            rank_state=None,
        )
        if rank == 0:
            parameter_names = {parameter: name for name, parameter in model.named_parameters()}
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": {
                        parameter_names[parameter]: {
                            key: (
                                value.detach().clone() if isinstance(value, torch.Tensor) else value
                            )
                            for key, value in state.items()
                        }
                        for parameter, state in optimizer.state.items()
                    },
                },
                expected_path,
            )
    finally:
        dist.destroy_process_group()


def _optimizer_reshard_load_worker(
    rank,
    world_size,
    init_file,
    checkpoint_path,
    expected_path,
    staging_root,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        local_world_size = 2
        node = rank // local_world_size
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
        os.environ["LOCAL_RANK"] = str(rank % local_world_size)
        os.environ["GROUP_RANK"] = str(node)
        os.environ["VFORGE_DCP_PLANNER_SELECTIVE"] = "1"
        os.environ["VFORGE_CKPT_LOCAL_ROOT"] = os.path.join(
            staging_root,
            f"node-{node}",
        )
        torch.manual_seed(99)
        model = _SparseGradientModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        from vforge.training.checkpoint import prepare_staged_dcp_path

        staged_path = prepare_staged_dcp_path(checkpoint_path, enabled=True)
        assert staged_path == checkpoint_path
        manifest, rank_state = load_training_state(
            model,
            optimizer,
            staged_path,
            resume=ResumeConfig(
                weights=True,
                optimizer=True,
                training_state=False,
                strict=False,
            ),
            warn_on_non_bit_exact=False,
        )
        expected = torch.load(expected_path, weights_only=False)
        assert manifest.world_size != world_size
        assert rank_state is None
        for name, value in model.state_dict().items():
            assert torch.equal(value, expected["model"][name])
        parameter_names = {parameter: name for name, parameter in model.named_parameters()}
        actual_optimizer = {
            parameter_names[parameter]: state for parameter, state in optimizer.state.items()
        }
        assert actual_optimizer.keys() == expected["optimizer"].keys()
        for name, expected_state in expected["optimizer"].items():
            actual_state = actual_optimizer[name]
            assert actual_state.keys() == expected_state.keys()
            for key, expected_value in expected_state.items():
                actual_value = actual_state[key]
                if isinstance(expected_value, torch.Tensor):
                    assert torch.equal(actual_value, expected_value)
                else:
                    assert actual_value == expected_value
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("async_save", [False, True])
def test_two_rank_full_dcp_roundtrip(tmp_path, async_save):
    world_size = 2
    init_file = str(tmp_path / f"{'async-' if async_save else ''}process-group")
    checkpoint_path = str(tmp_path / "00001.dcp")
    if os.path.exists(init_file):
        os.remove(init_file)

    mp.spawn(
        _distributed_dcp_worker,
        args=(world_size, init_file, checkpoint_path, async_save),
        nprocs=world_size,
        join=True,
    )


def test_optimizer_dcp_reshards_across_world_size(tmp_path):
    checkpoint_path = str(tmp_path / "00001.dcp")
    expected_path = str(tmp_path / "expected.pt")
    staging_root = str(tmp_path / "staging")
    save_init = str(tmp_path / "save-process-group")
    load_init = str(tmp_path / "load-process-group")

    mp.spawn(
        _optimizer_reshard_save_worker,
        args=(2, save_init, checkpoint_path, expected_path),
        nprocs=2,
        join=True,
    )
    mp.spawn(
        _optimizer_reshard_load_worker,
        args=(4, load_init, checkpoint_path, expected_path, staging_root),
        nprocs=4,
        join=True,
    )


@pytest.mark.parametrize(
    "phase",
    ["sidecar", "staging", "thread_submit", "writer_setup"],
)
def test_two_rank_async_phase_failure_propagates(tmp_path, phase):
    world_size = 2
    init_file = str(tmp_path / f"{phase}-failure-process-group")
    checkpoint_path = str(tmp_path / "00001.dcp")
    if os.path.exists(init_file):
        os.remove(init_file)

    mp.spawn(
        _distributed_async_phase_failure_worker,
        args=(world_size, init_file, checkpoint_path, phase),
        nprocs=world_size,
        join=True,
    )
    assert not os.path.exists(checkpoint_path)
