"""Same-topology full DCP and asynchronous snapshot isolation for EP banks."""

import copy
import dataclasses
import os
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from expert_parallel_models import MoE, ReferenceExperts, SequenceMoE
from torch.distributed.tensor import DTensor, Shard

from vforge.data.loader import build_resumable_loader
from vforge.distributed.expert_parallel import Expert
from vforge.distributed.parallel_state import (
    ParallelConfig,
    destroy_parallel_state,
    get_parallel_state,
)
from vforge.distributed.parallelize import (
    FSDP2Config,
    MixedPrecisionConfig,
    ParallelizeConfig,
    materialize_meta_module,
    parallelize_model,
)
from vforge.ops import initialize_ops
from vforge.ops.config import OpsConfig
from vforge.training.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointManifest,
    ResumeConfig,
)
from vforge.training.checkpoint.storage import (
    _get_dcp_state,
    _stage_dcp_state,
    convert_dcp_weights_to_pt,
    load_model_weights,
    load_training_state,
)
from vforge.training.engine.base import EnginePolicy, EngineRunContext
from vforge.training.engine.fsdp import FsdpEngine
from vforge.training.recipe import StepContext, StepOutput
from vforge.utils.rng import RandomStreams


def _assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        a = a.to_local() if hasattr(a, "to_local") else a
        b = b.to_local() if hasattr(b, "to_local") else b
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_equal(x, y)
    else:
        assert a == b


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return (value.to_local() if hasattr(value, "to_local") else value).detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return copy.deepcopy(value)


def run_checkpoint_case(
    rank,
    world,
    checkpoint_dir,
    *,
    device,
    ep=2,
    usp=1,
    dtype=torch.float64,
    amp=False,
    recompute=False,
    main_experts=8,
    accumulation=1,
    progress=lambda message: None,
):
    manager = None
    try:
        torch.manual_seed(42)
        config = ParallelizeConfig(
            parallel=ParallelConfig(
                dp_size=world // usp,
                usp_size=usp,
                extra_parallel_sizes=(ep,),
                device_type=device.type,
            )
        )
        config = dataclasses.replace(
            config,
            fsdp2=FSDP2Config(mixed_precision=MixedPrecisionConfig(enable=amp)),
            gradient_checkpointing=recompute,
        )
        engine = FsdpEngine(
            EnginePolicy(
                parallelize=config,
                gradient_clip_val=0.05,
                gradient_accumulation_steps=accumulation,
            )
        )
        experts_cls = Expert if amp else ReferenceExperts
        model_cls = SequenceMoE if usp > 1 else MoE
        model = engine.prepare_model(
            model_cls(
                main=main_experts, intermediate=16 if amp else 12, experts_cls=experts_cls
            ).to(device=device, dtype=dtype)
        )
        if usp > 1:
            initialize_ops(
                OpsConfig(backend="sdpa_math", allow_fallback=False, deterministic=True),
                device=device,
                parallel_state=get_parallel_state(),
            )
        assert engine.configure_optimizers(model, None, None) is None
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.0, foreach=False, fused=False
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=10
        )
        values = (
            torch.arange(world * 60 * accumulation, dtype=torch.float64)
            / (world * 60 * accumulation)
        )[:, None]
        values = values.expand(-1, 8)
        if usp > 1:
            values = values[:, None, :].expand(-1, 2 * usp, -1)
        dataset = torch.utils.data.TensorDataset(values)

        def make_loader():
            return build_resumable_loader(
                dataset,
                parallel_state=get_parallel_state(),
                batch_size=3,
                num_workers=0,
                drop_last=True,
                seed=42,
            )

        loader = make_loader()
        iterator = iter(loader)
        manager = CheckpointManager(
            CheckpointConfig(
                enabled=True, format="dcp", state="full", async_save=True, dirpath=checkpoint_dir
            )
        )
        torch.manual_seed(42 + rank)

        def step(current, opt, lr):
            def batches():
                for _ in range(accumulation):
                    inputs = next(iterator)[0].to(device=device, dtype=dtype)
                    noise = torch.randn_like(inputs)
                    if usp > 1:
                        state = get_parallel_state()
                        # USP peers consume the same stochastic batch; every rank
                        # still advances and restores its own RNG state.
                        dist.broadcast(noise, src=rank - state.usp_rank, group=state.usp_group)
                    yield inputs + noise * 0.1
                    del inputs, noise

            engine.run_update(
                model=current,
                microbatches=batches(),
                step=0,
                context=StepContext(device, get_parallel_state()),
                compute_step=lambda model, batch, step, context: StepOutput(
                    model(batch).square().sum()
                ),
                optimizer=opt,
                scheduler=lr,
                run=EngineRunContext(
                    precision_context=(
                        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
                        if amp
                        else nullcontext
                    )
                ),
            )

        for _ in range(5):
            step(model, optimizer, scheduler)
        progress("checkpoint: first 5 steps complete")
        saved = _snapshot(_get_dcp_state(model, optimizer))
        # Every bank uses standard packed DCP entries, including Main's EP axis.
        state = _get_dcp_state(model, optimizer)
        for name, parameter in model.main.named_parameters():
            key = f"main.{name}"
            for tensor in (
                state["model"][key],
                state["optimizer"]["state"][key]["exp_avg"],
                state["optimizer"]["state"][key]["exp_avg_sq"],
            ):
                assert isinstance(tensor, DTensor)
                assert tensor.shape == parameter.shape
                assert tensor.shape[0] == main_experts
                assert tensor.placements == (Shard(1), Shard(0))
        for name, parameter in model.block.named_parameters():
            key = f"block.{name}"
            weight = state["model"][key]
            assert weight.shape == parameter.shape
            assert weight.device_mesh.size() == world
            assert state["optimizer"]["state"][key]["exp_avg"].shape == parameter.shape
        from torch.utils._pytree import tree_leaves

        staged = _stage_dcp_state(_get_dcp_state(model, optimizer))
        assert all(
            value.device.type == "cpu"
            for value in tree_leaves(staged)
            if isinstance(value, torch.Tensor)
        )
        del staged
        saved_scheduler = copy.deepcopy(scheduler.state_dict())
        path = checkpoint_dir + "/00005.dcp"
        manager.save(
            model,
            optimizer,
            path,
            manifest=CheckpointManifest.create(state="full", completed_steps=5),
            rank_state={
                "rng": RandomStreams().state_dict(),
                "scheduler": scheduler.state_dict(),
                "data_position": 5 * accumulation,
                "data_loader": loader.state_dict(),
            },
        )
        progress("checkpoint: snapshot submitted")
        # Mutate while the asynchronous upload is in flight.
        for _ in range(5):
            step(model, optimizer, scheduler)
        expected = _snapshot(_get_dcp_state(model, optimizer))
        expected_rng = (
            torch.cuda.get_rng_state().clone()
            if device.type == "cuda"
            else torch.get_rng_state().clone()
        )
        expected_scheduler = copy.deepcopy(scheduler.state_dict())
        expected_loader = copy.deepcopy(loader.state_dict())
        manager.wait()
        progress("checkpoint: continuous 10 steps and async save complete")

        if device.type == "cpu":
            export_path = checkpoint_dir + "/export.pt"
            if rank == 0:
                convert_dcp_weights_to_pt(path, export_path)
            dist.barrier()

        with torch.device("meta"):
            restored = model_cls(
                main=main_experts, intermediate=16 if amp else 12, experts_cls=experts_cls
            ).to(dtype=dtype)
        restored = parallelize_model(restored, config)
        materialize_meta_module(restored, device)
        assert restored.runtime_scale.item() == 0.5
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(), lr=0.001, weight_decay=0.0, foreach=False, fused=False
        )
        restored_scheduler = torch.optim.lr_scheduler.LinearLR(
            restored_optimizer, start_factor=1.0, end_factor=0.1, total_iters=10
        )
        progress("checkpoint: loading state")
        _, sidecar = load_training_state(
            restored, restored_optimizer, path, resume=ResumeConfig(strict=True)
        )
        _assert_equal(_get_dcp_state(restored, restored_optimizer), saved)
        progress("checkpoint: loaded tensors equal snapshot")
        if device.type == "cpu":
            # Standard full-state loading places global weights into the
            # target DTensor shards after meta materialization.
            load_model_weights(restored, export_path, strict=True)
            # Portable PT exports intentionally use FP32 floating weights.
            torch.testing.assert_close(
                _snapshot(_get_dcp_state(restored, None))["model"],
                saved["model"],
                rtol=1e-7,
                atol=1e-8,
            )
            load_training_state(
                restored, restored_optimizer, path, resume=ResumeConfig(strict=True)
            )
        assert sidecar["data_position"] == 5 * accumulation
        restored_scheduler.load_state_dict(sidecar["scheduler"])
        _assert_equal(restored_scheduler.state_dict(), saved_scheduler)
        loader = make_loader()
        loader.load_state_dict(sidecar["data_loader"])
        iterator = iter(loader)
        RandomStreams().load_state_dict(sidecar["rng"])
        for _ in range(5):
            step(restored, restored_optimizer, restored_scheduler)
        _assert_equal(_get_dcp_state(restored, restored_optimizer), expected)
        _assert_equal(restored_scheduler.state_dict(), expected_scheduler)
        _assert_equal(loader.state_dict(), expected_loader)
        actual_rng = torch.cuda.get_rng_state() if device.type == "cuda" else torch.get_rng_state()
        assert torch.equal(actual_rng, expected_rng)
        progress("checkpoint: resumed 10 steps equal continuous run")
    finally:
        if manager is not None:
            manager.close()


def _checkpoint_worker(rank, world, init_file, checkpoint_dir, main_experts, recompute):
    os.environ["LOCAL_RANK"] = str(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        run_checkpoint_case(
            rank,
            world,
            checkpoint_dir,
            device=torch.device("cpu"),
            main_experts=main_experts,
            recompute=recompute,
        )
    finally:
        destroy_parallel_state()
        dist.destroy_process_group()


@pytest.mark.parametrize("main_experts,recompute", [(8, False), (2, False), (8, True)])
def test_ep_full_dcp_async_snapshot_and_exact_resume(tmp_path, main_experts, recompute):
    mp.spawn(
        _checkpoint_worker,
        args=(4, str(tmp_path / "pg"), str(tmp_path), main_experts, recompute),
        nprocs=4,
        join=True,
    )
