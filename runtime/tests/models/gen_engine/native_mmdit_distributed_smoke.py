"""Multi-rank native MMDiT smoke: FSDP2 everywhere, USP with NCCL."""

from __future__ import annotations

import gc
import os

import torch
import torch.distributed as dist

from vforge.distributed.parallel_state import (
    ParallelConfig,
    destroy_parallel_state,
    init_parallel_state,
)
from vforge.distributed.parallelize import (
    FSDP2Config,
    ParallelizeConfig,
    materialize_meta_module,
    parallelize_model,
)
from vforge.models.gen_engine.dit import GenEngineModel
from vforge.ops import initialize_ops, reset_ops
from vforge.ops.config import OpsConfig


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _assert_replicated(value: torch.Tensor, *, group=None) -> None:
    replicas = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
    dist.all_gather(replicas, value, group=group)
    for replica in replicas[1:]:
        torch.testing.assert_close(replica, replicas[0])


def _assert_globally_nonzero_gradient(parameter: torch.Tensor) -> None:
    local_parameter = _local_tensor(parameter)
    count = torch.zeros((), dtype=torch.int64, device=local_parameter.device)
    if parameter.grad is not None:
        count = torch.count_nonzero(_local_tensor(parameter.grad)).to(torch.int64)
    dist.all_reduce(count)
    assert count.item() > 0


def _config(architecture: str, *, num_attention_heads: int) -> dict:
    hidden_size = 8 * num_attention_heads
    value = {
        "schema": "vforge.gen_engine.dit",
        "schema_version": 1,
        "architecture": architecture,
        "num_attention_heads": num_attention_heads,
        "attention_head_dim": 8,
        "hidden_size": hidden_size,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "ffn_dim": 2 * hidden_size,
        "in_channels": 1,
        "audio_in_channels": 3,
        "patch_size": [1, 1, 1],
        "text_dim": 4,
        "freq_dim": 8,
        "time_embed_hidden_dim": hidden_size,
        "time_embed_dim": hidden_size // 2,
        "rope_freq_dim": 1,
        "max_reference_index": 4,
    }
    if architecture == "mmdit_moe":
        value["moe"] = {
            "start_layer": 1,
            "expert_intermediate_size": 8,
            "main_experts": 3,
            "main_top_k": 2,
            "block_experts": 2,
            "block_top_k": 1,
            "shared_multiplier": 2,
            "shared_gate": "sigmoid",
        }
    return value


def _inputs(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(1, 3, 1, device=device),
        "audio_hidden_states": torch.randn(1, 2, 3, device=device),
        "encoder_hidden_states": torch.randn(1, 2, 4, device=device),
        "timestep": torch.tensor([0.25, 0.75], device=device),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 0, 1], device=device),
        "token_tags": torch.tensor([1, 1, 0, 0, 0, 2, 2], device=device),
        "position_ids": torch.randn(7, 3, device=device),
        "text_indices": torch.tensor([0, 1], device=device),
        "video_indices": torch.tensor([2, 3, 4], device=device),
        "audio_indices": torch.tensor([5, 6], device=device),
        "reference_modality_ids": torch.tensor(
            [0, 0, 1, 2, 0, 3, 0],
            device=device,
        ),
        "reference_index_ids": torch.tensor([0, 0, 1, 2, 0, 3, 0], device=device),
        "audio_plane_ids": torch.tensor([0, 0, 0, 0, 0, 1, 2], device=device),
    }


def _run_variant(
    architecture: str,
    *,
    device: torch.device,
    state,
) -> None:
    debug = os.environ.get("VFORGE_MMDIT_SMOKE_DEBUG") == "1"

    def stage(name: str) -> None:
        if debug:
            print(f"rank={dist.get_rank()} architecture={architecture} stage={name}", flush=True)

    model = GenEngineModel.build_empty(
        config=_config(
            architecture,
            num_attention_heads=max(2, int(state.config.usp_size)),
        ),
        parameter_dtype=torch.float32,
    )
    parallelize_model(
        model,
        ParallelizeConfig(
            parallel=state.config,
            fsdp2=FSDP2Config(enable=True),
        ),
    )
    stage("parallelized")
    materialize_meta_module(model, device)
    model.initialize_parameters(seed=17)
    first_local_state = {
        name: _local_tensor(parameter).detach().clone()
        for name, parameter in model.named_parameters()
    }
    model.initialize_parameters(seed=17)
    for name, parameter in model.named_parameters():
        assert torch.equal(_local_tensor(parameter), first_local_state[name]), name
    assert all(not parameter.is_meta for parameter in model.parameters())
    if dist.get_world_size(state.fsdp_group) > 1:
        from torch.distributed.tensor import DTensor

        assert any(isinstance(parameter, DTensor) for parameter in model.parameters())
    if state.config.dp_replicate_size > 1:
        projection = _local_tensor(model.transformer.proj_in.weight).float()
        checksum = torch.stack((projection.sum(), projection.square().sum()))
        replica_group = state.device_mesh.get_group("dp_replicate")
        _assert_replicated(checksum, group=replica_group)
    stage("initialized")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.manual_seed(23 if state.config.usp_size > 1 else 23 + dist.get_rank())
    output = model(**_inputs(device))
    stage("forward")
    assert torch.count_nonzero(output.sample) == 0
    assert torch.count_nonzero(output.audio_sample) == 0
    video_target = torch.full_like(output.sample, 0.25)
    audio_target = torch.full_like(output.audio_sample, -0.5)
    loss = (output.sample.float() - video_target).square().mean()
    loss = loss + (output.audio_sample.float() - audio_target).square().mean()
    loss.backward()
    stage("backward")
    _assert_globally_nonzero_gradient(model.transformer.proj_out.weight)
    _assert_globally_nonzero_gradient(model.transformer.audio_proj_out.weight)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    stage("optimizer")
    metrics = model.post_optimizer_step(
        completed_step=1,
        total_steps=10,
        process_group=state.fsdp_group,
    )
    stage("post_step")
    assert torch.isfinite(loss)
    if architecture == "mmdit_moe":
        assert "moe/layer_1/main/load_cv" in metrics
        bias = model.transformer.transformer_blocks[1].visual_moe.router.main_routing_bias
        _assert_replicated(bias)
    else:
        assert metrics == {}
    del optimizer, model
    gc.collect()
    dist.barrier()


def main() -> None:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"])) if use_cuda else torch.device("cpu")
    )
    dist.init_process_group("nccl" if use_cuda else "gloo")
    world_size = dist.get_world_size()
    if world_size not in {2, 4, 8}:
        raise RuntimeError("native_mmdit_distributed_smoke.py requires 2, 4, or 8 ranks")
    use_hsdp = not use_cuda and world_size >= 4 and os.environ.get("VFORGE_MMDIT_HSDP") == "1"
    state = init_parallel_state(
        ParallelConfig(
            dp_size=1 if use_cuda else world_size,
            dp_replicate_size=2 if use_hsdp else 1,
            dp_shard_size=world_size // 2 if use_hsdp else 1,
            usp_size=world_size if use_cuda else 1,
            dp_mode="fsdp2",
            device_type=device.type,
        )
    )
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=device,
        parallel_state=state,
    )
    try:
        requested = os.environ.get("VFORGE_MMDIT_VARIANT", "both")
        variants = ("mmdit_dense", "mmdit_moe") if requested == "both" else (requested,)
        for architecture in variants:
            _run_variant(architecture, device=device, state=state)
    finally:
        reset_ops()
        destroy_parallel_state()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
