"""Numerical and layout contracts for ordinary FSDP2 plus expert parallelism."""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from expert_parallel_models import MoE, ReferenceExperts, SequenceMoE
from torch.distributed.tensor import DTensor, Shard

from vforge.distributed.expert_parallel import (
    Expert,
    ExpertLayout,
)
from vforge.distributed.parallel_state import (
    ParallelConfig,
    get_parallel_state,
)
from vforge.distributed.parallelize import (
    FSDP2Config,
    MixedPrecisionConfig,
    ParallelizeConfig,
    parallelize_model,
)
from vforge.ops import initialize_ops
from vforge.ops.config import OpsConfig
from vforge.training.diagnostics import clip_grad_norm_logical_


@pytest.mark.parametrize("experts,ep", [(400, 20), (400, 40), (8, 2), (8, 4), (1, 1)])
def test_equal_contiguous_layout(experts, ep):
    layouts = [ExpertLayout(experts, ep, rank) for rank in range(ep)]
    assert sum(layout.count for layout in layouts) == experts
    assert all(layout.count == experts // ep for layout in layouts)
    assert [
        i for layout in layouts for i in range(layout.start, layout.start + layout.count)
    ] == list(range(experts))


@pytest.mark.parametrize("experts,ep", [(400, 32), (5, 2), (2, 4)])
def test_uneven_expert_layout_is_rejected(experts, ep):
    with pytest.raises(ValueError, match="expert count must be divisible"):
        ExpertLayout(experts, ep, 0)


def run_numerical_cases(
    rank,
    world,
    *,
    device,
    ep=2,
    usp=1,
    dtype=torch.float32,
    amp=False,
    recompute=False,
    progress=lambda message: None,
):
    rtol, atol = (5e-2, 5e-3) if amp else (1e-4, 1e-5)
    input_dtype = torch.bfloat16 if amp else dtype
    precision = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else nullcontext
    for main, block, route in [
        (8, 0, "normal"),
        (8, 2, "normal"),
        (8, 2, "skew"),
        (ep, 2, "normal"),
        (8, 2, "empty"),
        (8, 2, "empty_rank"),
        (8, 2, "frozen_input"),
        (8, 2, "zero_assignments"),
    ]:
        progress(f"numerical main={main} block={block} route={route}")
        torch.manual_seed(42)
        model_cls = SequenceMoE if usp > 1 else MoE
        reference = model_cls(
            main=main, block=block, intermediate=16 if amp else 12, experts_cls=ReferenceExperts
        ).to(device=device, dtype=dtype)
        if usp > 1:
            reference.sequence_parallel = False
        # The public grouped-MM module is CUDA BF16; other precision gates
        # exercise the EP/FSDP contract with the independent dense bank.
        candidate = model_cls(
            main=main,
            block=block,
            intermediate=16 if amp else 12,
            experts_cls=Expert if amp else ReferenceExperts,
        ).to(device=device, dtype=dtype)
        candidate.load_state_dict(reference.state_dict())
        model = parallelize_model(
            candidate,
            ParallelizeConfig(
                parallel=ParallelConfig(
                    dp_size=world // usp,
                    usp_size=usp,
                    extra_parallel_sizes=(ep,),
                    device_type=device.type,
                ),
                fsdp2=FSDP2Config(mixed_precision=MixedPrecisionConfig(enable=amp)),
                gradient_checkpointing=recompute,
            ),
        )
        # BF16 GEMM grouping changes rounding of near-zero gradients. Use a
        # training-scale learning rate for the update comparison.
        learning_rate = 0.001
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.1, foreach=False, fused=False
        )
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=learning_rate, weight_decay=0.1, foreach=False, fused=False
        )
        # Different FSDP meshes do not require different optimizers or groups.
        assert len(optimizer.param_groups) == 1
        state = get_parallel_state()
        if usp > 1:
            initialize_ops(
                OpsConfig(backend="sdpa_math", allow_fallback=False, deterministic=True),
                device=device,
                parallel_state=state,
            )
            assert state.usp_rank == rank % usp
            assert state.dp_rank == rank // usp
            assert state.batch_group is state.usp_group
            assert dist.get_process_group_ranks(state.ep_group) == list(
                range(rank // ep * ep, (rank // ep + 1) * ep)
            )
            assert dist.get_process_group_ranks(state.ep_fsdp_mesh.get_group()) == list(
                range(rank % ep, world, ep)
            )
        assert state.ep_rank == rank % ep
        assert state.ep_fsdp_mesh.size() == world // ep
        if ep > 1:
            for name, parameter in model.main.named_parameters():
                assert isinstance(parameter, DTensor)
                assert parameter.shape == dict(reference.main.named_parameters())[name].shape
                assert parameter.device_mesh.mesh_dim_names == ("ep_fsdp", "ep")
                assert parameter.placements == (Shard(1), Shard(0))
                assert parameter.to_local().shape[0] == main // ep
                torch.testing.assert_close(
                    parameter.full_tensor(), dict(reference.main.named_parameters())[name]
                )
        if model.block is not None:
            assert not hasattr(model.block, "_vforge_expert_layout")
            parameters = optimizer.param_groups[0]["params"]
            for parameter in model.block.parameters():
                assert parameter.shape[0] == block
                assert parameter.device_mesh.size() == world
                assert any(parameter is value for value in parameters)
        for step in range(2):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            torch.manual_seed(42 + step)
            shape = (world // usp, 1, 3 * usp, 8) if usp > 1 else (world, 3, 8)
            full_inputs = torch.randn(shape, device=device, dtype=input_dtype)
            source_rank = state.dp_rank
            inputs = full_inputs[source_rank].clone().requires_grad_()
            if route == "frozen_input":
                inputs = inputs.detach()
            if route == "empty" or (route == "empty_rank" and source_rank == 0):
                inputs = inputs[:0].detach().requires_grad_()
            with precision():
                output = model(inputs, route)
            loss = output.square().sum()
            loss.backward()
            for source in range(world // usp):
                source_input = full_inputs[source].clone().requires_grad_()
                if route == "empty" or (route == "empty_rank" and source == 0):
                    source_input = source_input[:0].detach().requires_grad_()
                with precision():
                    expected = reference(source_input, route)
                (expected.square().sum() / (world // usp)).backward()
                if source == source_rank:
                    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
                    if route != "frozen_input":
                        torch.testing.assert_close(
                            inputs.grad, source_input.grad * (world // usp), rtol=rtol, atol=atol
                        )
            for name, parameter in model.named_parameters():
                expected = dict(reference.named_parameters())[name].grad
                actual = (
                    parameter.grad.full_tensor()
                    if hasattr(parameter.grad, "full_tensor")
                    else parameter.grad
                )
                if amp and actual.numel():
                    changed_sign = (actual * expected < 0) & (expected.abs() < 1e-6)
                    if changed_sign.any():
                        positions = changed_sign.flatten().nonzero().flatten()
                        index = positions[(actual - expected).abs().flatten()[positions].argmax()]
                        progress(
                            f"BF16 small-gradient rounding {name}: "
                            f"actual={actual.flatten()[index].item():.9g} "
                            f"reference={expected.flatten()[index].item():.9g} "
                            f"lr={learning_rate}"
                        )
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    msg=lambda message, route=route, name=name: f"{route} {name}: {message}",
                )
            actual_norm = clip_grad_norm_logical_(model, 0.05)
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.05)
            assert actual_norm == pytest.approx(expected_norm.item(), rel=rtol, abs=atol)
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                expected = dict(reference.named_parameters())[name]
                torch.testing.assert_close(
                    parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    msg=lambda message, route=route, name=name, step=step: (
                        f"update {step} {route} {name}: {message}"
                    ),
                )
