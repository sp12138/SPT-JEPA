from copy import deepcopy

import pytest
import torch
from torch import nn

from vforge.tasks.rep_world.optimizer import MuonWithAuxAdam


def reference_step(parameter, gradient, state, lr, decay):
    # Independently spell out the original normal-path equations, including
    # flattening Conv3d and using ORIGINAL first two dimensions for LR scaling.
    if parameter.ndim == 1:
        first, second, step = state
        step += 1
        first.lerp_(gradient, 0.1)
        second.lerp_(gradient.square(), 0.05)
        update = (first / (1 - 0.9**step + 1e-16)) / (
            (second / (1 - 0.95**step + 1e-16)).sqrt() + 1e-8
        )
        parameter.add_(update, alpha=-lr)
        return first, second, step
    gradient = gradient.reshape(gradient.shape[0], -1)
    state.lerp_(gradient, 1 - 0.95)
    value = gradient.lerp(state, 0.95).bfloat16()
    tall = value.shape[0] > value.shape[1]
    if tall:
        value = value.mT
    value = value / (value.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        gram = value @ value.mT
        value = 3.4445 * value + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ value
    if tall:
        value = value.mT
    value = value.float() * max(1, gradient.shape[0] / gradient.shape[1]) ** 0.5
    parameter.mul_(1 - lr * decay)
    parameter.add_(value.reshape_as(parameter), alpha=-lr * (0.2 * max(parameter.shape[:2]) ** 0.5))
    return state


@pytest.mark.parametrize("shape", [(4, 9), (9, 4), (4, 3, 2, 2, 2), (9,)])
def test_muon_original_equations_missing_gradients_and_resume(shape):
    torch.manual_seed(502)
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.randn(shape)))
    optimizer = MuonWithAuxAdam(model, lr=1e-4, weight_decay=0.03)
    expected = model.weight.detach().clone()
    reference_state = (
        (torch.zeros_like(expected), torch.zeros_like(expected), 0)
        if len(shape) == 1
        else torch.zeros(shape[0], expected.numel() // shape[0])
    )
    for step in range(5):
        gradient = torch.randn(shape) if step != 2 else torch.zeros(shape)
        model.weight.grad = gradient.clone() if step != 2 else None
        with torch.no_grad():
            reference_state = reference_step(expected, gradient, reference_state, 1e-4, 0.03)
        optimizer.step()
        torch.testing.assert_close(model.weight, expected, atol=2e-7, rtol=0)
        if step == 1:
            saved = deepcopy(optimizer.state_dict())
            optimizer = MuonWithAuxAdam(model, lr=9, weight_decay=0)
            optimizer.load_state_dict(saved)


def test_groups_leave_bias_and_norm_in_adam_and_frozen_outside_optimizer():
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    model[0].weight.requires_grad_(False)
    optimizer = MuonWithAuxAdam(model, lr=1e-4, weight_decay=0.1)
    assert optimizer.param_groups[1]["params"] == []
    assert len(optimizer.param_groups[0]["params"]) == 3
    assert optimizer.param_groups[0]["weight_decay"] == 0


def _dtensor_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        torch.manual_seed(403)
        original = nn.Sequential(nn.Conv3d(3, 5, 2), nn.Flatten(), nn.Linear(5, 7))
        sharded = deepcopy(original)
        for module in sharded.modules():
            for name, value in tuple(module.named_parameters(recurse=False)):
                module.register_parameter(
                    name, nn.Parameter(distribute_tensor(value.detach(), mesh, [Shard(0)]))
                )
        reference = MuonWithAuxAdam(original, lr=1e-4, weight_decay=0.01)
        optimizer = MuonWithAuxAdam(sharded, lr=1e-4, weight_decay=0.01)
        for step in range(3):
            for whole, shard in zip(original.parameters(), sharded.parameters(), strict=True):
                gradient = torch.randn_like(whole) if step != 1 else None
                whole.grad = gradient
                shard.grad = (
                    distribute_tensor(gradient, mesh, [Shard(0)]) if gradient is not None else None
                )
            reference.step()
            optimizer.step()
            for whole, shard in zip(original.parameters(), sharded.parameters(), strict=True):
                torch.testing.assert_close(shard.full_tensor(), whole, atol=0, rtol=0)
                for key, value in reference.state[whole].items():
                    actual = optimizer.state[shard][key]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(actual.full_tensor(), value, atol=0, rtol=0)
                    else:
                        assert actual == value
    finally:
        dist.destroy_process_group()


def test_dtensor_muon_orthogonalizes_full_matrix_and_keeps_sharded_state(tmp_path):
    torch.multiprocessing.spawn(
        _dtensor_worker, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True
    )
