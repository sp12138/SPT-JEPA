"""Local grouped-MM gate before the multi-rank FSDP2 + EP experiments."""

import pytest
import torch
from expert_parallel_models import ReferenceExperts

from vforge.distributed.expert_parallel import Expert


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("projection_order", ["gate_up", "up_gate"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "activation_fn",
    [torch.nn.functional.silu, torch.nn.functional.gelu, torch.sigmoid],
    ids=["swiglu", "geglu", "glu"],
)
@pytest.mark.parametrize("route", ["mixed", "skew", "empty", "zero_assignments", "frozen_input"])
def test_grouped_expert_dispatch_matches_dense(
    route, activation_fn, projection_order, input_dtype, monkeypatch
):
    assert torch.__version__.split("+")[0].startswith("2.13."), torch.__version__
    if route == "mixed":
        print(f"Grouped MM runtime: torch={torch.__version__}, GPU={torch.cuda.get_device_name()}")
    torch.manual_seed(42)
    reference = ReferenceExperts(
        4, 16, 32, init_std=0.1, activation_fn=activation_fn, projection_order=projection_order
    ).to(device="cuda", dtype=torch.bfloat16)
    grouped = Expert(4, 16, 32, activation_fn=activation_fn, projection_order=projection_order).to(
        device="cuda", dtype=torch.bfloat16
    )
    grouped.load_state_dict(reference.state_dict())
    ids = torch.tensor([[3, 0, 3], [1, 0, 1], [3, 3, 0], [0, 1, 3]], device="cuda")
    if route == "skew":
        ids.zero_()
    elif route == "empty":
        ids = ids[:0]
    elif route == "zero_assignments":
        ids = ids[:, :0]
    inputs = torch.randn(len(ids), 16, device="cuda", dtype=input_dtype)
    inputs.requires_grad_(route != "frozen_input")
    expected_inputs = inputs.detach().clone().requires_grad_(inputs.requires_grad)
    # Deliberately unnormalized, with duplicate assignments to the same expert.
    weights = torch.randn(ids.shape, device="cuda", requires_grad=True)
    expected_weights = weights.detach().clone().requires_grad_()

    calls = []
    grouped_mm = torch.nn.functional.grouped_mm

    def record_grouped_mm(a, b, **kwargs):
        calls.append((a.shape, b.shape))
        return grouped_mm(a, b, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "grouped_mm", record_grouped_mm)
    output = grouped.dispatch(inputs, ids, weights)
    top_k = ids.shape[1]
    assignments = reference(
        expected_inputs.repeat_interleave(top_k, dim=0).to(torch.bfloat16), ids.reshape(-1)
    )
    expected = (
        (assignments.reshape(len(ids), top_k, 16) * expected_weights.unsqueeze(-1))
        .to(inputs.dtype)
        .sum(1)
    )
    assert len(calls) == (2 if ids.numel() else 0)
    assert output.dtype == input_dtype
    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-3)
    # Distinct assignment gradients make a missing inverse permutation visible.
    grad = torch.randn_like(output)
    output.backward(grad)
    expected.backward(grad)
    if inputs.requires_grad:
        torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=5e-2, atol=5e-3)
    torch.testing.assert_close(weights.grad, expected_weights.grad, rtol=5e-2, atol=5e-3)
    unhit = torch.bincount(ids.reshape(-1), minlength=4) == 0
    for actual, baseline in zip(grouped.parameters(), reference.parameters(), strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, baseline.grad, rtol=5e-2, atol=5e-3)
        assert torch.count_nonzero(actual.grad[unhit]).item() == 0
    for model in (grouped, reference):
        torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1, foreach=False).step()
    for actual, baseline in zip(grouped.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, baseline, rtol=5e-2, atol=5e-3)


@pytest.mark.gpu(gpus=1)
def test_fp16_weights_are_rejected_by_grouped_mm_backend():
    experts = Expert(4, 16, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="CUDA BF16 weights"):
        experts.dispatch(
            torch.ones(1, 16, device="cuda", dtype=torch.float16),
            torch.zeros(1, 1, device="cuda", dtype=torch.long),
            torch.ones(1, 1, device="cuda"),
            cast_routing_weights_to_output=True,
        )


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("expert_id", [-1, 4])
def test_invalid_local_expert_id_is_rejected(expert_id):
    experts = Expert(4, 16, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="local expert ID"):
        experts(
            torch.zeros(1, 16, device="cuda", dtype=torch.bfloat16),
            torch.tensor([expert_id], device="cuda"),
        )
