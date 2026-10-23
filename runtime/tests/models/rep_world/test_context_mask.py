from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from vforge.models.rep_world.conditioner import CausalViT, sample_context_indices
from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig
from vforge.ops import OpsConfig, initialize_ops, reset_ops


@pytest.fixture(autouse=True)
def ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    yield
    reset_ops()


def make_model(ratio=0.8):
    geometry = CodecGeometry("test", "tiny", 4, 3, 2, grid=4)
    config = ConditionerConfig(
        input="latent",
        width=24,
        depth=2,
        heads=4,
        feature_layers=(0, 1),
        context_mask_ratio=ratio,
    )
    return CausalViT(config, geometry)


def test_duet_per_plane_selection_and_original_indices():
    torch.manual_seed(44)
    indices = sample_context_indices(3, 9, 4, 256, 0.8, device="cpu")
    assert indices.shape == (3, 9 * 4 * 51)
    chunks = indices.reshape(3, 9, 4, 51)
    for state in range(9):
        for plane in range(4):
            assert torch.all(chunks[:, state, plane] // 256 == state * 4 + plane)
    assert torch.all(indices[:, 1:] > indices[:, :-1])
    assert not torch.equal(indices[0], indices[1])
    # Independently reproduce the original DuET rule, including its rounding.
    torch.manual_seed(44)
    scores = torch.rand(3, 9, 1024)
    expected = scores.reshape(3, 9, 4, 256).argsort(-1)[..., :51]
    expected = expected + torch.arange(4)[None, None, :, None] * 256
    expected = expected.reshape(3, 9, 204).sort(-1).values
    expected = expected + torch.arange(9)[None, :, None] * 1024
    torch.testing.assert_close(indices, expected.flatten(1), rtol=0, atol=0)


def test_masked_matches_dense_key_mask_forward_and_gradient():
    sparse = make_model()
    dense = make_model(0)
    dense.load_state_dict(sparse.state_dict(), strict=True)
    source = torch.randn(2, 6, 3, 4, 4, requires_grad=True)
    reference_source = source.detach().clone().requires_grad_(True)
    torch.manual_seed(18)
    selected = sample_context_indices(2, 3, 2, 16, 0.8, device="cpu")
    key_keep = torch.zeros(2, 96, dtype=torch.bool).scatter(1, selected, True)
    # Full sequence execution with invisible keys disabled is an independent
    # oracle for sparse execution, causal boundaries and original 3D RoPE.
    for block in dense.encoder.blocks:
        original = block.forward

        def forward(x, *, mask, positions, sequence_parallel=False, original=original):
            return original(
                x,
                mask=mask[None, None] & key_keep[:, None, None],
                positions=positions,
                sequence_parallel=sequence_parallel,
            )

        block.forward = forward
    seen = []
    handle = sparse.encoder.blocks[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].shape[1])
    )
    torch.manual_seed(18)
    actual = sparse(source)
    handle.remove()
    expected = (
        dense(reference_source)
        .flatten(1, 2)
        .gather(1, selected[..., None].expand(-1, -1, 24))
        .reshape_as(actual)
    )
    assert seen == [18] and actual.shape == (2, 3, 6, 24)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(source.grad, reference_source.grad, atol=5e-6, rtol=5e-5)
    for left, right in zip(sparse.parameters(), dense.parameters(), strict=True):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=2e-4)


def test_masked_inputs_and_future_cannot_leak():
    model = make_model()
    source = torch.randn(2, 6, 3, 4, 4, requires_grad=True)
    torch.manual_seed(19)
    selected = sample_context_indices(2, 3, 2, 16, 0.8, device="cpu")
    visible = torch.zeros(2, 96, dtype=torch.bool).scatter(1, selected, True)
    visible = visible.reshape(2, 3, 2, 4, 4).permute(0, 2, 1, 3, 4)
    visible = visible[:, :, None].expand(-1, -1, 3, -1, -1, -1).reshape_as(source)
    torch.manual_seed(19)
    expected = model(source)
    changed = source.detach().clone()
    changed[~visible] += 100
    changed[:, :, 2] += 100
    torch.manual_seed(19)
    actual = model(changed)
    torch.testing.assert_close(actual[:, :2], expected[:, :2], rtol=0, atol=0)
    expected[:, 1, :, 0].sum().backward()
    assert source.grad[~visible].count_nonzero() == 0
    assert source.grad[:, :, 2].count_nonzero() == 0
    assert source.grad[visible].abs().sum() > 0


def test_eval_dense_and_rng_and_checkpoint_compatibility():
    masked, baseline = make_model().eval(), make_model(0).eval()
    baseline.load_state_dict(masked.state_dict(), strict=True)
    source = torch.randn(2, 6, 3, 4, 4)
    before = torch.get_rng_state()
    actual = masked(source)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    assert actual.shape == (2, 3, 32, 24)
    torch.testing.assert_close(actual, baseline(source), rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [-0.1, 1.0, float("nan")])
def test_bad_mask_ratio(ratio):
    with pytest.raises(ValueError, match="context_mask_ratio"):
        make_model(ratio)


def test_latent_only_and_usp_guard():
    with pytest.raises(ValueError, match="latent"):
        replace(make_model().config, input="rgb")
    with patch("vforge.models.rep_world.conditioner.usp_state", return_value=object()):
        with pytest.raises(ValueError, match="USP=1"):
            make_model()(torch.randn(1, 6, 2, 4, 4))
