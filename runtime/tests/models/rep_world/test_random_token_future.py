"""Balanced token future targets preserve position, source identity and exact budget."""

import pytest
import torch

from vforge.models.rep_world.horizon import sample_token_offsets, token_future_targets


def test_exact_half_independent_origins_planes_and_rng_replay():
    torch.manual_seed(53)
    state = torch.get_rng_state()
    assignments = sample_token_offsets((3, 8), 1000, "cpu")
    assert assignments.sum(-1).eq(500).all()
    assert assignments.flatten(0, 1).unique(dim=0).shape[0] == 24
    assert not torch.equal(assignments[..., :500], assignments[..., 500:])
    torch.set_rng_state(state)
    assert torch.equal(assignments, sample_token_offsets((3, 8), 1000, "cpu"))
    rng = torch.get_rng_state()
    assert torch.equal(assignments, sample_token_offsets((3, 8), 1000, "cpu", offsets=assignments))
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("patch", [1, 2])
def test_source_state_channels_planes_and_patch_coordinates(patch):
    b, s, c, p, h, w = 2, 5, 3, 4, 4, 8
    states = torch.arange(b * s * c * p * h * w).reshape(b, s, c, p, h, w)
    offsets = sample_token_offsets((b, s - 2), p * (h // patch) * (w // patch), "cpu")
    actual, tokens, pixels = token_future_targets(states, offsets, patch)
    expected = torch.empty_like(actual)
    for batch in range(b):
        for origin in range(s - 2):
            for plane in range(p):
                for y in range(h):
                    for x in range(w):
                        index = (
                            plane * (h // patch) * (w // patch)
                            + (y // patch) * (w // patch)
                            + x // patch
                        )
                        offset = offsets[batch, origin, index].item()
                        expected[batch, origin, :, plane, y, x] = states[
                            batch, origin + 1 + offset, :, plane, y, x
                        ]
                        assert pixels[batch * (s - 2) + origin, plane, y, x] == offset
    assert torch.equal(actual, expected)
    assert torch.equal(tokens, offsets.flatten(0, 1))
    assert tokens.sum(-1).eq(tokens.shape[-1] // 2).all()


def test_reject_unbalanced_or_invalid_assignments():
    with pytest.raises(ValueError, match="even"):
        sample_token_offsets((1, 2), 9, "cpu")
    with pytest.raises(ValueError, match="exactly half"):
        sample_token_offsets((1, 2), 8, "cpu", offsets=torch.zeros(1, 2, 8, dtype=torch.long))
    with pytest.raises(ValueError, match="exactly half"):
        sample_token_offsets((1, 2), 8, "cpu", offsets=torch.full((1, 2, 8), 2, dtype=torch.long))
    assignment = sample_token_offsets((1, 2), 16, "cpu")
    with pytest.raises(ValueError, match="S-2"):
        token_future_targets(torch.zeros(1, 3, 2, 4, 2, 2), assignment)
    with pytest.raises(ValueError, match="divide"):
        token_future_targets(torch.zeros(1, 4, 2, 4, 3, 2), assignment, patch_size=2)
