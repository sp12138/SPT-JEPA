import torch

from vforge.tasks.rep_world.train import random_nine_state_view


def test_random_nine_state_view_uses_only_consecutive_windows():
    latents = torch.arange(2 * 3 * 10).reshape(2, 3, 10, 1, 1).float()
    view, starts = random_nine_state_view(latents, torch.tensor([0, 1]))
    assert starts.tolist() == [0, 1]
    assert torch.equal(view[0], latents[0, :, :9])
    assert torch.equal(view[1], latents[1, :, 1:])


def test_random_nine_state_view_samples_both_starts():
    torch.manual_seed(123)
    latents = torch.zeros(128, 1, 10, 1, 1)
    _, starts = random_nine_state_view(latents)
    assert set(starts.tolist()) == {0, 1}
    assert 40 <= int(starts.sum()) <= 88
