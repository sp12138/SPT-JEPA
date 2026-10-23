import random

import numpy as np
import pytest
import torch

from vforge.utils.rng import RandomStreams, seed_process


def test_seed_process_replays_a_continuous_sequence():
    seed_process(42)
    expected = [(random.random(), np.random.rand(), torch.rand(8)) for _ in range(3)]
    assert not torch.equal(expected[0][2], expected[1][2])
    seed_process(42)
    for python, numpy, tensor in expected:
        assert random.random() == python
        assert np.random.rand() == numpy
        assert torch.equal(torch.rand(8), tensor)


def test_named_streams_advance_independently_and_restore_with_process_rngs():
    streams = RandomStreams()
    device = torch.device("cpu")
    noise = streams.generator("noise", seed=42, device=device)
    lengths = streams.generator("length", seed=42, device=device)
    torch.randn(64, generator=noise)
    saved = streams.state_dict()
    expected = (random.random(), np.random.rand(), torch.rand(8))
    expected_noise = torch.randn(16, generator=noise)
    expected_lengths = torch.randint(1, 100, (16,), generator=lengths)

    # A fresh run restores streams even before a task first asks for them.
    resumed = RandomStreams()
    resumed.load_state_dict(saved)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(8), expected[2])
    restored_noise = resumed.generator("noise", seed=42, device=device)
    restored_lengths = resumed.generator("length", seed=42, device=device)
    assert restored_noise is resumed.generator("noise", seed=42, device=device)
    assert torch.equal(torch.randn(16, generator=restored_noise), expected_noise)
    assert torch.equal(torch.randint(1, 100, (16,), generator=restored_lengths), expected_lengths)
    # Noise consumption did not affect the separate length stream.
    reference = torch.Generator().manual_seed(42)
    assert torch.equal(torch.randint(1, 100, (16,), generator=reference), expected_lengths)

    streams.load_state_dict(saved)
    assert streams.generator("noise", seed=42, device=device) is noise
    assert torch.equal(torch.randn(16, generator=noise), expected_noise)


@pytest.mark.gpu(gpus=1)
def test_cuda_stream_restores_on_the_current_device():
    device = torch.device("cuda", torch.cuda.current_device())
    streams = RandomStreams()
    generator = streams.generator("input", seed=42, device=device)
    torch.randn(16, device=device, generator=generator)
    saved = streams.state_dict()
    expected = torch.randn(16, device=device, generator=generator)
    expected_global = torch.randn(16, device=device)

    resumed = RandomStreams()
    resumed.load_state_dict(saved)
    generator = resumed.generator("input", seed=42, device=device)
    assert generator.device == device
    assert torch.equal(torch.randn(16, device=device, generator=generator), expected)
    assert torch.equal(torch.randn(16, device=device), expected_global)
