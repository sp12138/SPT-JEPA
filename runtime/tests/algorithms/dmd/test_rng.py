"""DMD input sampling shares the training run's resumable random streams."""

from types import SimpleNamespace

import torch

from vforge.algorithms.dmd.model import DmdNoiseLevel, PreparedDmdBatch
from vforge.algorithms.dmd.runtime import DistributionMatchingRuntime
from vforge.utils.rng import RandomStreams


def test_score_noise_advances_and_restores_without_an_update_index():
    runtime = DistributionMatchingRuntime(
        family=SimpleNamespace(
            score_noise_shape=lambda _: (1,),
            resolve_noise_level=lambda scheduler, value, prepared: DmdNoiseLevel(
                value, {"video": value}, value
            ),
            postprocess_score_noise=None,
        ),
        config=SimpleNamespace(
            score=SimpleNamespace(
                num_train_timesteps=1000,
                timestep_shift=1.0,
                sampled_timestep_min=0,
                effective_timestep_clamp=(0, 999),
            ),
            rollout=SimpleNamespace(
                denoising_steps=(1000, 500, 0),
                terminal_zero_is_boundary=True,
                context_noise=0.0,
                last_step_only=True,
            ),
        ),
        seed=42,
    )
    prepared = PreparedDmdBatch({"video": torch.ones(1, 4, 8)}, None)
    streams = RandomStreams()
    runtime.sample_score_noise(prepared.clean, prepared, streams, 17)
    saved = streams.state_dict()
    expected_level, expected_noisy, expected_noise = runtime.sample_score_noise(
        prepared.clean, prepared, streams, 17
    )
    _, _, following = runtime.sample_score_noise(prepared.clean, prepared, streams, 17)
    assert not torch.equal(following["video"], expected_noise["video"])

    resumed = RandomStreams()
    resumed.load_state_dict(saved)
    level, noisy, noise = runtime.sample_score_noise(prepared.clean, prepared, resumed, 17)
    assert torch.equal(level.normalized, expected_level.normalized)
    assert torch.equal(noisy["video"], expected_noisy["video"])
    assert torch.equal(noise["video"], expected_noise["video"])
