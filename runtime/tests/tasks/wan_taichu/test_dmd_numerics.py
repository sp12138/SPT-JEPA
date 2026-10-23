"""Keep standard and block DMD numerical policies independent."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vforge.algorithms.dmd.blockwise import (
    BlockDistributionMatchingRuntime,
    BlockDistributionMatchingSteps,
)
from vforge.algorithms.dmd.model import (
    DmdCriticAdapter,
    DmdNoiseLevel,
    PreparedDmdBatch,
    RoleModelBundle,
)
from vforge.tasks.wan_taichu.dmd.algorithm import WanDistributionMatchingSteps
from vforge.tasks.wan_taichu.dmd.runtime import WanDistributionMatchingRuntime
from vforge.training.recipe import StepContext
from vforge.utils.rng import RandomStreams


@pytest.mark.parametrize("precision,dtype", [("bf16", torch.bfloat16), ("fp32", torch.float32)])
def test_critic_timestep_precision_is_independent_of_fp32_weight_storage(precision, dtype):
    from vforge.tasks.requests import TrainRequest
    from vforge.tasks.wan_taichu.dmd.inputs import WanDmdInputs
    from vforge.tasks.wan_taichu.dmd.integration import create_wan_critic_adapter

    class ScoreTower(torch.nn.Module):
        config = SimpleNamespace(patch_size=(1, 2, 2))

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

        def forward(self, *, hidden_states, timestep, **kwargs):
            assert timestep.dtype == dtype
            assert torch.all(timestep == torch.tensor(883.0).to(dtype))
            return (hidden_states,)

    latent = torch.ones(1, 1, 2, 4, 4, dtype=dtype)
    prompt = torch.ones(1, 5, 4, dtype=dtype)
    batch = PreparedDmdBatch(
        clean={"latent": latent},
        model_inputs=WanDmdInputs(prompt, prompt, latent.permute(0, 2, 1, 3, 4)),
    )
    level = DmdNoiseLevel(torch.tensor(0.883), {"latent": torch.tensor(0.883)}, torch.tensor(883.0))
    adapter = create_wan_critic_adapter(TrainRequest(precision=precision), "fake_score")
    adapter.predict_velocity(ScoreTower(), batch.clean, level, batch, "conditional")


@pytest.mark.parametrize("raw_timestep", [745, 746, 747, 748, 749, 750, 751])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("conditioning", ["conditional", "unconditional"])
def test_critic_routes_before_quantizing_timestep(raw_timestep, dtype, conditioning):
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    from vforge.models.wan_taichu.model import WanTaichuModel
    from vforge.tasks.wan_taichu.dmd.critic import WanDmdCriticAdapter
    from vforge.tasks.wan_taichu.dmd.inputs import WanDmdInputs

    calls = []

    class ScoreExpert(torch.nn.Module):
        config = SimpleNamespace(patch_size=(1, 2, 2))

        def __init__(self, name):
            super().__init__()
            self.name = name
            self.weight = torch.nn.Parameter(torch.ones((), dtype=torch.float32))

        def forward(self, *, hidden_states, timestep, encoder_hidden_states, **kwargs):
            calls.append(self.name)
            assert timestep.dtype == dtype
            assert torch.all(timestep == model_timestep.to(dtype))
            assert encoder_hidden_states is (prompt if conditioning == "conditional" else negative)
            return (hidden_states * self.weight,)

    model = WanTaichuModel(
        variant="wan22_a14b",
        architecture="wan22_a14b",
        attention="bidirectional",
        transformers={name: ScoreExpert(name) for name in ("wan", "wan_2")},
        boundary_ratio=0.9,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    scheduler.set_timesteps(1000)
    model_timestep = scheduler.timesteps[1000 - raw_timestep].clone()
    original_timestep = model_timestep.clone()
    expected_expert = "wan" if raw_timestep >= 750 else "wan_2"
    if 746 <= raw_timestep <= 749:
        assert model_timestep < 900
        assert model_timestep.bfloat16() == 900

    latent = torch.ones(1, 1, 2, 4, 4, dtype=dtype)
    prompt = torch.ones(1, 5, 4, dtype=dtype)
    negative = torch.zeros_like(prompt)
    batch = PreparedDmdBatch(
        clean={"latent": latent},
        model_inputs=WanDmdInputs(prompt, negative, latent.permute(0, 2, 1, 3, 4)),
    )
    sigma = model_timestep / 1000
    level = DmdNoiseLevel(sigma, {"latent": sigma}, model_timestep)
    rng_before = torch.get_rng_state().clone()
    result = WanDmdCriticAdapter(compute_dtype=dtype).predict_velocity(
        model, batch.clean, level, batch, conditioning
    )
    assert calls == [expected_expert]
    assert torch.equal(model_timestep, original_timestep)
    assert torch.equal(torch.get_rng_state(), rng_before)
    torch.testing.assert_close(result["latent"], latent, rtol=0, atol=0)
    result["latent"].sum().backward()
    for name in model.expert_names:
        assert (getattr(model, name).weight.grad is not None) == (name == expected_expert)


def _runtime(cls):
    score = SimpleNamespace(
        num_train_timesteps=1000,
        timestep_shift=5.0,
        sampled_timestep_min=20,
        sampled_timestep_max=980,
        effective_timestep_clamp=(20, 980),
        timestep_floor_at_rollout_exit=False,
    )
    rollout = SimpleNamespace(
        denoising_steps=(1000, 700, 300, 0),
        terminal_zero_is_boundary=True,
        last_step_only=True,
        context_noise=0.0,
        context_full_trajectory=False,
    )
    family = SimpleNamespace(
        score_noise_shape=lambda _: (1, 1, 1),
        models={"fake_score": SimpleNamespace(boundary_ratio=None)},
        postprocess_score_noise=None,
        resolve_noise_level=lambda _, level, batch: DmdNoiseLevel(
            level, {key: level for key in batch.clean}, level * 1000
        ),
    )
    return cls(
        family=family,
        config=SimpleNamespace(score=score, rollout=rollout),
        seed=42,
    )


def test_score_x0_rounds_fp32_noisy_latents_before_conversion():
    noisy = {"latent": torch.tensor([1.003, -0.997], dtype=torch.float32)}
    flow = torch.tensor([0.73, -0.52], dtype=torch.bfloat16, requires_grad=True)
    sigma = torch.tensor(0.277)
    adapter = SimpleNamespace(predict_velocity=lambda *args: {"latent": flow})
    level = DmdNoiseLevel(sigma, {"latent": sigma}, sigma * 1000)

    result = WanDistributionMatchingSteps.predict_x0(
        adapter, None, noisy, level, None, "conditional"
    )["latent"]

    expected = (noisy["latent"].bfloat16().double() - sigma.double() * flow.double()).bfloat16()
    unrounded = (noisy["latent"].double() - sigma.double() * flow.double()).bfloat16()
    assert not torch.equal(expected, unrounded)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.float().sum().backward()
    assert flow.grad is not None and bool((flow.grad != 0).all())


@pytest.mark.parametrize("blockwise", [False, True])
def test_noise_preserves_global_or_named_rng_policy(blockwise):
    cls = BlockDistributionMatchingRuntime if blockwise else WanDistributionMatchingRuntime
    runtime = _runtime(cls)
    batch = PreparedDmdBatch({"video": torch.ones(1, 4, 1)}, None)
    streams = RandomStreams()
    with torch.random.fork_rng():
        torch.manual_seed(17)
        initial_rng = torch.get_rng_state().clone()
        _, _, first = runtime.sample_score_noise(batch.clean, batch, streams, 17)
        assert torch.equal(initial_rng, torch.get_rng_state()) == blockwise
        torch.manual_seed(17)
        _, _, second = runtime.sample_score_noise(batch.clean, batch, streams, 104746)
    # Wan ignores phase offsets and consumes the process RNG. Block DMD
    # uses persistent named streams without consuming the process RNG.
    assert torch.equal(first["video"], second["video"]) != blockwise


def test_wan_noise_draws_do_not_depend_on_component_insertion_order():
    runtime = _runtime(WanDistributionMatchingRuntime)
    clean = {key: torch.ones(1, 4, 1) for key in ("video", "audio")}
    batch = PreparedDmdBatch(clean, None)
    streams = RandomStreams()
    with torch.random.fork_rng():
        torch.manual_seed(17)
        first = runtime.sample_score_noise(clean, batch, streams, 17)
        torch.manual_seed(17)
        second = runtime.sample_score_noise(dict(reversed(list(clean.items()))), batch, streams, 17)
    for key in clean:
        torch.testing.assert_close(first[2][key], second[2][key], rtol=0, atol=0)


@pytest.mark.parametrize("cls", [WanDistributionMatchingRuntime, BlockDistributionMatchingRuntime])
def test_score_noise_advances_and_resumes_through_training_random_streams(cls):
    runtime = _runtime(cls)
    batch = PreparedDmdBatch({"video": torch.ones(1, 4, 1)}, None)
    streams = RandomStreams()
    with torch.random.fork_rng():
        torch.manual_seed(17)
        runtime.sample_score_noise(batch.clean, batch, streams, 17)
        saved = streams.state_dict()
        expected = runtime.sample_score_noise(batch.clean, batch, streams, 17)
        following = runtime.sample_score_noise(batch.clean, batch, streams, 17)
        assert not torch.equal(expected[2]["video"], following[2]["video"])

        resumed = RandomStreams()
        resumed.load_state_dict(saved)
        actual = _runtime(cls).sample_score_noise(batch.clean, batch, resumed, 17)
        torch.testing.assert_close(
            actual[0].model_timestep, expected[0].model_timestep, rtol=0, atol=0
        )
        for index in (1, 2):
            torch.testing.assert_close(
                actual[index]["video"], expected[index]["video"], rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "cls, expected",
    [(WanDistributionMatchingSteps, 0.734375), (BlockDistributionMatchingSteps, 0.73046875)],
)
def test_critic_preserves_x0_roundtrip_or_direct_velocity(cls, expected):
    flow = torch.tensor([0.73046875], dtype=torch.bfloat16, requires_grad=True)
    generated = {"video": torch.ones(1, dtype=torch.bfloat16, requires_grad=True)}
    prepared = PreparedDmdBatch(generated, None)
    noisy = {"video": torch.ones(1, dtype=torch.bfloat16)}
    level = DmdNoiseLevel(torch.tensor(0.277), {"video": torch.tensor(0.277)}, torch.tensor(277))
    root = RoleModelBundle({role: torch.nn.Identity() for role in ("generator", "fake_score")})

    context = StepContext(device=torch.device("cpu"), parallel_state=None)

    def rollout(generator, adapter, batch, streams, phase, seed_offset):
        assert generator is root.model_for("generator")
        assert batch is prepared
        assert streams is context.random_streams
        assert (phase, seed_offset) == ("critic", 104729)
        return generated

    def sample_score_noise(clean, batch, streams, seed_offset):
        assert not clean["video"].requires_grad
        assert batch is prepared
        assert streams is context.random_streams
        assert seed_offset == 104746
        return level, noisy, generated

    class Critic(DmdCriticAdapter):
        def predict_velocity(self, tower, values, noise_level, batch, conditioning):
            assert tower is root.model_for("fake_score")
            assert values is noisy and noise_level is level and batch is prepared
            assert conditioning == "conditional"
            return {"video": flow}

    adapter = Critic()
    data = SimpleNamespace(
        random_context=lambda *_: nullcontext(),
        prepare_batch=lambda batch, device: prepared,
        rollout=rollout,
        sample_score_noise=sample_score_noise,
        metrics=lambda noise_level, gradient: {"score_timestep": noise_level.model_timestep},
        workload=lambda clean, generator: {"local_tokens": 1},
    )
    steps = cls(
        config=None,
        data=data,
        generator_adapter=None,
        fake_score_adapter=adapter,
        real_score_adapter=adapter,
    )
    # Observe the prediction through the actual phase entry, including detach,
    # RNG offsets, conditioning and propagation of metrics/workload.
    steps.critic_loss = lambda prediction, *_: prediction["video"].sum()
    output = steps.critic_step(root, None, 3, context)
    torch.testing.assert_close(
        output.loss, torch.tensor(expected, dtype=torch.bfloat16), rtol=0, atol=0
    )
    assert output.metrics["loss_scalar"].requires_grad is False
    assert output.metrics["score_timestep"] is level.model_timestep
    assert output.workload == {"local_tokens": 1}
    output.loss.backward()
    assert flow.grad is not None and bool((flow.grad != 0).all())


def test_wan_phase_rng_matches_two_loader_seed_draws_and_resumes():
    runtime = _runtime(WanDistributionMatchingRuntime)
    streams = RandomStreams()
    expected = torch.Generator().manual_seed(42)
    for _ in range(2):
        torch.empty((), dtype=torch.int64).random_(generator=expected)
    original = torch.get_rng_state().clone()
    with runtime.random_context(streams, torch.device("cpu")):
        torch.testing.assert_close(torch.rand(8), torch.rand(8, generator=expected), rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), original)
    saved = streams.state_dict()
    with runtime.random_context(streams, torch.device("cpu")):
        following = torch.rand(8)
    restored = RandomStreams()
    restored.load_state_dict(saved)
    with _runtime(WanDistributionMatchingRuntime).random_context(restored, torch.device("cpu")):
        torch.testing.assert_close(torch.rand(8), following, rtol=0, atol=0)


def test_wan_exit_draw_consumes_one_index_per_block():
    runtime = _runtime(WanDistributionMatchingRuntime)
    runtime.config.rollout.last_step_only = False
    with torch.random.fork_rng():
        torch.manual_seed(19)
        expected = torch.randint(3, (7,))
        after = torch.get_rng_state()
        torch.manual_seed(19)
        assert runtime._rollout_exit_index(torch.device("cpu"), 7) == expected[0].item()
        assert torch.equal(torch.get_rng_state(), after)
