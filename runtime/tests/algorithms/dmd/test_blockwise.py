"""DMD arithmetic, replay updates and shared rollout contracts."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from vforge.algorithms.dmd.blockwise import (
    BlockDistributionMatchingRuntime,
    BlockDistributionMatchingSteps,
)
from vforge.algorithms.dmd.config import BlockDistributionMatchingConfig
from vforge.algorithms.dmd.model import (
    DistributionMatchingData,
    DmdCriticAdapter,
    DmdFamilyComponents,
    DmdGeneratorAdapter,
    DmdNoiseLevel,
    DmdRolloutBlock,
    PreparedDmdBatch,
    RoleModelBundle,
)
from vforge.algorithms.dmd.runtime import DmdRolloutSchedule, rollout_generator
from vforge.training.diagnostics import clip_and_step, run_backward
from vforge.utils.rng import RandomStreams


def test_surrogate_weights_blocks_equally_and_detaches_targets():
    video = torch.zeros(1, 4, 1, requires_grad=True)
    gradient = torch.tensor([[[0.0], [2.0], [2.0], [2.0]]], requires_grad=True)
    loss = BlockDistributionMatchingSteps._dmd_surrogate_loss_bundle(
        {"video": video},
        {"video": gradient},
        partitions=[{"video": slice(0, 1)}, {"video": slice(1, 4)}],
    )
    # Equal block means: (0 + 0.5 * 4) / 2 = 1, not the token-weighted 1.5.
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    torch.testing.assert_close(video.grad, torch.tensor([[[0.0], [1 / 3], [1 / 3], [1 / 3]]]))
    assert gradient.grad is None


def test_critic_uses_noise_minus_clean_and_both_modalities():
    dtype = torch.bfloat16
    video = torch.tensor([[[1.0]]], dtype=dtype, requires_grad=True)
    audio = torch.tensor([[[2.0]]], dtype=dtype, requires_grad=True)
    loss = BlockDistributionMatchingSteps._flow_critic_loss_bundle(
        {"video": video, "audio": audio},
        {"video": torch.ones_like(video), "audio": torch.full_like(audio, 2)},
        {"video": torch.full_like(video, 4), "audio": torch.ones_like(audio)},
        component_weights={"video": 1, "audio": 2},
    )
    # Video target 3, audio target -1: (-2)^2 + 2 * 3^2 = 22.
    torch.testing.assert_close(loss, torch.tensor(22.0, dtype=dtype))
    loss.backward()
    torch.testing.assert_close(video.grad, torch.full_like(video, -4))
    torch.testing.assert_close(audio.grad, torch.full_like(audio, 12))


def test_score_guidance_scales_are_per_modality_and_detached():
    clean = {key: torch.zeros(1, 1, 1, requires_grad=True) for key in ("video", "audio")}
    gradient = BlockDistributionMatchingSteps._dmd_score_gradient_bundle(
        clean,
        clean,
        None,
        fake_score=lambda *_: {key: torch.full_like(value, 4) for key, value in clean.items()},
        real_cond_score=lambda *_: {key: torch.full_like(value, 3) for key, value in clean.items()},
        real_uncond_score=lambda *_: {key: torch.ones_like(value) for key, value in clean.items()},
        component_guidance_scales={"video": 3, "audio": 5},
    )
    # Guided scores 7 and 11; normalize fake-real by |clean-real|.
    for key, expected in (("video", -3 / 7), ("audio", -7 / 11)):
        torch.testing.assert_close(gradient[key], torch.full_like(clean[key], expected))
        assert not gradient[key].requires_grad


class ToyGenerator(DmdGeneratorAdapter):
    def __init__(self, anchor):
        self.anchor = anchor
        self.closed = 0
        self.events = []

    @contextmanager
    def open_rollout_context(self, generator, batch):
        try:
            yield {key: 0.0 for key in batch.clean}
        finally:
            self.closed += 1

    def block_layout(self, batch, initial_noise):
        return [
            DmdRolloutBlock(
                region,
                {k: v[:, region] for k, v in initial_noise.items()},
                is_anchor=self.anchor and index == 0,
            )
            for index, region in enumerate((slice(0, 1), slice(1, 2), slice(2, 4)))
        ]

    def denoise_block(self, generator, noisy_block, selection, noise_level, batch, context):
        self.events.append(("forward", selection.start, torch.is_grad_enabled()))
        return {
            key: value * generator.weight + 0.1 * context[key] + noise_level.sigmas[key] * 0.2
            for key, value in noisy_block.items()
        }

    def commit_block(self, generator, selection, context_value, noise_level, batch, context):
        assert not torch.is_grad_enabled()
        for key, value in context_value.items():
            assert not value.requires_grad
            context[key] = context[key] + value.sum()
        self.events.append(("commit", selection.start, False))

    def concat_blocks(self, outputs, batch):
        return {key: torch.cat([value[key] for _, value in outputs], dim=1) for key in batch.clean}


class ToyCritic(DmdCriticAdapter):
    def predict_velocity(self, tower, noisy, noise_level, batch, conditioning):
        return {
            key: value * tower.weight + (0.2 if conditioning == "conditional" else 0.1)
            for key, value in noisy.items()
        }


def _run(mode, anchor):
    config = BlockDistributionMatchingConfig.from_mapping(
        {
            "algorithm": "block_distribution_matching",
            "rollout": {
                "denoising_steps": [1000, 700, 300, 0],
                "same_step_across_blocks": False,
                "backward_mode": mode,
            },
            "score": {
                "num_train_timesteps": 1000,
                "sampled_timestep_min": 20,
                "sampled_timestep_max": 980,
                "component_guidance_scales": {"video": 3, "audio": 5},
                "component_weights": {"video": 1, "audio": 2},
            },
            "schedule": {"generator_every_n_steps": 5},
            "ema": {"enabled": False, "decay": 0.99, "start_step": 0, "storage": "sharded_fp32"},
        }
    )
    models = {
        role: torch.nn.Linear(1, 1, bias=False)
        for role in ("generator", "real_score", "fake_score")
    }
    for index, model in enumerate(models.values()):
        model.weight.data.fill_(0.3 + index * 0.2)
    root = RoleModelBundle(models, frozen_roles={"real_score"})
    generator = ToyGenerator(anchor)
    critic = ToyCritic()
    batch = PreparedDmdBatch(
        clean={key: torch.ones(1, 4, 1) for key in ("video", "audio")}, model_inputs=None
    )

    def resolve(_scheduler, level, prepared):
        return DmdNoiseLevel(level, {key: level for key in prepared.clean}, level)

    def partitions(_batch):
        return [
            dict.fromkeys(batch.clean, region) for region in (slice(0, 1), slice(1, 2), slice(2, 4))
        ]

    family = DmdFamilyComponents(
        models=models,
        build_train_stream=None,
        prepare_batch=lambda batch, device: batch,
        generator_adapter=generator,
        critic_adapters={"real_score": critic, "fake_score": critic},
        pretrained_weights=(),
        resolve_noise_level=resolve,
        score_noise_shape=lambda _: (1, 1, 1),
        valid_masks=lambda _: None,
        loss_partitions=partitions,
        resolve_rollout_noise_level=resolve,
    )
    runtime = BlockDistributionMatchingRuntime(family=family, config=config, seed=42)
    data = DistributionMatchingData(
        dataloader=[],
        prepare_batch=family.prepare_batch,
        rollout=runtime.rollout,
        sample_score_noise=runtime.sample_score_noise,
        valid_masks=family.valid_masks,
        batch_selector=lambda batch, phase: batch,
        loss_partitions=partitions,
        rollout_trace=runtime.rollout_trace,
    )
    steps = BlockDistributionMatchingSteps(
        config=config,
        data=data,
        generator_adapter=generator,
        real_score_adapter=critic,
        fake_score_adapter=critic,
    )
    context = SimpleNamespace(device=torch.device("cpu"), random_streams=RandomStreams())
    output = steps.generator_step(root, batch, 5, context)
    before = output.loss.detach().clone()
    optimizer = torch.optim.SGD(models["generator"].parameters(), lr=0.1)
    run_backward(
        loss=output.loss,
        backward_losses=output.backward_losses,
    )
    clip_and_step(
        model=models["generator"],
        optimizer=optimizer,
        scheduler=None,
        usp_size=1,
        gradient_clip_val=0,
        parity=None,
    )
    torch.testing.assert_close(output.loss, before)
    assert models["real_score"].weight.grad is None
    assert models["fake_score"].weight.grad is None
    assert generator.closed == (2 if mode == "block_replay" else 1)
    return (
        output.loss.detach(),
        models["generator"].weight.grad,
        models["generator"].weight.detach(),
    )


@pytest.mark.parametrize("anchor", [False, True])
def test_replay_matches_retained_loss_gradient_and_update(anchor):
    retained = _run("retained_graph", anchor)
    replay = _run("block_replay", anchor)
    for actual, expected in zip(replay, retained, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert replay[1].abs().sum() > 0


@pytest.mark.parametrize("last_step_only", [False, True])
@pytest.mark.parametrize("clean_context", [False, True])
def test_shared_rollout_keeps_gradient_scope_and_history_noise(last_step_only, clean_context):
    batch = PreparedDmdBatch({"video": torch.ones(1, 2, 1)}, None)
    model = torch.nn.Linear(1, 1, bias=False)
    forwards, commits, noises = [], [], []

    def denoise(model, values, selection, level, batch, context):
        forwards.append((selection, level.normalized.item(), torch.is_grad_enabled()))
        return {"video": values["video"] * model.weight}

    def commit(model, selection, values, level, batch, context):
        assert not torch.is_grad_enabled()
        assert not values["video"].requires_grad
        commits.append(selection)

    def noise(value):
        noises.append(value.shape)
        return torch.zeros_like(value)

    adapter = SimpleNamespace(
        open_rollout_context=lambda *_: nullcontext(),
        block_layout=lambda batch, values: [
            DmdRolloutBlock(index, {"video": value})
            for index, value in enumerate(values["video"].split(1, dim=1))
        ],
        denoise_block=denoise,
        commit_block=commit,
        concat_blocks=lambda outputs, _: {"video": torch.cat([v["video"] for _, v in outputs], 1)},
    )
    generated = rollout_generator(
        model,
        adapter,
        batch,
        batch.clean,
        resolve_noise_level=lambda level, _: DmdNoiseLevel(level, {"video": level}, level),
        schedule=DmdRolloutSchedule((1.0, 0.5, 0.0), last_step_only=last_step_only),
        phase="generator",
        noise_fn=noise,
        clean_context=clean_context,
    )
    assert forwards == [
        (0, 1.0, not last_step_only),
        (0, 0.5, True),
        (1, 1.0, not last_step_only),
        (1, 0.5, True),
    ]
    assert commits == [0, 1]
    # Standard history still draws noise at sigma=0; clean history must not consume RNG.
    assert len(noises) == (2 if clean_context else 4)
    generated["video"].sum().backward()
    assert model.weight.grad is not None


def test_noising_keeps_native_promotion_and_explicit_precision_boundary():
    from vforge.algorithms.dmd.runtime import add_noise_bundle

    clean = {"video": torch.ones(1, dtype=torch.bfloat16)}
    sigmas = {"video": torch.tensor(0.3)}

    def noise_fn(value):
        return torch.full(value.shape, 2.0, dtype=torch.float32)

    native = add_noise_bundle(clean, sigmas, noise_fn=noise_fn)
    aligned = add_noise_bundle(clean, sigmas, noise_fn=noise_fn, compute_dtype=torch.float32)
    # Native BF16 sigma is 0.30078125; adding FP32 noise preserves type promotion.
    torch.testing.assert_close(native["video"], torch.tensor([1.30078125]))
    # Explicit FP32 noising rounds 1.3 back to the latent's BF16 representation.
    torch.testing.assert_close(aligned["video"], torch.tensor([1.296875], dtype=torch.bfloat16))


def test_segment_chain_commits_generator_once_and_restores_mid_chain():
    from copy import deepcopy

    from torchdata.stateful_dataloader import StatefulDataLoader

    from vforge.algorithms.dmd.config import SegmentChainConfig
    from vforge.algorithms.dmd.segment_chain import SegmentChainDataLoader

    class CarryAdapter:
        def prepare_segment(self, sample, carry):
            return sample, carry

        def advance_segment(self, generated, prepared, carry):
            return {"value": generated["video"] + (0 if carry is None else carry["value"])}

    config = SegmentChainConfig(segment_slots=4, max_rollout_blocks=6, local_blocks=1)

    def make():
        return SegmentChainDataLoader(
            StatefulDataLoader([torch.tensor(11), torch.tensor(22)], batch_size=None),
            config,
            CarryAdapter(),
        )

    loader = make()
    assert next(loader).item() == 11
    g = torch.tensor([2.0], requires_grad=True)
    loader.retain({"video": g}, None, "generator")
    loader.retain({"video": torch.tensor([99.0])}, None, "critic")
    with pytest.raises(RuntimeError, match="committed"):
        loader.state_dict()
    loader.commit(None, 0)
    assert loader.segment == 1
    assert loader.carry["value"].item() == 2
    assert not loader.carry["value"].requires_grad
    snapshot = loader.state_dict()
    saved = deepcopy(snapshot)
    resumed = make()
    resumed.load_state_dict(snapshot)
    for active in (loader, resumed):
        assert next(active).item() == 11
        assert active.prepare(None)[1]["value"].item() == 2
        active.retain({"video": torch.tensor([3.0])}, None, "critic")
        active.commit(None, 1)
        assert active.carry["value"].item() == 5
        with pytest.raises(RuntimeError, match="no completed"):
            active.commit(None, 1)
        assert next(active).item() == 22
        assert active.carry is None
        assert active.segment == 0
    assert torch.equal(snapshot["carry"]["value"], saved["carry"]["value"])


def test_long_tuning_compose_and_round_up_chain():
    from vforge.algorithms.dmd import create_algorithm
    from vforge.algorithms.dmd.config import SegmentChainDistributionMatchingConfig
    from vforge.tasks.common import compose_standalone_config

    cfg = compose_standalone_config("train", overrides=["experiment=ltx23_long_tuning"])
    dmd = compose_standalone_config("train", overrides=["experiment=ltx23_dmd"])
    assert cfg.runtime.autocast is dmd.runtime.autocast is False
    assert cfg.parallel.mixed_precision == dmd.parallel.mixed_precision
    parsed = SegmentChainDistributionMatchingConfig.from_mapping(cfg.training.dmd)
    assert parsed.chain.segments_per_sample == 9
    assert parsed.chain.actual_rollout_blocks == 57
    assert cfg.training.models.real_score.init_weights == ""
    assert create_algorithm(parsed_algorithm := cfg.training.dmd.algorithm).name == parsed_algorithm
    cfg.training.dmd.chain.max_rollout_blocks = 58
    rounded = SegmentChainDistributionMatchingConfig.from_mapping(cfg.training.dmd)
    assert rounded.chain.segments_per_sample == 10
    assert rounded.chain.actual_rollout_blocks == 63


def test_long_tuning_single_frame_and_full_cache_produce_same_conditioning():
    from dataclasses import fields

    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.ltx.dmd.inputs import prepare_batch

    cfg = compose_standalone_config("train", overrides=["experiment=ltx23_long_tuning"])
    latent = torch.randn(128, 30, 2, 3)
    condition = {
        "video_prompt_embeds": torch.ones(4, 8),
        "prompt_attention_mask": torch.ones(4),
    }
    connectors = SimpleNamespace(create_embeddings=lambda video, audio, mask: (video, audio, mask))
    batches = [
        prepare_batch(
            {"video_latents": {"latents": latent[:, :frames], "fps": 24}, "conditions": condition},
            torch.device("cpu"),
            config=cfg.training.models.generator.config,
            connectors=connectors,
            negative={},
            segment_slots=9,
        )
        for frames in (1, 30)
    ]
    first, full = (batch.model_inputs for batch in batches)
    assert first.layout.frames == 28
    assert len(first.layout.video_ranges) == 9
    for field in fields(first):
        a, b = getattr(first, field.name), getattr(full, field.name)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    for key in batches[0].clean:
        assert torch.equal(batches[0].clean[key], batches[1].clean[key])


def test_ltx_chain_score_window_offsets_and_generated_block_denominator():
    from vforge.algorithms.dmd.config import SegmentChainConfig
    from vforge.tasks.ltx.dmd.critic import LtxDmdCriticAdapter
    from vforge.tasks.ltx.dmd.generator import LtxDmdGeneratorAdapter
    from vforge.tasks.ltx.dmd.inputs import (
        LtxDmdInputs,
        block_layout,
        loss_partitions,
        valid_masks,
    )

    layout = block_layout(28, 1, 2, 24)
    video = torch.zeros(1, layout.video_ranges[-1][1], 128)
    audio = torch.zeros(1, layout.audio_ranges[-1][1], 128)
    inputs = LtxDmdInputs(
        layout,
        24,
        video[:, :2],
        audio[:, :1],
        None,
        None,
        None,
        None,
        None,
        None,
        {},
        True,
    )
    batch = PreparedDmdBatch({"video": video, "audio": audio}, inputs)
    carry = {}
    for key, value, bounds in (
        ("video", video, layout.video_ranges),
        ("audio", audio, layout.audio_ranges),
    ):
        carry[f"sink_{key}"] = value[:, : bounds[0][1]].clone()
        carry[f"local_{key}"] = value[:, bounds[-2][0] :].clone()
        carry[f"handover_{key}"] = value[:, : 2 if key == "video" else 1].clone()
    generator = LtxDmdGeneratorAdapter(chain=SegmentChainConfig())
    prepared = generator.prepare_segment(batch, carry)
    assert len(generator.block_layout(prepared, prepared.clean)) == 6
    regions = loss_partitions(prepared)
    assert len(regions) == 6
    assert regions[0]["video"].start == 0
    assert regions[-1]["video"].stop == prepared.clean["video"].shape[1]
    assert valid_masks(prepared)["video"].all()
    critic = LtxDmdCriticAdapter()
    window, scored = critic.score_window(prepared.clean, prepared)
    assert window["video"].shape[1] == prepared.clean["video"].shape[1] + 2
    assert window["audio"].shape[1] == prepared.clean["audio"].shape[1] + 1
    assert not valid_masks(scored)["video"][:, :2].any()
    assert valid_masks(scored)["audio"] is None  # Audio handover participates in normalization.
    for key, value in critic.generated_region(window, prepared).items():
        assert torch.equal(value, prepared.clean[key])
    values = {
        key: torch.zeros_like(value, requires_grad=True) for key, value in prepared.clean.items()
    }
    loss = BlockDistributionMatchingSteps._dmd_surrogate_loss_bundle(
        values,
        {key: torch.ones_like(value) for key, value in values.items()},
        component_weights={"video": 1, "audio": 1},
        partitions=regions,
    )
    assert loss.item() == 1.0  # Six generated blocks, not nine slots, form the denominator.
    loss.backward()
    assert all(value.grad is not None for value in values.values())
    assert all(value.grad is None for value in carry.values())


def test_chain_score_normalizes_noisy_suffix_including_audio_handover():
    from vforge.algorithms.dmd.segment_chain import SegmentChainDistributionMatchingSteps

    random = torch.Generator().manual_seed(123)
    clean = {key: torch.randn(1, 17, 8, generator=random).bfloat16() for key in ("video", "audio")}
    fake, cond, uncond = [
        {key: torch.randn(value.shape, generator=random).bfloat16() for key, value in clean.items()}
        for _ in range(3)
    ]
    mask = torch.ones(1, 17, 1, dtype=torch.bool)
    mask[:, :2] = False
    steps = SegmentChainDistributionMatchingSteps(
        config=SimpleNamespace(
            score=SimpleNamespace(component_guidance_scales={"video": 3, "audio": 5})
        ),
        data=SimpleNamespace(valid_masks=lambda _: {"video": mask, "audio": None}),
        generator_adapter=None,
        real_score_adapter=None,
        fake_score_adapter=None,
    )
    actual = steps.score_gradient(
        clean,
        clean,
        None,
        None,
        fake_score=lambda *_: fake,
        real_cond_score=lambda *_: cond,
        real_uncond_score=lambda *_: uncond,
    )
    for key, offset, scale in (("video", 2, 3), ("audio", 0, 5)):
        real = cond[key][:, offset:] + (scale - 1) * (
            cond[key][:, offset:] - uncond[key][:, offset:]
        )
        norm = (clean[key][:, offset:] - real).abs().mean(dim=(1, 2), keepdim=True)
        expected = torch.nan_to_num((fake[key][:, offset:] - real) / (norm + 1e-8))
        assert torch.equal(actual[key][:, offset:], expected)
        assert not actual[key][:, :offset].count_nonzero()


def test_segment_chain_mid_chain_resume_preserves_losses_parameters_and_adamw():
    import io
    from dataclasses import replace

    from torchdata.stateful_dataloader import StatefulDataLoader

    from vforge.algorithms.dmd.config import SegmentChainDistributionMatchingConfig
    from vforge.algorithms.dmd.segment_chain import (
        SegmentChainDataLoader,
        SegmentChainDistributionMatchingSteps,
    )
    from vforge.tasks.common import compose_standalone_config

    class ChainToy(ToyGenerator):
        def denoise_block(self, generator, noisy_block, selection, noise_level, batch, context):
            values = super().denoise_block(
                generator, noisy_block, selection, noise_level, batch, context
            )
            return {key: value + batch.clean[key][:, :1] for key, value in values.items()}

        def prepare_segment(self, batch, carry):
            if carry is None:
                return batch
            return replace(
                batch, clean={key: value + carry[key] for key, value in batch.clean.items()}
            )

        def advance_segment(self, generated, prepared, carry):
            return {key: value[:, -1:].detach() for key, value in generated.items()}

    cfg = compose_standalone_config("train", overrides=["experiment=ltx23_long_tuning"])
    config = SegmentChainDistributionMatchingConfig.from_mapping(cfg.training.dmd)

    def build():
        models = {
            role: torch.nn.Linear(1, 1, bias=False)
            for role in ("generator", "real_score", "fake_score")
        }
        for index, model in enumerate(models.values()):
            model.weight.data.fill_(0.3 + index * 0.2)
        root = RoleModelBundle(models, frozen_roles={"real_score"})
        adapter, critic = ChainToy(False), ToyCritic()
        source = StatefulDataLoader([1.0, 2.0], batch_size=None)
        loader = SegmentChainDataLoader(source, config.chain, adapter)
        family = DmdFamilyComponents(
            models=models,
            build_train_stream=None,
            prepare_batch=lambda sample, device: loader.prepare(
                PreparedDmdBatch(
                    {key: torch.full((1, 4, 1), sample) for key in ("video", "audio")}, None
                )
            ),
            generator_adapter=adapter,
            critic_adapters={"real_score": critic, "fake_score": critic},
            pretrained_weights=(),
            resolve_noise_level=lambda _, level, batch: DmdNoiseLevel(
                level, dict.fromkeys(batch.clean, level), level
            ),
            resolve_rollout_noise_level=lambda _, level, batch: DmdNoiseLevel(
                level, dict.fromkeys(batch.clean, level), level
            ),
            score_noise_shape=lambda _: (1,),
            valid_masks=lambda _: None,
            loss_partitions=lambda _: [
                dict.fromkeys(("video", "audio"), region)
                for region in (slice(0, 1), slice(1, 2), slice(2, 4))
            ],
        )
        runtime = BlockDistributionMatchingRuntime(family=family, config=config, seed=42)
        data = DistributionMatchingData(
            dataloader=loader,
            prepare_batch=family.prepare_batch,
            rollout=runtime.rollout,
            rollout_trace=runtime.rollout_trace,
            sample_score_noise=runtime.sample_score_noise,
            valid_masks=family.valid_masks,
            loss_partitions=family.loss_partitions,
            batch_selector=lambda batch, _: batch,
        )
        steps = SegmentChainDistributionMatchingSteps(
            config=config,
            data=data,
            generator_adapter=adapter,
            real_score_adapter=critic,
            fake_score_adapter=critic,
        )
        opts = {
            phase: torch.optim.AdamW(
                models[role].parameters(), lr=1e-5, betas=(0.0, 0.999), weight_decay=0
            )
            for phase, role in (("generator", "generator"), ("critic", "fake_score"))
        }
        return root, loader, steps, opts, RandomStreams()

    def run(state, start, end):
        root, loader, steps, opts, random_streams = state
        context = SimpleNamespace(device=torch.device("cpu"), random_streams=random_streams)
        losses = []
        for step in range(start, end):
            batch = next(loader)
            for phase in ["generator", "critic"] if step % 5 == 0 else ["critic"]:
                opts[phase].zero_grad(set_to_none=True)
                output = getattr(steps, f"{phase}_step")(root, batch, step, context)
                run_backward(
                    loss=output.loss,
                    backward_losses=output.backward_losses,
                )
                clip_and_step(
                    model=root.model_for("generator" if phase == "generator" else "fake_score"),
                    optimizer=opts[phase],
                    scheduler=None,
                    usp_size=1,
                    gradient_clip_val=1,
                    parity=None,
                )
                losses.append(output.loss.detach().clone())
            loader.commit(root, step)
        return losses

    continuous = build()
    run(continuous, 0, 6)
    root, loader, _, opts, random_streams = continuous
    stream = io.BytesIO()
    torch.save(
        {
            "model": root.state_dict(),
            "loader": loader.state_dict(),
            "optimizers": {key: opt.state_dict() for key, opt in opts.items()},
            "rng_streams": random_streams.state_dict(),
        },
        stream,
    )
    expected_losses = run(continuous, 6, 18)
    expected_rng = random_streams.state_dict()
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=False)
    resumed = build()
    resumed[0].load_state_dict(checkpoint["model"])
    resumed[1].load_state_dict(checkpoint["loader"])
    for key, opt in resumed[3].items():
        opt.load_state_dict(checkpoint["optimizers"][key])
    resumed[4].load_state_dict(checkpoint["rng_streams"])
    actual_losses = run(resumed, 6, 18)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        else:
            assert a == b

    equal(expected_losses, actual_losses)
    equal(continuous[0].state_dict(), resumed[0].state_dict())
    equal(continuous[1].state_dict(), resumed[1].state_dict())
    equal(expected_rng["streams"], resumed[4].state_dict()["streams"])
    for key in opts:
        equal(continuous[3][key].state_dict(), resumed[3][key].state_dict())


def test_vae_resume_identity_uses_content_not_staging_path(tmp_path):
    import json
    import shutil

    from safetensors.torch import save_file

    from vforge.models.ltx.video_vae import video_vae_identity

    first, staged = tmp_path / "base.safetensors", tmp_path / "staged.safetensors"
    metadata = {"config": json.dumps({"vae": {"dims": 3}})}
    state = {"vae.encoder.weight": torch.ones(4), "model.diffusion_model.weight": torch.zeros(2)}
    save_file(state, first, metadata=metadata)
    shutil.copyfile(first, staged)
    identity = video_vae_identity(first)
    assert video_vae_identity(staged) == identity
    state["vae.encoder.weight"][0] += 1
    save_file(state, staged, metadata=metadata)
    assert video_vae_identity(staged) != identity


def test_video_handover_preserves_next_rollout_rng_and_decoded_result():
    from vforge.models.ltx.video_vae import VideoHandover
    from vforge.models.ltx.video_vae.model import ConvVideoDecoder, VideoEncoder

    # Exercise real stochastic decoding, with small layers and no external weights.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(123)
        handover = object.__new__(VideoHandover)
        handover.decoder = ConvVideoDecoder(
            patch_size=1, base_channels=8, timestep_conditioning=True
        ).to(torch.bfloat16)
        handover.encoder = VideoEncoder(patch_size=1).to(torch.bfloat16)
        for model in (handover.decoder, handover.encoder):
            model.eval().requires_grad_(False)
            for parameter in model.parameters():
                parameter.uniform_(-0.02, 0.02)
        tail = torch.randn(1, 18, 128, dtype=torch.bfloat16)
        before = torch.get_rng_state()

        # Same incoming RNG must still produce the original decode/encode result.
        latent = tail.reshape(1, 3, 2, 3, 128).permute(0, 4, 1, 2, 3)
        pixels = handover.decoder(latent)[:, :, -1:].contiguous()
        expected = handover.encoder(pixels)[:, :128]
        expected = expected.permute(0, 2, 3, 4, 1).reshape(1, 6, 128)
        assert not torch.equal(torch.get_rng_state(), before)

        # Simulate a USP peer that skips the source-only VAE operation.
        torch.set_rng_state(before)
        peer_noise = torch.randn_like(tail)
        torch.set_rng_state(before)
        actual = handover(tail, height=2, width=3)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), before)
        assert torch.equal(torch.randn_like(tail), peer_noise)
