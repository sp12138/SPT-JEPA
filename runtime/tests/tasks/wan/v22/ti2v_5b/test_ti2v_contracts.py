"""Small-model contracts for the Wan22 TI2V encode/process/step/sample boundaries.

Everything here pins a behaviour that separates TI2V from Wan21 I2V: temporal
first-frame in-painting, per-token timesteps, the 2*z_dim input channel layout
and the widened ``patch_embedding`` load.
"""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

Z_DIM = 2
LATENT_FRAMES = 3
LATENT_H = 4
LATENT_W = 4
PATCH_SIZE = (1, 2, 2)
TOKENS_PER_FRAME = (LATENT_H // PATCH_SIZE[1]) * (LATENT_W // PATCH_SIZE[2])


@pytest.fixture
def cpu_ops_runtime():
    """vforge attention refuses to dispatch until a runtime exists."""
    from vforge.distributed.parallel_state import ParallelConfig
    from vforge.distributed.runtime import destroy_runtime, initialize_runtime
    from vforge.ops.config import OpsConfig

    initialize_runtime(
        ParallelConfig(),
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
    )
    try:
        yield
    finally:
        destroy_runtime()


#: Distinct fill per stream so a concat order mistake cannot pass.
STREAM_FILL = {"normal": 0.25, "grid": 0.5, "rgb": 0.75}
PARTIAL_NORMAL_SOURCES = ("normal_fg_body", "normal_bg_body")


def _stream_fields(streams: tuple[str, ...]) -> dict[str, str]:
    from vforge.tasks.wan.v22.ti2v_5b.shared import stream_fields

    return stream_fields(streams)


def _encoded_batch(
    *,
    with_first_frame: bool = True,
    streams: tuple[str, ...] = ("normal",),
) -> dict[str, torch.Tensor | None]:
    latent_shape = (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    return {
        "prompt": torch.ones(1, 3, 4),
        "frame_latents": torch.arange(
            int(torch.tensor(latent_shape).prod()),
            dtype=torch.float32,
        ).reshape(latent_shape),
        **{
            f"condition_latents_{name}": torch.full(latent_shape, STREAM_FILL[name])
            for name in streams
        },
        "first_frame_latents": (
            torch.full((1, Z_DIM, 1, LATENT_H, LATENT_W), -7.0) if with_first_frame else None
        ),
    }


def _long_encoded_batch(*, frames: int = 8) -> dict[str, torch.Tensor | int]:
    latent_shape = (1, Z_DIM, frames, LATENT_H, LATENT_W)
    return {
        "prompt": torch.ones(1, 3, 4),
        "frame_latents": torch.arange(
            int(torch.tensor(latent_shape).prod()),
            dtype=torch.float32,
        ).reshape(latent_shape),
        "condition_latents_normal": torch.full(latent_shape, STREAM_FILL["normal"]),
        "first_frame_latents": torch.full((1, Z_DIM, 1, LATENT_H, LATENT_W), -7.0),
        "min_latents": 3,
        "max_latents": 6,
    }


def _extras(**overrides) -> dict:
    return {
        "noise_shift": 1.0,
        "text_drop_prob": 0.0,
        "normal_drop_prob": 0.0,
        "patch_size": PATCH_SIZE,
        **overrides,
    }


def _processor(**overrides):
    from vforge.tasks.wan.v22.ti2v_5b.ct.train import _ConditionProcessor

    return _ConditionProcessor(extras=_extras(**overrides), precision="fp32")


def _tiny_transformer(**overrides):
    from vforge.models.wan.v22.ti2v_5b.ct.transformer import WanTransformer3DModel

    config = {
        "patch_size": PATCH_SIZE,
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": 2 * Z_DIM,
        "out_channels": Z_DIM,
        "text_dim": 4,
        "freq_dim": 16,
        "ffn_dim": 16,
        "num_layers": 1,
        "rope_max_seq_len": 32,
    }
    config.update(overrides)
    return WanTransformer3DModel(**config)


def test_per_token_timestep_is_frame_major_with_a_clean_first_frame() -> None:
    from vforge.models.wan.v22.ti2v_5b.ct.inference import build_per_token_timestep

    latent = torch.zeros(2, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)

    timestep = build_per_token_timestep(latent, torch.tensor(731.0), PATCH_SIZE)

    assert timestep.shape == (2, LATENT_FRAMES * TOKENS_PER_FRAME)
    # Frame 0 holds the given reference frame and is therefore not noisy.
    assert torch.equal(timestep[:, :TOKENS_PER_FRAME], torch.zeros(2, TOKENS_PER_FRAME))
    assert torch.equal(
        timestep[:, TOKENS_PER_FRAME:],
        torch.full((2, (LATENT_FRAMES - 1) * TOKENS_PER_FRAME), 731.0),
    )


def test_encoded_codec_requires_the_reference_frame_and_has_no_image_branch() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec

    codec = _codec({})

    assert "first_frame_latents" in codec.required_keys
    assert not any(key.startswith("image_condition_") for key in codec.keys)


def test_encoded_codec_declares_one_required_field_per_condition_stream() -> None:
    """A cache is only usable by the stream list that wrote it, both ways round."""
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec, encoded_batch_keys

    assert encoded_batch_keys([]) == (
        "prompt",
        "frame_latents",
        "first_frame_latents",
    )
    assert encoded_batch_keys(["grid", "normal"]) == (
        "prompt",
        "frame_latents",
        "condition_latents_grid",
        "condition_latents_normal",
        "first_frame_latents",
    )
    # Nothing is optional: a stream the list declares is a stream the cache owes.
    assert _codec({"condition_streams": ["rgb", "grid", "normal"]}).optional_keys == ()


def test_encoded_codec_rejects_a_cache_written_for_another_stream_list(tmp_path) -> None:
    from vforge.data.cache import save_latent_bundle
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec

    path = tmp_path / "normal.pt"
    save_latent_bundle(_codec({}).canonicalize(_encoded_batch()), str(path))

    assert _codec({}).verify(str(path))["condition_latents_normal"] is not None
    with pytest.raises(ValueError, match="condition_latents_grid"):
        _codec({"condition_streams": ["grid", "normal"]}).verify(str(path))


def test_encoded_codec_rejects_a_wan21_cache(tmp_path) -> None:
    """TI2V in-paints latent frame 0, so a cache without the anchor is unusable."""
    from vforge.data.cache import save_latent_bundle
    from vforge.tasks.wan.v21.shared import WAN21_ENCODED_BATCH_CODEC
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec

    path = tmp_path / "wan21.pt"
    save_latent_bundle(
        WAN21_ENCODED_BATCH_CODEC.canonicalize(
            {
                "prompt": torch.ones(1, 3, 4),
                "frame_latents": torch.ones(1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W),
                "image_condition_latent_vae": torch.ones(
                    1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W
                ),
                "condition_latents_normal": torch.ones(1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W),
            }
        ),
        str(path),
    )
    with pytest.raises(ValueError, match="first_frame_latents"):
        _codec({}).verify(str(path))


def test_encoded_codec_collates_a_single_latent_reference_frame() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec, _collate_encoded_bundles

    collated = _collate_encoded_bundles(
        _codec({}),
        [{**_encoded_batch(), "sample_id": "ti2v"}],
    )
    assert collated["first_frame_latents"].shape == (1, Z_DIM, 1, LATENT_H, LATENT_W)


def test_encoded_codec_collates_homogeneous_latent_bounds() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec, _collate_encoded_bundles

    sample = {**_encoded_batch(), "sample_id": "ti2v", "min_latents": 21, "max_latents": 38}
    collated = _collate_encoded_bundles(_codec({}), [sample, dict(sample)])
    assert collated["min_latents"] == 21
    assert collated["max_latents"] == 38


def test_encoded_codec_rejects_mixed_latent_bounds() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec, _collate_encoded_bundles

    left = {**_encoded_batch(), "sample_id": "a", "min_latents": 21, "max_latents": 38}
    right = {**_encoded_batch(), "sample_id": "b", "min_latents": 10, "max_latents": 38}
    with pytest.raises(ValueError, match="min_latents"):
        _collate_encoded_bundles(_codec({}), [left, right])


def test_condition_processor_inpaints_first_frame_and_drops_it_from_loss() -> None:

    encoded = _encoded_batch()
    processed = _processor().process(
        encoded,
        step=0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
        apply_dropouts=False,
    )

    hidden = processed["model_inputs"]["hidden_states"]
    assert hidden.shape[1] == 2 * Z_DIM
    assert torch.equal(
        hidden[:, :Z_DIM, 0:1],
        encoded["first_frame_latents"],
    )
    assert torch.equal(hidden[:, Z_DIM:], encoded["condition_latents_normal"])
    assert processed["targets"].shape[1] == Z_DIM

    assert processed["model_inputs"]["timestep"].shape == (
        1,
        LATENT_FRAMES * TOKENS_PER_FRAME,
    )

    weights = processed["loss_weights"]
    assert weights.shape == encoded["frame_latents"].shape
    assert torch.count_nonzero(weights[:, :, 0]) == 0
    assert torch.count_nonzero(weights[:, :, 1:]) == weights[:, :, 1:].numel()


def test_condition_processor_truncates_with_a_separate_length_generator() -> None:
    encoded = _long_encoded_batch()
    first_frame = encoded["first_frame_latents"].clone()
    seed, step = 11, 4
    device = torch.device("cpu")
    expected = int(
        torch.randint(
            3,
            7,
            (1,),
            device=device,
            generator=torch.Generator().manual_seed(seed),
        ).item()
    )

    processed = _processor().process(
        encoded,
        step=step,
        device=device,
        generator=torch.Generator().manual_seed(0),
        length_generator=torch.Generator().manual_seed(seed),
        apply_dropouts=False,
    )

    assert processed["targets"].shape[2] == expected
    assert processed["model_inputs"]["hidden_states"].shape[2] == expected
    assert processed["loss_weights"].shape[2] == expected
    assert torch.equal(encoded["first_frame_latents"], first_frame)
    assert torch.equal(
        processed["model_inputs"]["hidden_states"][:, :Z_DIM, 0:1],
        first_frame,
    )


def test_condition_processor_skips_truncation_when_bounds_are_absent() -> None:
    encoded = _encoded_batch()
    processed = _processor().process(
        encoded,
        step=9,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
        apply_dropouts=False,
    )
    assert processed["targets"].shape[2] == LATENT_FRAMES


def test_condition_processor_keeps_first_frame_in_loss_when_asked() -> None:

    processed = _processor(exclude_first_frame_from_loss=False).process(
        _encoded_batch(),
        step=0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
        apply_dropouts=False,
    )

    weights = processed["loss_weights"]
    assert torch.count_nonzero(weights[:, :, 0]) == weights[:, :, 0].numel()


def test_condition_processor_rejects_a_cache_without_first_frame_latents() -> None:

    with pytest.raises(ValueError, match="first_frame_latents"):
        _processor().process(
            _encoded_batch(with_first_frame=False),
            step=0,
            device=torch.device("cpu"),
            apply_dropouts=False,
        )


def test_normal_dropout_never_drops_the_inpainted_first_frame() -> None:
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_encoded_training_dropouts

    bundle = _encoded_batch()

    result = apply_encoded_training_dropouts(
        bundle,
        stream_fields=_stream_fields(("normal",)),
        stream_drop_probs={"normal": 1.0},
        text_drop_prob=1.0,
    )

    assert torch.count_nonzero(result["condition_latents_normal"]) == 0
    assert torch.count_nonzero(result["prompt"]) == 0
    assert torch.equal(result["first_frame_latents"], bundle["first_frame_latents"])


def test_each_condition_stream_drops_independently() -> None:
    """CFG needs every combination of present conditions, not all-or-nothing."""
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_encoded_training_dropouts

    bundle = _encoded_batch(streams=("grid", "normal"))

    result = apply_encoded_training_dropouts(
        bundle,
        stream_fields=_stream_fields(("grid", "normal")),
        stream_drop_probs={"grid": 1.0, "normal": 0.0},
        text_drop_prob=0.0,
    )

    assert torch.count_nonzero(result["condition_latents_grid"]) == 0
    assert torch.equal(
        result["condition_latents_normal"],
        bundle["condition_latents_normal"],
    )
    assert torch.equal(result["prompt"], bundle["prompt"])


def _bundle_with_partial_normal_sources(**overrides):
    latent_shape = (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    return {
        **_encoded_batch(**overrides),
        "normal_fg_body": torch.full(latent_shape, 1.5),
        "normal_bg_body": torch.full(latent_shape, 2.5),
    }


@pytest.mark.parametrize(
    ("variant", "expected"),
    [(0, 0.25), (1, 1.5), (2, 2.5)],
)
def test_partial_normal_augmentation_selects_a_body_per_variant(
    variant: int,
    expected: float,
) -> None:
    """Variant 0 keeps the full normal; 1 and 2 take the fg / bg bodies."""
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_partial_normal_augmentation

    result = apply_partial_normal_augmentation(
        _bundle_with_partial_normal_sources(),
        variant,
        partial_normal_sources=PARTIAL_NORMAL_SOURCES,
        normal_field="condition_latents_normal",
    )

    assert torch.allclose(
        result["condition_latents_normal"],
        torch.full_like(result["condition_latents_normal"], expected),
    )


def test_partial_normal_augmentation_consumes_its_sources() -> None:
    """Leaving them in would let a second pass augment an already-swapped stream."""
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_partial_normal_augmentation

    result = apply_partial_normal_augmentation(
        _bundle_with_partial_normal_sources(),
        1,
        partial_normal_sources=PARTIAL_NORMAL_SOURCES,
        normal_field="condition_latents_normal",
    )

    assert "normal_fg_body" not in result
    assert "normal_bg_body" not in result


def test_partial_normal_augmentation_rejects_a_bundle_without_the_bodies() -> None:
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_partial_normal_augmentation

    with pytest.raises(ValueError, match="normal_fg_body"):
        apply_partial_normal_augmentation(
            _encoded_batch(),
            1,
            partial_normal_sources=PARTIAL_NORMAL_SOURCES,
            normal_field="condition_latents_normal",
        )


def test_partial_normal_variant_is_uniform_over_three_choices() -> None:
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_encoded_training_dropouts

    counts: dict[float, int] = {}
    for seed in range(300):
        random.seed(seed)
        result = apply_encoded_training_dropouts(
            _bundle_with_partial_normal_sources(),
            stream_fields=_stream_fields(("normal",)),
            stream_drop_probs={"normal": 0.0},
            text_drop_prob=0.0,
            partial_normal_aug=True,
            partial_normal_sources=PARTIAL_NORMAL_SOURCES,
        )
        value = float(result["condition_latents_normal"].flatten()[0])
        counts[value] = counts.get(value, 0) + 1

    assert sorted(counts) == [0.25, 1.5, 2.5]
    assert all(60 <= count <= 140 for count in counts.values()), counts


def test_partial_normal_draws_exactly_one_value_when_no_stream_can_drop() -> None:
    """The augmentation is the only RNG consumer at zero dropout.

    Taichu draws the variant after every dropout decision, so with all
    probabilities at zero both runtimes take exactly one draw and land on the
    same variant. A stray draw here would silently desynchronise them.
    """
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_encoded_training_dropouts

    draws = 0
    real_randrange = random.randrange
    real_random = random.random

    def counting_randrange(*args, **kwargs):
        nonlocal draws
        draws += 1
        return real_randrange(*args, **kwargs)

    def counting_random(*args, **kwargs):
        nonlocal draws
        draws += 1
        return real_random(*args, **kwargs)

    random.randrange = counting_randrange
    random.random = counting_random
    try:
        apply_encoded_training_dropouts(
            _bundle_with_partial_normal_sources(streams=("grid", "normal")),
            stream_fields=_stream_fields(("grid", "normal")),
            stream_drop_probs={"grid": 0.0, "normal": 0.0},
            text_drop_prob=0.0,
            partial_normal_aug=True,
            partial_normal_sources=PARTIAL_NORMAL_SOURCES,
        )
    finally:
        random.randrange = real_randrange
        random.random = real_random

    assert draws == 1


def test_partial_normal_is_off_by_default() -> None:
    """A cache that happens to carry the bodies must not change training."""
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_encoded_training_dropouts

    bundle = _bundle_with_partial_normal_sources()
    for seed in range(20):
        random.seed(seed)
        result = apply_encoded_training_dropouts(
            bundle,
            stream_fields=_stream_fields(("normal",)),
            stream_drop_probs={"normal": 0.0},
            text_drop_prob=0.0,
        )
        assert torch.equal(
            result["condition_latents_normal"],
            bundle["condition_latents_normal"],
        )


def test_dropping_the_normal_stream_wins_over_the_augmentation() -> None:
    """Order matters: a CFG-dropped stream carries no condition at all."""
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_encoded_training_dropouts

    result = apply_encoded_training_dropouts(
        _bundle_with_partial_normal_sources(),
        stream_fields=_stream_fields(("normal",)),
        stream_drop_probs={"normal": 1.0},
        text_drop_prob=0.0,
        partial_normal_aug=True,
        partial_normal_sources=PARTIAL_NORMAL_SOURCES,
    )

    assert torch.count_nonzero(result["condition_latents_normal"]) == 0


def test_partial_normal_codec_requires_the_alternate_bodies() -> None:
    """Required, so a cache that cannot serve the augmentation fails on read."""
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec

    extras = {"condition_streams": ["grid", "normal"], "partial_normal_aug": True}
    codec = _codec(extras)

    assert "normal_fg_body" in codec.required_keys
    assert "normal_bg_body" in codec.required_keys

    plain = _codec({"condition_streams": ["grid", "normal"]})
    assert "normal_fg_body" not in plain.keys


def test_partial_normal_codec_rejects_a_stream_list_without_normal() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import encoded_batch_keys

    with pytest.raises(ValueError, match="partial_normal_aug"):
        encoded_batch_keys(["rgb"], partial_normal_aug=True)


def test_task_rejects_partial_normal_aug_on_the_online_path() -> None:
    """This encoder has no fg / bg pixel source, so only a cache can serve it."""
    from vforge.tasks.wan.v22.ti2v_5b.shared import _validate

    _validate(
        extras=_extras(partial_normal_aug=True),
        latent_cache_mode="offline",
        usp_size=1,
    )
    with pytest.raises(ValueError, match="latent_cache.mode=offline"):
        _validate(
            extras=_extras(partial_normal_aug=True),
            latent_cache_mode="online",
            usp_size=1,
        )


def test_encode_and_eval_stay_on_the_schema_this_encoder_can_produce() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec, _encoder_codec

    extras = _extras(partial_normal_aug=True)

    assert "normal_fg_body" in _codec(extras).keys
    assert "normal_fg_body" not in _encoder_codec(extras).keys


def test_condition_processor_applies_the_partial_normal_augmentation() -> None:
    """The offline path owns dropouts, so the augmentation has to reach it."""
    random.seed(1)
    processed = _processor(partial_normal_aug=True).process(
        _bundle_with_partial_normal_sources(),
        step=0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
        apply_dropouts=True,
    )

    normal = processed["model_inputs"]["hidden_states"][:, Z_DIM:]
    assert float(normal.flatten()[0]) in {0.25, 1.5, 2.5}


def test_condition_processor_concatenates_streams_in_declared_order() -> None:
    encoded = _encoded_batch(streams=("grid", "normal"))
    processed = _processor(condition_streams=["grid", "normal"]).process(
        encoded,
        step=0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
        apply_dropouts=False,
    )

    hidden = processed["model_inputs"]["hidden_states"]
    assert hidden.shape[1] == 3 * Z_DIM
    # Order is noisy latent, then the declared order: grid before normal.
    assert torch.equal(hidden[:, Z_DIM : 2 * Z_DIM], encoded["condition_latents_grid"])
    assert torch.equal(hidden[:, 2 * Z_DIM :], encoded["condition_latents_normal"])
    # The prediction target stays z_dim wide however many streams come in.
    assert processed["targets"].shape[1] == Z_DIM


def test_condition_processor_rejects_a_batch_missing_a_declared_stream() -> None:

    with pytest.raises(ValueError, match="condition_latents_grid"):
        _processor(condition_streams=["grid", "normal"]).process(
            _encoded_batch(streams=("normal",)),
            step=0,
            device=torch.device("cpu"),
            apply_dropouts=False,
        )


def test_condition_processor_feeds_the_bare_latent_when_no_stream_is_declared() -> None:

    processed = _processor(condition_streams=[]).process(
        _encoded_batch(streams=()),
        step=0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
        apply_dropouts=False,
    )

    assert processed["model_inputs"]["hidden_states"].shape[1] == Z_DIM


def test_uniform_per_token_timestep_matches_the_scalar_timestep_path(
    cpu_ops_runtime,
) -> None:
    torch.manual_seed(0)
    model = _tiny_transformer().eval()
    hidden = torch.randn(1, 2 * Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    prompt = torch.randn(1, 5, 4)
    tokens = LATENT_FRAMES * TOKENS_PER_FRAME

    per_token = model(
        hidden_states=hidden,
        timestep=torch.full((1, tokens), 500.0),
        encoder_hidden_states=prompt,
    )
    scalar = model(
        hidden_states=hidden,
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=prompt,
    )

    assert per_token.shape == (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    torch.testing.assert_close(per_token, scalar)


def test_sequence_parallel_shards_per_token_modulation_with_hidden_states(
    cpu_ops_runtime,
    monkeypatch,
) -> None:
    """Per-token AdaLN carries a sequence axis, so Ulysses must shard it too.

    If ``timestep_proj``/``temb`` stayed full-length while ``hidden_states`` was
    sharded, every rank but the first would modulate with the wrong tokens.
    """
    import vforge.distributed.ulysses as usp_ops
    import vforge.models.wan.v22.ti2v_5b.ct.transformer as transformer_module

    sliced: list[tuple[int, ...]] = []

    def record_slice(tensor, *, dim, group=None, scale_grad=False):
        assert dim == 1
        sliced.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(usp_ops, "slice_sequence", record_slice)
    monkeypatch.setattr(
        usp_ops,
        "gather_sequence",
        lambda tensor, *, dim, group=None, scale_grad=False: tensor,
    )
    monkeypatch.setattr(
        transformer_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            config=SimpleNamespace(usp_size=2),
            ulysses_group=None,
            usp_group=None,
        ),
    )

    torch.manual_seed(0)
    inner_dim = 2 * 12
    model = _tiny_transformer().eval()
    hidden = torch.randn(1, 2 * Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    prompt = torch.randn(1, 5, 4)
    tokens = LATENT_FRAMES * TOKENS_PER_FRAME

    model(
        hidden_states=hidden,
        timestep=torch.full((1, tokens), 500.0),
        encoder_hidden_states=prompt,
    )
    per_token_slices = list(sliced)
    sliced.clear()
    model(
        hidden_states=hidden,
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=prompt,
    )

    assert per_token_slices[0] == (1, tokens, inner_dim)
    assert (1, tokens, 6, inner_dim) in per_token_slices
    assert per_token_slices.count((1, tokens, inner_dim)) == 2
    # A scalar timestep has no sequence axis: only hidden states and RoPE shard.
    assert sliced == per_token_slices[:3]


def test_transformer_rejects_wan21_image_cross_attention_configs() -> None:
    with pytest.raises(ValueError, match="image cross-attention"):
        _tiny_transformer(image_dim=64, added_kv_proj_dim=24)


def test_widened_patch_embedding_keeps_pretrained_weights_and_zeroes_the_tail() -> None:
    from vforge.models.wan.model import WanChannelLayout
    from vforge.tasks.wan.v22.ti2v_5b.ct.train import _widen_patch_embedding

    layout = WanChannelLayout(z_dim=48, pretrained_in_channels=48, in_channels=144)
    resolve = _widen_patch_embedding(layout)
    published = torch.randn(8, 48, 1, 2, 2)
    target = torch.zeros(8, 144, 1, 2, 2)

    widened = resolve("transformer.patch_embedding.weight", published, target)

    assert widened is not None
    assert widened.shape == target.shape
    assert torch.equal(widened[:, :48], published)
    assert torch.count_nonzero(widened[:, 48:]) == 0
    # Every other key must fall through untouched.
    assert resolve("transformer.proj_out.weight", published, target) is None


def _published_ti2v_config(tmp_path) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "WanTransformer3DModel",
                "in_channels": 48,
                "out_channels": 48,
                "num_layers": 30,
            }
        ),
        encoding="utf-8",
    )


def test_channel_layout_widens_the_published_ti2v_config(tmp_path) -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _resolve_model_config

    _published_ti2v_config(tmp_path)

    config, layout = _resolve_model_config({"condition_streams": ["normal"]}, str(tmp_path))

    assert config["in_channels"] == 96
    assert config["out_channels"] == 48
    assert layout == type(layout)(z_dim=48, pretrained_in_channels=48, in_channels=96)
    assert layout.is_widened


@pytest.mark.parametrize(
    ("streams", "in_channels"),
    [
        ([], 48),
        (["normal"], 96),
        (["grid", "normal"], 144),
        (["rgb", "grid", "normal"], 192),
    ],
)
def test_channel_layout_adds_one_z_dim_block_per_stream(
    tmp_path,
    streams: list[str],
    in_channels: int,
) -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _resolve_model_config

    _published_ti2v_config(tmp_path)

    config, layout = _resolve_model_config({"condition_streams": streams}, str(tmp_path))

    assert config["in_channels"] == in_channels
    assert config["out_channels"] == 48
    assert layout.is_widened is (in_channels > 48)


def test_channel_layout_rejects_a_checkpoint_widened_for_another_stream_list(
    tmp_path,
) -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _resolve_model_config

    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"in_channels": 96, "out_channels": 48, "num_layers": 30}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requested width"):
        _resolve_model_config({"condition_streams": ["grid", "normal"]}, str(tmp_path))


def test_sampling_pins_the_first_frame_and_is_seed_reproducible() -> None:
    from vforge.models.wan.v22.ti2v_5b.ct.inference import Wan22Ti2v5bPipeline

    seen_timesteps: list[torch.Tensor] = []

    class Transformer(torch.nn.Module):
        config = SimpleNamespace(in_channels=2 * Z_DIM, patch_size=PATCH_SIZE)

        @property
        def dtype(self):
            return torch.float32

        def cache_context(self, _name):
            return nullcontext()

        def forward(self, hidden_states, timestep, **_kwargs):
            seen_timesteps.append(timestep)
            return (torch.ones_like(hidden_states[:, :Z_DIM]),)

    class Scheduler:
        def set_timesteps(self, steps, device):
            self.timesteps = [torch.tensor(float(step + 1), device=device) for step in range(steps)]

        def step(self, prediction, _timestep, latents, return_dict):
            assert return_dict is False
            return (latents + prediction,)

    transformer = Transformer()
    pipeline = Wan22Ti2v5bPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=transformer,
        engine_forward=transformer,
        scheduler=Scheduler(),
        vae_scale_factor_temporal=4,
        vae_scale_factor_spatial=16,
    )
    first_frame = torch.full((1, Z_DIM, 1, LATENT_H, LATENT_W), 5.0)
    normal = torch.zeros(1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)

    def sample():
        return pipeline.call_with_latent(
            torch.ones(1, 3, 4),
            y=normal,
            first_frame_latents=first_frame,
            height=LATENT_H * 16,
            width=LATENT_W * 16,
            num_frames=(LATENT_FRAMES - 1) * 4 + 1,
            num_inference_steps=2,
            guidance_scale=1.0,
            generator=torch.Generator().manual_seed(99),
            decode_output=False,
        )

    first = sample()
    second = sample()

    assert first.shape == (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    # The stub predicts ones and the stub scheduler adds the prediction, so two
    # steps move every frame by +2 -- except frame 0, which is re-projected
    # after each step and must come back exactly as it went in.
    assert torch.equal(first[:, :, 0:1], first_frame)
    noise = randn_tensor(
        (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W),
        generator=torch.Generator().manual_seed(99),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(first[:, :, 1:], noise[:, :, 1:] + 2.0)
    assert torch.equal(first, second)
    assert all(
        torch.count_nonzero(timestep[:, :TOKENS_PER_FRAME]) == 0 for timestep in seen_timesteps
    )


def test_task_rejects_a_stream_this_version_cannot_encode() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import _validate

    _validate(extras=_extras(), latent_cache_mode="offline", usp_size=1)
    _validate(
        extras=_extras(condition_streams=["grid", "normal"], grid_drop_prob=0.3),
        latent_cache_mode="offline",
        usp_size=1,
    )
    with pytest.raises(ValueError, match="condition_streams"):
        _validate(
            extras=_extras(condition_streams=["depth"]),
            latent_cache_mode="offline",
            usp_size=1,
        )


def test_task_rejects_a_dropout_for_a_stream_not_in_the_list() -> None:
    """Otherwise a leftover key silently trains without the dropout it asks for."""
    from vforge.tasks.wan.v22.ti2v_5b.shared import _validate

    _validate(
        extras=_extras(condition_streams=["normal"], grid_drop_prob=0.0),
        latent_cache_mode="offline",
        usp_size=1,
    )
    with pytest.raises(ValueError, match="grid_drop_prob"):
        _validate(
            extras=_extras(condition_streams=["normal"], grid_drop_prob=0.3),
            latent_cache_mode="offline",
            usp_size=1,
        )


def test_task_schema_rejects_a_stream_this_version_cannot_encode() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.shared import condition_spec

    with pytest.raises(ValueError, match="depth"):
        condition_spec(["depth"])


class _StubVAE(torch.nn.Module):
    """Causal-VAE stand-in: keeps the first frame, then one latent per 4 frames."""

    config = SimpleNamespace(
        z_dim=Z_DIM,
        latents_mean=[0.0] * Z_DIM,
        latents_std=[1.0] * Z_DIM,
    )

    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.seen: list[torch.Tensor] = []

    def encode(self, pixels: torch.Tensor):
        self.seen.append(pixels.detach().clone())
        latent = pixels[:, :Z_DIM, ::4].clone()
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))


class _StubTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        batch, length = input_ids.shape
        return SimpleNamespace(last_hidden_state=torch.ones(batch, length, 4))


def _stub_encoder(streams, **kwargs):
    from vforge.models.wan.v22.ti2v_5b.ct.encoder import Wan22Ti2v5bConditionEncoder
    from vforge.tasks.wan.v22.ti2v_5b.shared import STREAM_PIXEL_KEYS, stream_fields

    return Wan22Ti2v5bConditionEncoder(
        vae=_StubVAE(),
        text_encoder=_StubTextEncoder(),
        tokenizer=None,
        video_processor=None,
        training_dtype=torch.float32,
        stream_fields=stream_fields(streams),
        raw_stream_keys=STREAM_PIXEL_KEYS,
        **kwargs,
    )


def _raw_batch(*, pixel_frames: int = 9, with_grid: bool = True) -> dict[str, torch.Tensor]:
    shape = (1, pixel_frames, 3, 8, 8)
    batch = {
        "video_frames": torch.rand(shape),
        "normal_condition": torch.full(shape, 0.5),
        "text_inputs": torch.ones(1, 4, dtype=torch.long),
        "text_mask": torch.ones(1, 4, dtype=torch.long),
    }
    if with_grid:
        batch["grid_condition"] = torch.full(shape, 0.25)
    return batch


def test_encoder_emits_one_latent_field_per_declared_stream() -> None:
    encoder = _stub_encoder(["grid", "normal"])

    bundle = encoder.encode_batch(_raw_batch())

    latent_frames = (9 - 1) // 4 + 1
    assert set(bundle) == {
        "prompt",
        "frame_latents",
        "condition_latents_grid",
        "condition_latents_normal",
        "first_frame_latents",
    }
    for field in ("frame_latents", "condition_latents_grid", "condition_latents_normal"):
        assert bundle[field].shape == (1, Z_DIM, latent_frames, 8, 8)
    # Every stream is a plain z_dim latent: TI2V has no 4-channel mask prefix.
    assert bundle["condition_latents_normal"].shape[1] == Z_DIM
    # The anchor is latent frame 0 of the clip, which a causal VAE reproduces
    # when handed the first frame alone.
    assert bundle["first_frame_latents"].shape == (1, Z_DIM, 1, 8, 8)
    torch.testing.assert_close(
        bundle["first_frame_latents"],
        bundle["frame_latents"][:, :, 0:1],
    )


def test_encoder_builds_the_rgb_stream_from_the_reference_frame_and_a_blank_tail() -> None:
    encoder = _stub_encoder(["rgb"])
    batch = _raw_batch(with_grid=False)

    encoder.encode_batch(batch)

    # The rgb stream is encoded last, so it is the final pixel clip the VAE saw.
    rgb_pixels = encoder.vae.seen[-1]
    expected_reference = (batch["video_frames"][:, 0] * 2 - 1).to(rgb_pixels.dtype)
    torch.testing.assert_close(rgb_pixels[:, :, 0], expected_reference)
    assert torch.count_nonzero(rgb_pixels[:, :, 1:]) == 0


def test_encoder_rejects_a_batch_without_the_pixels_a_stream_needs() -> None:
    encoder = _stub_encoder(["grid", "normal"])

    with pytest.raises(ValueError, match="grid_condition"):
        encoder.encode_batch(_raw_batch(with_grid=False))


def test_encoder_rejects_a_dropout_for_a_stream_not_in_the_list() -> None:
    with pytest.raises(ValueError, match="grid"):
        _stub_encoder(["normal"], stream_drop_probs={"grid": 0.3})


def test_sampler_concatenates_condition_streams_in_declared_order() -> None:
    from vforge.models.wan.v22.ti2v_5b.ct.inference import _concat_condition_streams
    from vforge.tasks.wan.v22.ti2v_5b.shared import stream_fields

    encoded = _encoded_batch(streams=("grid", "normal"))
    device = torch.device("cpu")

    assert _concat_condition_streams(encoded, stream_fields([]), device) is None
    condition = _concat_condition_streams(encoded, stream_fields(["grid", "normal"]), device)
    assert condition.shape[1] == 2 * Z_DIM
    assert torch.equal(condition[:, :Z_DIM], encoded["condition_latents_grid"])
    assert torch.equal(condition[:, Z_DIM:], encoded["condition_latents_normal"])
    with pytest.raises(ValueError, match="condition_latents_rgb"):
        _concat_condition_streams(encoded, stream_fields(["rgb", "normal"]), device)


def test_train_step_reports_tokens_and_excludes_the_given_frame_from_loss() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.ct.train import _wan22_train_step
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec
    from vforge.training.recipe import StepContext

    class Model(torch.nn.Module):
        config = SimpleNamespace(patch_size=PATCH_SIZE)

        def forward(self, hidden_states, **_kwargs):
            return (torch.zeros_like(hidden_states[:, :Z_DIM]),)

    context = StepContext(
        device=torch.device("cpu"),
        parallel_state=SimpleNamespace(
            config=SimpleNamespace(usp_size=1),
            usp_rank=0,
            global_rank=0,
            ulysses_group=None,
            usp_group=None,
        ),
    )
    output = _wan22_train_step(
        Model(),
        {**_encoded_batch(), "sample_id": ["sample"]},
        0,
        context,
        latent_cache_mode="offline",
        seed=42,
        codec=_codec({}),
        processor=_processor(),
        encoder=None,
    )

    assert output.workload["local_tokens"] == LATENT_FRAMES * TOKENS_PER_FRAME
    assert output.workload["local_batch"] == 1
    assert output.metadata == {}


def test_train_step_drives_a_real_transformer_widened_for_two_streams(
    cpu_ops_runtime,
) -> None:
    """The widened patch embedding has to accept every declared stream."""
    from vforge.tasks.wan.v22.ti2v_5b.ct.train import _wan22_train_step
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec
    from vforge.training.recipe import StepContext

    torch.manual_seed(0)
    streams = ["grid", "normal"]
    model = _tiny_transformer(in_channels=3 * Z_DIM)
    context = StepContext(
        device=torch.device("cpu"),
        parallel_state=SimpleNamespace(
            config=SimpleNamespace(usp_size=1),
            usp_rank=0,
            global_rank=0,
            ulysses_group=None,
            usp_group=None,
        ),
    )
    output = _wan22_train_step(
        model,
        {**_encoded_batch(streams=("grid", "normal")), "sample_id": ["sample"]},
        0,
        context,
        latent_cache_mode="offline",
        seed=42,
        codec=_codec({"condition_streams": streams}),
        processor=_processor(condition_streams=streams),
        encoder=None,
    )

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.workload["local_tokens"] == LATENT_FRAMES * TOKENS_PER_FRAME
