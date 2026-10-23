import pytest
import torch

from vforge.models.loading import load_pretrained_weights
from vforge.tasks.rep_world.shared import model_spec, parse_config, pretrained_source
from vforge.tasks.rep_world.weights import load_legacy_model


def test_native_pt_warmstart_rejects_partial_state_even_in_compatible_mode(tmp_path):
    config = parse_config(
        dict(
            codec=dict(family="vrae", variant="eupe"),
            world=dict(width=48, depth=1, heads=4),
            conditioner=dict(input_size=32, width=24, depth=2, heads=4, feature_layers=[1]),
        )
    )
    original = model_spec(config, "cpu").build()
    path = tmp_path / "native.pt"
    torch.save(original.state_dict(), path)
    restored = model_spec(config, "meta").build().to_empty(device="cpu")
    source = pretrained_source(str(path), config)
    load_pretrained_weights(restored, source, strict=False)
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    incomplete = dict(original.state_dict())
    incomplete.pop(next(iter(incomplete)))
    torch.save(incomplete, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_pretrained_weights(restored, source, strict=False)


def test_pure_tf_legacy_import_converts_only_derived_rope_window(tmp_path):
    config = parse_config(
        dict(
            codec=dict(family="videorae", variant="3d"),
            states=3,
            flow=dict(mode="pure_tf"),
            world=dict(width=48, depth=1, heads=4),
            initial_weights_format="legacy",
        )
    )
    original = model_spec(config, "cpu").build()
    state = dict(original.state_dict())
    rope = original.net.feat_rope
    cos, sin = rope._build_freqs(
        12,
        rope.base_freqs_t,
        rope.base_freqs_h,
        rope.base_freqs_w,
        rope.frame_height,
        rope.frame_width,
        rope.dim,
    )
    state["net.feat_rope.freqs_cos"], state["net.feat_rope.freqs_sin"] = cos, sin
    payload = dict(
        model=state,
        meta=dict(
            latent_codec="videorae",
            rae_image_latent_channels=64,
            rae_group_size=4,
            next_latent_use_full_history=True,
        ),
    )
    path = tmp_path / "pure_tf.pt"
    torch.save(payload, path)
    restored = model_spec(config, "meta").build().to_empty(device="cpu")
    load_pretrained_weights(restored, pretrained_source(str(path), config), strict=True)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    payload["model"]["net.feat_rope.freqs_sin"] = sin[:-1]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="training window"):
        load_legacy_model(path, config)


@pytest.mark.parametrize("domain", ["rgb", "latent"])
@pytest.mark.parametrize("legacy_defaults", [False, True, None, []])
def test_legacy_joint_weights_import_strictly_without_optimizer_or_image_branches(
    tmp_path, domain, legacy_defaults
):
    config = parse_config(
        dict(
            initial_weights_format="legacy",
            codec=dict(family="vrae", variant="eupe"),
            world=dict(width=48, depth=1, heads=4),
            conditioner=dict(
                input=domain, input_size=32, width=24, depth=2, heads=4, feature_layers=[1]
            ),
        )
    )
    original = model_spec(config, "cpu").build()
    source = original.state_dict()
    encoder = {
        key.removeprefix("causal_semantic_encoder."): value
        for key, value in source.items()
        if key.startswith("causal_semantic_encoder.")
    }
    encoder["encoder.img_mod_embed"] = torch.zeros(1, 1, 24)
    if domain == "latent":
        encoder["encoder.video_mod_embed"] = torch.zeros(1, 1, 24)
    payload = dict(
        model={key: value for key, value in source.items() if key.startswith("net.")},
        causal_semantic_encoder=encoder,
        optimizer={"must_not_load": True},
        global_step=500,
        meta=dict(
            latent_codec="vrae",
            rae_group_size=1,
            rae_image_latent_channels=768,
            next_latent_use_previous_context=False,
            causal_encoder_input=domain,
            semantic_input_size=32,
            causal_encoder_feature_layers=[1],
        ),
    )
    path = tmp_path / "legacy.pt"
    if legacy_defaults is not False:
        payload["meta"].pop("next_latent_use_previous_context")
        if legacy_defaults is True:
            payload["meta"].pop("causal_encoder_feature_layers")
        else:
            payload["meta"]["causal_encoder_feature_layers"] = legacy_defaults
    torch.save(payload, path)
    restored = model_spec(config, "meta").build().to_empty(device="cpu")
    load_pretrained_weights(restored, pretrained_source(str(path), config), strict=True)
    assert set(restored.state_dict()) == set(source)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, source[key], atol=0, rtol=0)
    payload["causal_semantic_encoder"]["unrecognized.weight"] = torch.ones(1)
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_legacy_model(path, config)
    payload["meta"]["causal_encoder_input"] = "other"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="input domain"):
        load_legacy_model(path, config)
