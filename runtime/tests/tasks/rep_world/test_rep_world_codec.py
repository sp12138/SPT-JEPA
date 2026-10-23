from copy import deepcopy

import pytest
import torch

from vforge.tasks.rep_world.codec import build_videorae, build_vrae, load_videorae, load_vrae


def release_config(variant="eupe"):
    eupe = variant == "eupe"
    return {
        "encoder": {
            "name": variant,
            "variant": "vit_b16" if eupe else "vit_l16",
            "layers": list(range(5, 12)) if eupe else [11, 13, 15, 17, 19, 21, 23],
            "fusion": "sum" if eupe else "mean_plus_final_spatial_mean",
            "hidden_size": 768 if eupe else 1024,
            "num_blocks": 12 if eupe else 24,
            "patch_size": 16,
            "encoder_tubelet_size": 1 if eupe else 2,
            "pixel_normalization": "imagenet",
        },
        "pooling": {
            "name": "temporal_attention",
            "group_size": 4 if eupe else 2,
            "num_heads": 16,
            "output_norm_affine": False,
        },
        "decoder": {
            "name": "vrae_decoder",
            "hidden_size": 24,
            "depth": 1,
            "num_heads": 4,
            "attention_mode": "chunk_causal" if eupe else "full",
            "init": "scratch",
            "attention_backend": "auto",
        },
    }


@pytest.mark.parametrize("variant", ["eupe", "vjepa2_1"])
def test_published_config_translates_without_backbone_download(variant):
    with torch.device("meta"):
        model = build_vrae(release_config(variant), variant)
    assert all(p.is_meta and not p.requires_grad for p in model.parameters())
    assert model.temporal_pool.group_size * (1 if variant == "eupe" else 2) == 4
    assert model.decoder.config.attention_mode == ("chunk_causal" if variant == "eupe" else "full")
    assert not any("patch_embed_img" in name for name in model.state_dict())


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("encoder", "fusion", "mean"),
        ("encoder", "layers", [11]),
        ("encoder", "pixel_normalization", "none"),
        ("pooling", "group_size", 2),
        ("pooling", "output_norm_affine", True),
        ("decoder", "rope_theta", 2000),
        ("decoder", "unrecognized_setting", 1),
    ],
)
def test_changed_published_codec_contract_is_not_silently_accepted(section, key, value):
    config = deepcopy(release_config())
    config[section][key] = value
    with torch.device("meta"), pytest.raises(ValueError):
        build_vrae(config, "eupe")


def test_missing_ema_is_not_replaced_by_online_weights(tmp_path):
    path = tmp_path / "incomplete.pt"
    torch.save({"model": {}, "resolved_config": {"model": release_config()}}, path)
    with pytest.raises(ValueError, match="EMA shadow"):
        load_vrae(path, "eupe", device="cpu")


@pytest.mark.parametrize("variant", ["3d", "1d"])
def test_video_release_ignores_generic_but_rejects_changed_active_token_count(variant):
    name = "autoencoder_vjepa21_3d_c64_repa" if variant == "3d" else "autoencoder_vfm_ae_c32_repa"
    config = dict(name=name, args=dict(bottleneck_token_num=1024))
    with torch.device("meta"):
        model = build_videorae(config, variant)
        if variant == "1d":
            assert model.tokenizer_encoder.tokens == 512
        key = "latent_temporal" if variant == "3d" else "num_latent_tokens"
        config["args"][key] = 99
        with pytest.raises(ValueError, match=key):
            build_videorae(config, variant)


def test_video_loader_does_not_accept_incomplete_or_wrong_variant_state(tmp_path):
    path = tmp_path / "incomplete.pt"
    torch.save(dict(model=dict(name="autoencoder_vfm_ae_c32_repa", args={}, sd={})), path)
    with pytest.raises(ValueError, match="variant"):
        load_videorae(path, "3d", device="cpu")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_videorae(path, "1d", device="cpu")
