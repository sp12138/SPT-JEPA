import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from vforge.tasks.rep_world.shared import parse_config


@pytest.mark.parametrize("verb", ["train", "eval"])
@pytest.mark.parametrize("probability", [0, 0.25, 1])
def test_public_distant_target_probability(verb, probability):
    from vforge.tasks.common import build_and_snapshot

    assert build_and_snapshot
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name=verb,
            overrides=[
                "experiment=rep_world",
                f"model.extras.flow.distant_target_prob={probability}",
                "checkpoint.enabled=false",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    assert parsed.flow.distant_target_prob == probability
    assert parsed.flow.target_count(10) == (7 if probability else 9)


def test_short_clip_is_rejected_before_allocating_model():
    with pytest.raises(ValueError, match="at least 4"):
        parse_config({"flow": {"distant_target_prob": 0.25}, "states": 3})


@pytest.mark.parametrize("verb", ["train", "eval"])
def test_public_quadrant_layout(verb):
    from vforge.tasks.common import build_and_snapshot

    assert build_and_snapshot
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name=verb,
            overrides=[
                "experiment=rep_world",
                "model.extras.flow.target_layout=quadrant_mosaic",
                "checkpoint.enabled=false",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    assert parsed.flow.target_layout == "quadrant_mosaic"
    assert parsed.flow.target_count(10) == 6


def test_quadrant_preflight_rejects_short_or_nonspatial_codec():
    with pytest.raises(ValueError, match="at least 5"):
        parse_config({"flow": {"target_layout": "quadrant_mosaic"}, "states": 4})
    with pytest.raises(ValueError, match="spatially ordered"):
        parse_config(
            {
                "flow": {"target_layout": "quadrant_mosaic"},
                "codec": {"family": "videorae", "variant": "1d"},
            }
        )
