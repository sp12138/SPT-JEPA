"""Public composition and train/eval shared model configuration."""

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from vforge.tasks.rep_world.shared import parse_config


@pytest.mark.parametrize("verb", ["train", "eval"])
@pytest.mark.parametrize("position", ["before_self", "after_self"])
def test_cross_attention_order_composes_and_reaches_model(verb, position):
    from vforge.tasks.common import build_and_snapshot

    assert build_and_snapshot
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name=verb,
            overrides=[
                "experiment=rep_world",
                "model.extras.world.cross_attention_every=12",
                "model.extras.world.cross_attention_position=" + position,
                "checkpoint.enabled=false",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    assert parsed.world.cross_attention_position == position
    assert parsed.world.cross_attention_every == parsed.world.depth
