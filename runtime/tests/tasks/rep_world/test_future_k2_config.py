"""Public configuration and phase-switch contract."""

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from vforge.tasks.rep_world.shared import parse_config


@pytest.mark.parametrize("verb", ["train", "eval"])
@pytest.mark.parametrize(
    "layout,p",
    [
        ("single_state", 0),
        ("single_state", 0.5),
        ("left_right_mosaic", 0),
        ("random_token_mosaic", 0),
    ],
)
def test_declared_k2_fields_and_tuning_switch(verb, layout, p):
    from vforge.tasks.common import build_and_snapshot

    assert build_and_snapshot
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name=verb,
            overrides=[
                "experiment=rep_world",
                "model.extras.conditioner.input=latent",
                "model.extras.flow.future_target_choices=2",
                "model.extras.flow.target_origin_reserve=2",
                f"model.extras.flow.target_layout={layout}",
                f"model.extras.flow.distant_target_prob={p}",
                "model.extras.next_state_tuning_start_step=10000",
                "model.extras.ema.enabled=true",
                "model.extras.next_state_tuning_init=ema_reset_optimizer",
                "checkpoint.enabled=false",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    assert parsed.flow.target_count(10) == 8
    assert parsed.next_state_tuning_start_step == 10000


@pytest.mark.parametrize("start", [-1, True, 1.5])
def test_invalid_phase_boundary(start):
    with pytest.raises(ValueError, match="nonnegative"):
        parse_config({"flow": {"target_origin_reserve": 2}, "next_state_tuning_start_step": start})


def test_phase_boundary_requires_explicit_matched_origins():
    with pytest.raises(ValueError, match="explicit"):
        parse_config({"next_state_tuning_start_step": 10000})
