import copy
from dataclasses import replace

import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from vforge.tasks.rep_world.ema import EmaConfig, initialize_ema, update_ema, validate_ema
from vforge.tasks.rep_world.shared import parse_config


def test_joint_ema_formula_from_initial_weights_and_resume():
    root = torch.nn.Module()
    root.net = torch.nn.Linear(3, 2)
    root.causal_semantic_encoder = torch.nn.Linear(2, 3)
    config = EmaConfig(enabled=True, decay=0.99)
    initial = [p.detach().clone() for p in root.parameters()]
    ema = initialize_ema(root, config)
    assert ema.num_updates.item() == 0 and ema.initialized.item()
    assert all(s.dtype == torch.float32 for s in ema.shadows())
    with torch.no_grad():
        for p in root.parameters():
            p.add_(2)
    update_ema(root, 0, None, None, config=config)
    for s, p in zip(ema.shadows(), initial, strict=True):
        torch.testing.assert_close(s, p * 0.99 + (p + 2) * 0.01)
    saved = copy.deepcopy(root.state_dict())
    other = torch.nn.Module()
    other.net = torch.nn.Linear(3, 2)
    other.causal_semantic_encoder = torch.nn.Linear(2, 3)
    initialize_ema(other, config)
    other.load_state_dict(saved, strict=True)
    for target in (root, other):
        with torch.no_grad():
            for p in target.parameters():
                p.mul_(0.8)
        update_ema(target, 1, None, None, config=config)
    for key, value in root.state_dict().items():
        assert torch.equal(value, other.state_dict()[key])
    with pytest.raises(ValueError, match="update count"):
        update_ema(root, 3, None, None, config=config)
    with pytest.raises(ValueError, match="decay"):
        validate_ema(root, replace(config, decay=0.9))
    root.ema.copy_to(root)
    for p, s in zip(root.parameters(), root.ema.shadows(), strict=True):
        assert torch.equal(p, s)


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": 1},
        {"decay": -0.1},
        {"decay": 1},
        {"decay": float("nan")},
        {"evaluation_weights": "other"},
        {"evaluation_weights": "ema"},
    ],
)
def test_invalid_ema_config_fails(options):
    with pytest.raises(ValueError):
        EmaConfig(**options)


@pytest.mark.parametrize("verb", ["train", "eval"])
@pytest.mark.parametrize("weights", ["online", "ema"])
def test_ema_normal_overrides_are_declared(verb, weights):
    from vforge.tasks.common import build_and_snapshot

    assert build_and_snapshot
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name=verb,
            overrides=[
                "experiment=rep_world",
                "model.extras.ema.enabled=true",
                "model.extras.ema.decay=0.99",
                "model.extras.ema.evaluation_weights=" + weights,
                "checkpoint.enabled=false",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    assert parsed.ema == EmaConfig(enabled=True, decay=0.99, evaluation_weights=weights)
