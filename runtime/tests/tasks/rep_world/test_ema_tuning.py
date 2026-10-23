import pytest
import torch

from vforge.tasks.rep_world.ema import EmaConfig, initialize_ema, initialize_tuning_from_ema
from vforge.tasks.rep_world.optimizer import MuonWithAuxAdam
from vforge.tasks.rep_world.shared import parse_config


def test_ema_tuning_copies_shadow_and_resets_both_optimizers_without_changing_lr():
    torch.manual_seed(3)
    model = torch.nn.Linear(4, 3)
    config = EmaConfig(enabled=True, decay=0.99)
    initial = [p.detach().clone() for p in model.parameters()]
    ema = initialize_ema(model, config)
    optimizer = MuonWithAuxAdam(model, lr=1e-4, weight_decay=0)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    updated = [p.detach().clone() for p in model.parameters()]
    ema.update(model, 0)
    assert optimizer.state
    groups = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
    metrics = initialize_tuning_from_ema(model, optimizer, config, 1)
    for p, before, after in zip(model.parameters(), initial, updated, strict=True):
        torch.testing.assert_close(p, before.lerp(after, 0.01), rtol=1e-6, atol=1e-7)
    assert not optimizer.state
    assert groups == [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
    assert int(ema.num_updates) == 1
    assert metrics["tuning_init_ema"] == metrics["tuning_optimizer_reset"] == 1
    assert metrics["tuning_source_ema_updates"] == 1
    with pytest.raises(ValueError, match="phase boundary"):
        initialize_tuning_from_ema(model, optimizer, config, 2)


@pytest.mark.parametrize(
    "extras",
    [
        {"next_state_tuning_init": "bad"},
        {"next_state_tuning_init": "ema_reset_optimizer"},
        {
            "next_state_tuning_init": "ema_reset_optimizer",
            "flow": {"target_origin_reserve": 2},
            "next_state_tuning_start_step": 10000,
        },
    ],
)
def test_invalid_ema_tuning_configuration(extras):
    with pytest.raises(ValueError, match="tuning|EMA"):
        parse_config(extras)
