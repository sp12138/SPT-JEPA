import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from vforge.tasks.rep_world.shared import parse_config, validate_train_request
from vforge.tasks.requests import TrainRequest


@pytest.mark.parametrize("verb", ["train", "eval"])
def test_public_split_config_and_auto_budget(verb):
    from vforge.tasks.common import build_and_snapshot

    assert build_and_snapshot
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name=verb,
            overrides=[
                "experiment=rep_world",
                "checkpoint.enabled=false",
                "model.extras.conditioner.input=latent",
                "model.extras.conditioner.architecture=split_ec",
                "model.extras.conditioner.mask_stage=before_e",
                "model.extras.conditioner.depth=19",
                "model.extras.conditioner.feature_layers=[18]",
                "model.extras.rollout.use_vit_c_kv_cache=true",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    assert (parsed.conditioner.vit_e_depth, parsed.conditioner.vit_c_depth) == (5, 14)
    assert parsed.conditioner.mask_stage == "before_e"
    assert parsed.rollout["use_vit_c_kv_cache"] is True


def test_split_preflight_rejects_legacy_and_usp():
    extras = {"conditioner": {"input": "latent", "architecture": "split_ec"}}
    with pytest.raises(ValueError, match="native weights"):
        parse_config({**extras, "initial_weights_format": "legacy"})
    request = TrainRequest(train_manifest="unused", usp_size=2, extras=extras)
    with pytest.raises(ValueError, match="USP=1"):
        validate_train_request(request, "fsdp2")
