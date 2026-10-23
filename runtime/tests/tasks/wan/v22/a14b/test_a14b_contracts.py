"""Small-model contracts for the Wan22 A14B expert pair.

The two things A14B adds over Wan2.1 are timestep-ranged expert routing and a
root assembled from two published checkpoints, so that is what these pin down.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

_EXTRAS = {
    "noise_shift": 3.0,
    "boundary_ratio": 0.9,
    "text_drop_prob": 0.0,
    "rgb_cond_drop_prob": 0.0,
    "normal_drop_prob": 0.0,
}


class _Expert(nn.Module):
    """Stands in for a 14B tower; records the timesteps routed to it."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag
        self.seen: list[float] = []
        self.weight = nn.Parameter(torch.zeros(1))
        self.checkpointing = 0

    def forward(self, *, timestep: torch.Tensor, **_kwargs):
        self.seen.append(float(timestep.flatten()[0]))
        return self.tag

    def enable_gradient_checkpointing(self, *, gradient_checkpointing_func=None) -> None:
        assert callable(gradient_checkpointing_func)
        self.checkpointing += 1


def _model(**kwargs):
    from vforge.models.wan.v22.a14b.model import Wan22A14bModel

    return Wan22A14bModel(_Expert("high"), _Expert("low"), **kwargs)


def _processor(**overrides):
    from vforge.tasks.wan.v22.a14b.train import _A14bConditionProcessor

    processor = _A14bConditionProcessor(
        extras={**_EXTRAS, **overrides},
        precision="fp32",
    )
    processor._ensure_diffusion_helpers()
    return processor


def _sampled_timesteps(processor, steps: range) -> list[float]:
    scheduler = processor._scheduler
    return [
        float(
            scheduler.timesteps[
                processor._sample_timestep_index(
                    step=step,
                    device=torch.device("cpu"),
                    generator=None,
                )
            ]
        )
        for step in steps
    ]


def test_condition_input_channels_follow_the_masked_stream_layout() -> None:
    from vforge.tasks.wan.v22.a14b.shared import condition_spec

    # The published A14B checkpoint is the single-rgb-stream width.
    assert condition_spec([]).input_channels(z_dim=16) == 16
    assert condition_spec(["rgb"]).input_channels(z_dim=16) == 36
    assert condition_spec(["normal"]).input_channels(z_dim=16) == 36
    assert condition_spec(["rgb", "normal"]).input_channels(z_dim=16) == 56


def test_an_unknown_stream_names_what_this_model_encodes() -> None:
    from vforge.tasks.wan.v22.a14b.shared import condition_streams

    with pytest.raises(ValueError, match=r"this model encodes \['rgb', 'normal'\]"):
        condition_streams(["grid"])


def test_forward_routes_each_timestep_to_the_expert_that_owns_it() -> None:
    model = _model(boundary_ratio=0.9, num_train_timesteps=1000)

    assert model.forward(timestep=torch.tensor([950.0])) == "high"
    assert model.forward(timestep=torch.tensor([900.0])) == "high"
    assert model.forward(timestep=torch.tensor([899.0])) == "low"
    assert model.forward(timestep=torch.tensor([10.0])) == "low"
    assert model.wan.seen == [950.0, 900.0]
    assert model.wan_2.seen == [899.0, 10.0]


def test_forward_rejects_a_micro_batch_that_straddles_the_boundary() -> None:
    model = _model()

    with pytest.raises(ValueError, match="one shared timestep"):
        model.forward(timestep=torch.tensor([950.0, 10.0]))


def test_per_token_timesteps_route_as_one_when_they_agree() -> None:
    model = _model()

    assert model.forward(timestep=torch.full((2, 8), 950.0)) == "high"


def test_both_trainable_experts_alternate_by_global_step_parity() -> None:
    processor = _processor()
    boundary = 0.9 * 1000

    sampled = _sampled_timesteps(processor, range(8))
    high = [value for index, value in enumerate(sampled) if index % 2 == 0]
    low = [value for index, value in enumerate(sampled) if index % 2 == 1]

    assert all(value >= boundary for value in high)
    assert all(value < boundary for value in low)


def test_single_expert_training_samples_only_its_own_range() -> None:
    boundary = 0.9 * 1000

    high_only = _processor(train_expert="high")
    assert all(value >= boundary for value in _sampled_timesteps(high_only, range(6)))

    low_only = _processor(train_expert="low")
    assert all(value < boundary for value in _sampled_timesteps(low_only, range(6)))


def test_sampled_timesteps_cover_the_whole_range_they_are_allowed() -> None:
    """A restricted draw must still spread, not collapse onto one index."""
    torch.manual_seed(0)
    processor = _processor(train_expert="low")

    sampled = _sampled_timesteps(processor, range(64))

    assert len(set(sampled)) > 20


def test_base_wan21_sampling_stays_unrestricted() -> None:
    """The hook A14B overrides must not have narrowed Wan2.1's own draw."""
    from vforge.tasks.wan.v21.train import _ConditionProcessor

    processor = _ConditionProcessor(extras={"noise_shift": 3.0}, precision="fp32")
    processor._ensure_diffusion_helpers()
    torch.manual_seed(0)

    sampled = _sampled_timesteps(processor, range(64))

    assert min(sampled) < 0.9 * 1000 <= max(sampled)


def test_gradient_checkpointing_and_fsdp_roots_reach_both_experts() -> None:
    from vforge.distributed.parallelize import (
        FSDP2Config,
        ParallelizeConfig,
        parallelize_model,
    )

    model = _model()
    parallelize_model(
        model,
        ParallelizeConfig(
            fsdp2=FSDP2Config(enable=False),
            gradient_checkpointing=True,
        ),
    )

    assert model.fsdp_execution_roots() == (model.wan, model.wan_2)
    assert (model.wan.checkpointing, model.wan_2.checkpointing) == (1, 1)


def test_boundary_ratio_must_leave_both_experts_reachable() -> None:
    with pytest.raises(ValueError, match="boundary_ratio"):
        _model(boundary_ratio=1.0)


def test_validate_rejects_legacy_expert_booleans() -> None:
    from vforge.tasks.wan.v22.a14b.shared import _validate

    with pytest.raises(ValueError, match="deprecated Wan22 A14B expert flags"):
        _validate(
            extras={
                **_EXTRAS,
                "train_high_noise_expert": False,
                "train_low_noise_expert": False,
            },
            latent_cache_mode="offline",
            usp_size=1,
        )


def test_validate_rejects_an_unknown_train_expert() -> None:
    from vforge.tasks.wan.v22.a14b.shared import _validate

    with pytest.raises(ValueError, match="train_expert must be one of"):
        _validate(
            extras={**_EXTRAS, "train_expert": "sometimes"},
            latent_cache_mode="offline",
            usp_size=1,
        )


@pytest.mark.parametrize("expert", ["high", "low"])
def test_mcore_train_preflight_accepts_single_expert(expert) -> None:
    from vforge.tasks.requests import TrainRequest
    from vforge.tasks.wan.v22.a14b.train import validate_train_request

    validate_train_request(
        TrainRequest(extras={**_EXTRAS, "train_expert": expert}),
        "megatron",
    )


def test_mcore_train_preflight_rejects_both_but_fsdp_accepts_it() -> None:
    from vforge.tasks.requests import TrainRequest
    from vforge.tasks.wan.v22.a14b.train import validate_train_request

    request = TrainRequest(extras={**_EXTRAS, "train_expert": "both"})
    with pytest.raises(NotImplementedError, match="train_expert=high or low"):
        validate_train_request(request, "megatron")
    validate_train_request(request, "fsdp2")


def test_single_expert_root_registers_only_that_checkpoint_prefix() -> None:
    from vforge.models.wan.v22.a14b.model import Wan22A14bModel

    high = Wan22A14bModel(wan=_Expert("high"))
    low = Wan22A14bModel(wan_2=_Expert("low"))

    assert set(high.state_dict()) == {"wan.weight"}
    assert set(low.state_dict()) == {"wan_2.weight"}
    assert high.fsdp_execution_roots() == (high.wan,)
    assert low.fsdp_execution_roots() == (low.wan_2,)
    with pytest.raises(RuntimeError, match="cannot execute low-noise"):
        high(timestep=torch.tensor([10.0]))
    with pytest.raises(RuntimeError, match="cannot execute high-noise"):
        low(timestep=torch.tensor([950.0]))


def test_validate_rejects_image_branch_conditioning() -> None:
    from vforge.tasks.wan.v22.a14b.shared import _validate

    with pytest.raises(ValueError, match="no image conditioning branch"):
        _validate(
            extras={**_EXTRAS, "use_clip_condition": True},
            latent_cache_mode="offline",
            usp_size=1,
        )


def test_widening_keeps_published_channels_and_zeroes_the_new_tail() -> None:
    from vforge.models.wan.model import WanChannelLayout
    from vforge.tasks.wan.v22.a14b.train import _widen_patch_embedding

    layout = WanChannelLayout(z_dim=16, pretrained_in_channels=36, in_channels=56)
    source = torch.randn(8, 36, 1, 2, 2)
    target = torch.zeros(8, 56, 1, 2, 2)

    widened = _widen_patch_embedding(layout)("wan.patch_embedding.weight", source, target)

    assert widened is not None
    assert widened.shape == target.shape
    assert torch.equal(widened[:, :36], source)
    assert torch.count_nonzero(widened[:, 36:]) == 0


def test_widening_leaves_every_other_key_alone() -> None:
    from vforge.models.wan.model import WanChannelLayout
    from vforge.tasks.wan.v22.a14b.train import _widen_patch_embedding

    layout = WanChannelLayout(z_dim=16, pretrained_in_channels=36, in_channels=56)
    resolve = _widen_patch_embedding(layout)

    assert resolve("blocks.0.weight", torch.randn(2, 2), torch.zeros(4, 4)) is None


def _published_checkpoint(root, *, in_channels: int = 36) -> str:
    """A config-only stand-in for the two-subfolder A14B export."""
    import json

    config = {
        "_class_name": "WanTransformer3DModel",
        "in_channels": in_channels,
        "out_channels": 16,
        "num_layers": 2,
        "patch_size": [1, 2, 2],
    }
    for subfolder in ("transformer", "transformer_2"):
        directory = root / subfolder
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(root)


def test_each_expert_gets_its_own_narrowed_weight_source(tmp_path) -> None:
    from vforge.tasks.wan.v22.a14b.shared import _resolve_expert_configs
    from vforge.tasks.wan.v22.a14b.train import _a14b_pretrained_sources

    root = _published_checkpoint(tmp_path)
    _, layouts = _resolve_expert_configs({"condition_streams": ["rgb"]}, root)
    sources = _a14b_pretrained_sources(
        layouts=layouts,
        pretrained_path=root,
        weight_loading="auto",
    )

    assert [source.model_path for source in sources] == ["wan", "wan_2"]
    assert [source.subfolder for source in sources] == ["transformer", "transformer_2"]
    # Each expert owns its whole key space, so nothing is re-prefixed on load.
    assert {source.model_prefix for source in sources} == {""}


def test_single_expert_reads_and_loads_only_its_published_subfolder(tmp_path) -> None:
    from vforge.tasks.wan.v22.a14b.shared import _resolve_expert_configs
    from vforge.tasks.wan.v22.a14b.train import _a14b_pretrained_sources

    root = _published_checkpoint(tmp_path)
    _, layouts = _resolve_expert_configs(
        {"condition_streams": ["rgb"]},
        root,
        expert_names=("wan_2",),
    )
    sources = _a14b_pretrained_sources(
        layouts=layouts,
        pretrained_path=root,
        weight_loading="auto",
        expert_names=("wan_2",),
    )

    assert list(layouts) == ["wan_2"]
    assert [(source.model_path, source.subfolder) for source in sources] == [
        ("wan_2", "transformer_2")
    ]


def _expert_checkpoint_manifest(path, expert: str, *, step: int) -> None:
    from vforge.training.checkpoint.manifest import CheckpointManifest, write_manifest

    path.mkdir()
    manifest = CheckpointManifest.create(
        state="weights",
        completed_steps=step,
        fingerprints={
            "semantic_config": {
                "model": {
                    "name": "wan22_a14b",
                    "extras": {
                        **_EXTRAS,
                        "condition_streams": ["rgb"],
                        "train_expert": expert,
                    },
                }
            }
        },
    )
    write_manifest(str(path), manifest)


def test_paired_eval_accepts_independently_completed_expert_checkpoints(tmp_path) -> None:
    from vforge.tasks.wan.v22.a14b.eval import _validate_checkpoint_pair

    high = tmp_path / "high.dcp"
    low = tmp_path / "low.dcp"
    _expert_checkpoint_manifest(high, "high", step=11)
    _expert_checkpoint_manifest(low, "low", step=29)

    manifests = _validate_checkpoint_pair(
        {"high": str(high), "low": str(low)},
        {**_EXTRAS, "condition_streams": ["rgb"]},
    )

    assert manifests["high"].completed_steps == 11
    assert manifests["low"].completed_steps == 29


def test_paired_eval_rejects_a_checkpoint_with_the_wrong_expert_identity(tmp_path) -> None:
    from vforge.tasks.wan.v22.a14b.eval import _validate_checkpoint_pair

    high = tmp_path / "high.dcp"
    low = tmp_path / "low.dcp"
    _expert_checkpoint_manifest(high, "low", step=1)
    _expert_checkpoint_manifest(low, "low", step=2)

    with pytest.raises(ValueError, match="identifies train_expert='low'"):
        _validate_checkpoint_pair(
            {"high": str(high), "low": str(low)},
            {**_EXTRAS, "condition_streams": ["rgb"]},
        )


@pytest.mark.parametrize(
    ("checkpoint_path", "message"),
    [
        ({"high": "high.dcp"}, "requires exactly"),
        ({"high": "same.dcp", "low": "same.dcp"}, "must be different"),
    ],
)
def test_paired_eval_rejects_incomplete_or_duplicate_paths(checkpoint_path, message) -> None:
    from vforge.tasks.requests import EvalRequest
    from vforge.tasks.wan.v22.a14b.eval import _checkpoint_pair

    request = EvalRequest(checkpoint_path=checkpoint_path)

    with pytest.raises(ValueError, match=message):
        _checkpoint_pair(request)


def test_a_narrowed_source_only_writes_its_own_tower() -> None:
    from vforge.models.loading import PretrainedWeightSource, load_pretrained_weights

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wan = nn.Linear(2, 2, bias=False)
            self.wan_2 = nn.Linear(2, 2, bias=False)

    root = Root()
    with torch.no_grad():
        root.wan.weight.fill_(0.0)
        root.wan_2.weight.fill_(0.0)
    donor = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        donor.weight.fill_(7.0)

    load_pretrained_weights(
        root,
        PretrainedWeightSource(
            path="unused",
            model_path="wan_2",
            loading_mode="rank0_broadcast",
            fallback_model_factory=lambda: donor,
        ),
    )

    assert torch.count_nonzero(root.wan.weight) == 0
    assert torch.equal(root.wan_2.weight, torch.full((2, 2), 7.0))


def test_train_components_declare_one_weight_source_per_expert() -> None:
    from vforge.models.loading import PretrainedWeightSource
    from vforge.training.recipe import _weight_sources

    single = PretrainedWeightSource(path="a")
    pair = (PretrainedWeightSource(path="a"), PretrainedWeightSource(path="b"))

    assert _weight_sources(None) == ()
    assert _weight_sources(single) == (single,)
    assert _weight_sources(pair) == pair


def test_config_resolution_widens_only_past_the_published_width(tmp_path) -> None:
    from vforge.tasks.wan.v22.a14b.shared import _resolve_expert_configs

    root = _published_checkpoint(tmp_path)

    configs, layouts = _resolve_expert_configs({"condition_streams": ["rgb"]}, root)
    assert configs["wan"]["in_channels"] == 36
    assert layouts["wan"].is_widened is False

    configs, layouts = _resolve_expert_configs({"condition_streams": ["rgb", "normal"]}, root)
    assert configs["wan_2"]["in_channels"] == 56
    assert configs["wan_2"]["use_vae_condition"] is False
    assert layouts["wan_2"].pretrained_in_channels == 36
    assert layouts["wan_2"].is_widened is True


def test_a_mode_narrower_than_the_checkpoint_is_refused(tmp_path) -> None:
    from vforge.tasks.wan.v22.a14b.shared import _resolve_expert_configs

    with pytest.raises(ValueError, match="discard published weights"):
        _resolve_expert_configs({"condition_streams": []}, _published_checkpoint(tmp_path))


def test_both_experts_are_read_once_per_build(tmp_path, monkeypatch) -> None:
    """One read per checkpoint: the layouts and configs share it."""
    from vforge.models.wan.model import read_transformer_config
    from vforge.models.wan.v22.a14b import model as a14b_model
    from vforge.tasks.wan.v22.a14b.shared import _resolve_expert_configs

    reads: list[str] = []

    def counting_read(transformer_type, path, *, subfolder="transformer"):
        reads.append(subfolder)
        return read_transformer_config(transformer_type, path, subfolder=subfolder)

    monkeypatch.setattr(a14b_model, "read_transformer_config", counting_read)
    _resolve_expert_configs({"condition_streams": ["rgb"]}, _published_checkpoint(tmp_path))

    assert reads == ["transformer", "transformer_2"]
