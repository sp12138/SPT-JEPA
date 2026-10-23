"""Exercise the real Native Trainer/DCP lifecycle with the joint model + Muon."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vforge import (
    CheckpointConfig,
    FSDP2Config,
    ParallelConfig,
    ParallelizeConfig,
    Trainer,
    TrainerConfig,
)
from vforge.ops import OpsConfig
from vforge.tasks.rep_world.train import build_train_components
from vforge.tasks.requests import TrainRequest


class Reader:
    def __len__(self):
        return 40

    def get_batch(self, indices):
        value = torch.tensor(indices, dtype=torch.uint8)[:, None, None, None].expand(
            -1, 256, 256, 3
        )
        return SimpleNamespace(asnumpy=lambda: value.numpy())


class Codec:
    def encode(self, rgb):
        # Deterministic frozen stub; codec-published parity has separate gates.
        return rgb[:, :1, ::4, ::16, ::16].expand(-1, 768, -1, -1, -1).contiguous()


class Reporter:
    def __init__(self):
        self.rows = []

    def report_step(self, payload):
        self.rows.append(payload)

    def close(self):
        pass


@pytest.fixture
def single_thread_cpu():
    # CPU parallel reductions need not be bitwise stable across fresh allocations.
    # Keep this strict checkpoint/RNG gate independent of that kernel scheduling.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "distant_target_prob,target_layout,architecture,choices,reserve,tuning_start",
    [
        (0.0, "single_state", "causal", 3, None, None),
        (0.25, "single_state", "causal", 3, None, None),
        (0.0, "quadrant_mosaic", "causal", 3, None, None),
        (0.0, "single_state", "split_ec", 3, None, None),
        (0.25, "single_state", "split_ec", 3, None, None),
        (0.0, "quadrant_mosaic", "split_ec", 3, None, None),
        (0.5, "single_state", "causal", 2, 2, 1),
        (0.0, "left_right_mosaic", "causal", 2, 2, 1),
        (0.0, "random_token_mosaic", "causal", 2, 2, None),
        (0.0, "random_token_mosaic", "causal", 2, 2, 1),
        (0.0, "single_state", "split_ec", 2, 2, 1),
    ],
)
@pytest.mark.parametrize("ema_enabled", [False, True])
def test_full_dcp_restores_muon_joint_weights_rng_and_real_video_loader(
    tmp_path,
    monkeypatch,
    ema_enabled,
    distant_target_prob,
    target_layout,
    architecture,
    choices,
    reserve,
    tuning_start,
    single_thread_cpu,
    tuning_init="online",
):
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(dict(sample_id=str(i), video=f"{i}.mp4")) for i in range(8))
    )
    monkeypatch.setattr("vforge.tasks.rep_world.data.open_video", lambda path: Reader())
    monkeypatch.setattr(
        "vforge.tasks.rep_world.shared.CodecProvider.get", lambda self, device: Codec()
    )
    request = TrainRequest(
        train_manifest=str(manifest),
        latent_cache_mode="online",
        precision="fp32",
        betas=(0.9, 0.95),
        learning_rate=1e-4,
        extras=dict(
            ema=dict(enabled=ema_enabled, decay=0.99),
            codec=dict(family="vrae", variant="eupe"),
            states=6 if distant_target_prob or target_layout != "single_state" or reserve else 3,
            flow=dict(
                distant_target_prob=distant_target_prob,
                target_layout=target_layout,
                future_target_choices=choices,
                target_origin_reserve=reserve,
            ),
            next_state_tuning_start_step=tuning_start,
            next_state_tuning_init=tuning_init,
            world=dict(width=48, depth=1, heads=4),
            conditioner=dict(
                input="latent" if architecture == "split_ec" else "rgb",
                architecture=architecture,
                context_mask_ratio=0.8 if architecture == "split_ec" else 0,
                input_size=32,
                width=24,
                depth=2,
                heads=4,
                feature_layers=[1],
            ),
        ),
    )

    def run(directory, steps, resume=None):
        components = build_train_components(request)
        live = {}
        factory = components.optimizer_factory

        def optimizer_factory(model):
            live["model"] = model
            live["target_offsets"] = []
            if target_layout == "quadrant_mosaic":
                model.net.register_forward_pre_hook(
                    lambda m, a, kw: live["target_offsets"].append(
                        kw["target_offsets"].detach().clone()
                    ),
                    with_kwargs=True,
                )
            live["optimizer"] = factory(model)
            return live["optimizer"]

        components = replace(components, optimizer_factory=optimizer_factory)
        reporter = Reporter()
        config = TrainerConfig(
            max_steps=steps,
            seed=42,
            precision="fp32",
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(device_type="cpu"), fsdp2=FSDP2Config(enable=False)
            ),
            runtime_ops=OpsConfig(backend="sdpa_math", allow_fallback=False),
            checkpoint=CheckpointConfig(
                enabled=True,
                every_n_steps=1,
                dirpath=str(directory),
                state="full",
                format="dcp",
                async_save=False,
            ),
        )
        Trainer(config).fit(components, resume_path=resume, reporters=(reporter,))
        return live, reporter.rows

    continuous, rows = run(tmp_path / "continuous", 3)
    run(tmp_path / "split", 1)
    resumed, tail = run(tmp_path / "split", 3, str(tmp_path / "split" / "00001.dcp"))
    assert [row["step"] for row in tail] == [1, 2]
    assert [row["loss"] for row in tail] == [row["loss"] for row in rows[1:]]
    for key, value in continuous["model"].state_dict().items():
        assert torch.equal(value, resumed["model"].state_dict()[key]), key
    if ema_enabled:
        assert continuous["model"].ema.num_updates.item() == 3
        assert resumed["model"].ema.num_updates.item() == 3
        assert rows[-1]["metrics"]["ema_updates"] == 3
        from vforge.evaluation.runner import _load_initial_weights
        from vforge.tasks.rep_world.ema import EmaConfig, load_evaluation_ema
        from vforge.training.checkpoint import TrainingCheckpointWeightSource

        evaluated = copy.deepcopy(continuous["model"])
        del evaluated.ema
        with torch.no_grad():
            for parameter in evaluated.parameters():
                parameter.add_(99)
        source = TrainingCheckpointWeightSource(path=str(tmp_path / "continuous" / "00003.dcp"))
        _load_initial_weights(
            evaluated,
            tuple(replace(source, model_path=name) for name in ("net", "causal_semantic_encoder")),
            weights_strict=True,
        )
        for actual, expected in zip(
            evaluated.parameters(), continuous["model"].parameters(), strict=True
        ):
            assert torch.equal(actual, expected)
        evaluated.requires_grad_(False)
        count = load_evaluation_ema(evaluated, EmaConfig(enabled=True), source)
        assert count == 3
        for actual, expected in zip(
            evaluated.parameters(), continuous["model"].ema.shadows(), strict=True
        ):
            assert torch.equal(actual, expected)
        assert any(
            not torch.equal(actual, online)
            for actual, online in zip(
                evaluated.parameters(), continuous["model"].parameters(), strict=True
            )
        )

    if distant_target_prob:
        for resumed_row, original_row in zip(tail, rows[1:], strict=True):
            for gap in range(1, choices + 1):
                for key in (
                    f"target_horizon_{gap}_fraction",
                    f"loss_fm_horizon_{gap}_contribution",
                ):
                    assert resumed_row["metrics"][key] == original_row["metrics"][key]
            assert resumed_row["step_metadata"]["prediction_positions"] == (4 if reserve else 3)

    if target_layout == "quadrant_mosaic":
        for a, b in zip(continuous["target_offsets"][1:], resumed["target_offsets"], strict=True):
            assert torch.equal(a, b)
        for row in tail:
            assert row["step_metadata"]["prediction_positions"] == 2
            assert row["step_metadata"]["prediction_quadrants"] == 8
            for gap in range(1, 5):
                assert row["metrics"][f"target_horizon_{gap}_fraction"] == 0.25

    if tuning_start is not None:
        for row in tail:
            assert row["step_metadata"]["prediction_positions"] == 4
            assert row["step_metadata"]["target_phase"] == "next_state_tuning"
            assert row["metrics"]["target_horizon_1_fraction"] == 1
            assert "prediction_halves" not in row["step_metadata"]
        assert rows[0]["step_metadata"]["target_phase"] == "main"
        if target_layout == "left_right_mosaic":
            assert rows[0]["metrics"]["target_horizon_1_fraction"] == 0.5
            assert rows[0]["step_metadata"]["prediction_halves"] == 8

    original = continuous["optimizer"].state_dict()
    restored = resumed["optimizer"].state_dict()
    assert original["param_groups"] == restored["param_groups"]
    for parameter, state in original["state"].items():
        for key, value in state.items():
            actual = restored["state"][parameter][key]
            assert (
                torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
            )


@pytest.mark.parametrize(
    "layout,probability",
    [("single_state", 0.0), ("single_state", 0.5), ("random_token_mosaic", 0.0)],
)
def test_full_resume_across_ema_tuning_initialization(
    tmp_path, monkeypatch, single_thread_cpu, layout, probability
):
    test_full_dcp_restores_muon_joint_weights_rng_and_real_video_loader(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        ema_enabled=True,
        distant_target_prob=probability,
        target_layout=layout,
        architecture="causal",
        choices=2,
        reserve=2,
        tuning_start=1,
        single_thread_cpu=single_thread_cpu,
        tuning_init="ema_reset_optimizer",
    )
