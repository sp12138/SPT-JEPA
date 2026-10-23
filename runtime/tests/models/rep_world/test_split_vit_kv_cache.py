from types import SimpleNamespace

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.inference import RepWorldInference, RolloutConfig
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.models.rep_world.split_conditioner import SplitViTConditioner
from vforge.ops import OpsConfig, initialize_ops, reset_ops


@pytest.fixture(autouse=True)
def ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"), parallel_state=None,
    )
    yield
    reset_ops()


def test_cached_prefill_matches_full_state_causal_attention():
    torch.manual_seed(23)
    geometry = CodecGeometry("test", "tiny", 4, 4, 1, grid=2)
    config = ConditionerConfig(
        input="latent", width=24, depth=3, heads=4,
        feature_layers=(2,), architecture="split_ec", encoder_depth=1,
        context_mask_ratio=0.9,
    )
    conditioner = SplitViTConditioner(config, geometry).eval()
    encoded = torch.randn(2, 4, geometry.tokens_per_state, config.width)
    cache = None
    with torch.inference_mode():
        for index in range(encoded.shape[1]):
            current, cache = conditioner.prefill_cached(
                encoded[:, index : index + 1], cache=cache, start_state=index,
            )
            reference = conditioner.prefill(encoded[:, : index + 1])[:, -1]
            torch.testing.assert_close(current, reference, atol=1e-5, rtol=1e-5)
            assert all(
                k.shape[1] == (index + 1) * geometry.tokens_per_state for k, _ in cache
            )


def test_cached_prefill_rejects_training_and_grad_enabled():
    geometry = CodecGeometry("test", "tiny", 4, 4, 1, grid=2)
    config = ConditionerConfig(
        input="latent", width=24, depth=3, heads=4,
        feature_layers=(2,), architecture="split_ec", encoder_depth=1,
    )
    conditioner = SplitViTConditioner(config, geometry)
    encoded = torch.randn(1, 1, geometry.tokens_per_state, config.width)
    with pytest.raises(ValueError, match="inference-only"):
        conditioner.prefill_cached(encoded)
    conditioner.eval()
    with pytest.raises(ValueError, match="inference-only"):
        conditioner.prefill_cached(encoded)


def test_full_rollout_cache_matches_uncached():
    torch.manual_seed(47)
    geometry = CodecGeometry("test", "tiny", 4, 4, 1, grid=2)
    conditioner = ConditionerConfig(
        input="latent", width=24, depth=3, heads=4,
        feature_layers=(2,), architecture="split_ec", encoder_depth=1,
    )
    world = WorldConfig(
        channels=4, state_planes=1, grid=2, width=24, depth=1,
        heads=4, condition_width=24, cross_attention_3d_rope=True,
    )
    flow = FlowConfig()
    model = RepWorld(geometry, world, conditioner, flow).eval()
    with torch.no_grad():
        model.net.final_layer.linear.weight.normal_(std=0.01)

    class Codec:
        def encode(self, rgb):
            return rgb.new_full((rgb.shape[0], 4, 1, 2, 2), 0.125)

        def stream(self):
            return SimpleNamespace(
                prime=lambda _: None,
                step=lambda packed: packed.new_zeros(
                    packed.shape[0], 3, 4, 32, 32
                ),
            )

    sample = {"sample_id": "same", "rgb": torch.randn(1, 3, 16, 32, 32)}
    outputs = []
    for cached in (False, True):
        config = RolloutConfig(
            total_states=4, history_states=1, history_window_states=3,
            trained_states=4, steps=3, cfg_scale=1,
            use_vit_c_kv_cache=cached,
        )
        inference = RepWorldInference(
            model, Codec(), geometry, flow, conditioner, model.timestep_shift, config
        )
        with torch.inference_mode():
            outputs.append(inference.infer_sample(sample, seed=67)["latents"])
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-5, rtol=1e-5)
