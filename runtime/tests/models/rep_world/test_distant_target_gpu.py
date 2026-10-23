"""Opt-in H200/B300 BF16 FA4 gate for random target horizons, with no fallback."""

from unittest.mock import patch

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.horizon import quadrant_future_targets
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.ops._attention import dispatch


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("mode", ["entry_concat", "global", "same_spatial"])
@pytest.mark.parametrize("mask", [0.0, 0.8])
@pytest.mark.parametrize("layout", ["single_state", "quadrant_mosaic"])
def test_distant_horizons_fa4_backward_and_inference(mode, mask, layout):
    device = torch.device("cuda", 0)
    reset_ops()
    initialize_ops(
        OpsConfig(backend="flash_attention_4", allow_fallback=False, deterministic=False),
        device=device,
        parallel_state=None,
    )
    try:
        torch.manual_seed(42)
        geo = CodecGeometry("test", "test", 16, 8, 4, grid=4)
        enc = ConditionerConfig(
            input="latent",
            width=192,
            depth=2,
            heads=3,
            feature_layers=(1,),
            context_mask_ratio=mask,
        )
        cfg = WorldConfig(
            channels=8,
            state_planes=4,
            grid=4,
            width=192,
            depth=2,
            heads=3,
            condition_width=192,
            condition_dropout=0.1,
            adaln_lora_dim=64,
            condition_mode="entry_concat" if mode == "entry_concat" else "cross_attention",
            cross_attention_scope="same_spatial" if mode == "same_spatial" else "global",
            cross_attention_every=2,
            cross_attention_position="before_self",
            cross_attention_3d_rope=mode != "entry_concat",
        )
        flow = FlowConfig(
            target_layout=layout, distant_target_prob=0.25 if layout == "single_state" else 0
        )
        model = RepWorld(geo, cfg, enc, flow).to(device).train()
        model.gradient_checkpointing_enable()
        torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
        latents = torch.randn(2, 32, 6, 4, 4, device=device)
        target_uniform = torch.tensor([[0.1, 0.8, 0.9], [0.9, 0.1, 0.8]], device=device)
        if layout == "single_state":
            target_kwargs = {"target_uniform": target_uniform}
            inference_offsets = torch.tensor([1, 2], device=device)
        else:
            assignments = torch.tensor(
                [[[2, 0, 3, 1], [1, 3, 0, 2]], [[0, 3, 1, 2], [3, 1, 2, 0]]], device=device
            )
            target_kwargs = {"quadrant_offsets": assignments}
            _, inference_offsets, _ = quadrant_future_targets(
                torch.empty(2, 5, 8, 4, 4, 4, device=device), assignments[:, :1]
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.net.final_layer.linear.weight.detach().clone()
        with (
            patch.object(dispatch, "_run_sdpa", side_effect=AssertionError("Unexpected SDPA")),
            patch.object(
                dispatch, "_run_flash_attention", wraps=dispatch._run_flash_attention
            ) as dense,
            patch.object(
                dispatch, "_run_segmented_attention", wraps=dispatch._run_segmented_attention
            ) as segmented,
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(latents=latents, **target_kwargs, return_metrics=True)
                if layout == "quadrant_mosaic":
                    for gap in range(1, 5):
                        assert result["metrics"][f"target_horizon_{gap}_fraction"] == 0.25
            result["loss"].backward()
            assert torch.isfinite(result["loss"])
            grad = model.causal_semantic_encoder.latent_patch_embed.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            for parameter in model.parameters():
                assert parameter.grad is None or torch.isfinite(parameter.grad).all()
            optimizer.step()
            assert not torch.equal(before, model.net.final_layer.linear.weight)
            assert dense.call_count > 0 and segmented.call_count > 0
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                condition = model(operation="condition", source=latents[:, :, :1])
                x = torch.randn(2, 8, 4, 4, 4, device=device)
                t = torch.full((2, 1), 0.5, device=device)
                out = model(
                    operation="velocity",
                    noisy=x,
                    timestep=t,
                    condition=condition,
                    target_offsets=inference_offsets,
                )
            assert out.shape == x.shape and torch.isfinite(out).all()
    finally:
        reset_ops()
