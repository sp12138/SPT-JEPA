from __future__ import annotations

import torch

from vforge.models.video_diffusion import run_flow_match_denoising


class _Scheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([900.0, 100.0])
        self.steps: list[tuple[torch.Tensor, torch.dtype]] = []

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        assert steps == 2
        self.timesteps = self.timesteps.to(device)

    def step(
        self,
        prediction: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        assert return_dict is False
        self.steps.append((timestep, latents.dtype))
        return (latents.float() + prediction.float(),)


def test_flow_match_loop_preserves_cfg_order_routing_and_fp32_accumulation() -> None:
    scheduler = _Scheduler()
    calls: list[tuple[str, float, str, torch.dtype]] = []
    trace: list[tuple[str, int, torch.dtype]] = []

    def predict(current, timestep, conditioning, cache_name):
        expert = "high" if float(timestep) >= 500 else "low"
        calls.append((cache_name, float(timestep), expert, current.dtype))
        value = 2.0 if conditioning == "positive" else 1.0
        return torch.full(current.shape, value, dtype=torch.bfloat16)

    latents = torch.zeros(1, 1, 3, 1, 1, dtype=torch.float32)
    first = torch.full((1, 1, 1, 1, 1), 7.0)
    result = run_flow_match_denoising(
        scheduler=scheduler,
        latents=latents,
        num_inference_steps=2,
        conditioning="positive",
        negative_conditioning="negative",
        predict=predict,
        guidance_scale=2.0,
        frozen_prefix_latents=first,
        trace=lambda stage, step, tensor: trace.append((stage, step, tensor.dtype)),
    )

    assert [(cache, expert) for cache, _time, expert, _dtype in calls] == [
        ("cond", "high"),
        ("uncond", "high"),
        ("cond", "low"),
        ("uncond", "low"),
    ]
    assert all(dtype is torch.float32 for _cache, _time, _expert, dtype in calls)
    assert all(dtype is torch.float32 for _timestep, dtype in scheduler.steps)
    assert result.dtype is torch.float32
    assert torch.equal(result[:, :, :1], first)
    assert torch.equal(result[:, :, 1:], torch.full((1, 1, 2, 1, 1), 6.0))
    assert [(stage, step) for stage, step, _dtype in trace] == [
        ("timestep", 0),
        ("input_latents", 0),
        ("cond_prediction", 0),
        ("uncond_prediction", 0),
        ("guided_prediction", 0),
        ("output_latents", 0),
        ("timestep", 1),
        ("input_latents", 1),
        ("cond_prediction", 1),
        ("uncond_prediction", 1),
        ("guided_prediction", 1),
        ("output_latents", 1),
    ]
