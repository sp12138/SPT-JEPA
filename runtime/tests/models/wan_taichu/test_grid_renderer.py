from __future__ import annotations

import pytest
import torch

from vforge.models.wan_taichu.grid_renderer import render_infinite_grid_video


def _camera_inputs(frame_count: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    c2ws = torch.eye(4).repeat(frame_count, 1, 1)
    c2ws[:, 1, 3] = 1.5
    c2ws[:, 0, 3] = torch.arange(frame_count) * 0.25
    intrinsics = torch.tensor([[24.0, 0.0, 12.0], [0.0, 24.0, 8.0], [0.0, 0.0, 1.0]]).repeat(
        frame_count, 1, 1
    )
    return c2ws, intrinsics


def test_grid_renderer_rejects_misaligned_intrinsics() -> None:
    c2ws, intrinsics = _camera_inputs()

    with pytest.raises(ValueError, match="intrinsics must be"):
        render_infinite_grid_video(
            c2ws,
            intrinsics[:1],
            height=16,
            width=24,
            device="cpu",
            use_compile=False,
            ssaa=1,
        )


@pytest.mark.gpu(gpus=1)
def test_grid_renderer_cuda_shape_range_and_motion() -> None:
    c2ws, intrinsics = _camera_inputs()

    rendered = render_infinite_grid_video(
        c2ws,
        intrinsics,
        height=16,
        width=24,
        device="cuda",
        use_compile=False,
        ssaa=1,
    )

    assert rendered.shape == (2, 3, 16, 24)
    assert rendered.dtype == torch.float32
    assert rendered.device.type == "cuda"
    assert torch.isfinite(rendered).all()
    assert rendered.min() >= 0.0
    assert rendered.max() <= 1.0
    assert not torch.equal(rendered[0], rendered[1])
