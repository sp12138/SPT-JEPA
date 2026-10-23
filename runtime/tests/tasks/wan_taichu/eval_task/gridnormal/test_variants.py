from __future__ import annotations

from typing import Any

import pytest
import torch

from vforge.tasks.wan_taichu.eval_task.gridnormal.variants import (
    A14B_GRIDNORMAL_VARIANTS,
    GRIDNORMAL_VARIANTS,
    RUNNABLE_GRIDNORMAL_VARIANTS,
    TAICHU_GRIDNORMAL_VARIANTS,
    GridNormalCapabilityError,
    apply_gridnormal_variant,
    prepare_gridnormal_encoded,
    resolve_gridnormal_variants,
)


def _tensor(value: float) -> torch.Tensor:
    return torch.full((1, 1, 1, 1, 1), value, dtype=torch.float32)


def _bundle(*, ref: torch.Tensor | None = None, include_ref: bool = True) -> dict[str, Any]:
    encoded: dict[str, Any] = {
        "condition_latents_grid": _tensor(1),
        "condition_latents_normal": _tensor(2),
        "condition_fallback_latents_grid": _tensor(3),
        "condition_fallback_latents_normal": _tensor(4),
        "normal_fg_body": _tensor(5),
        "normal_bg_body": _tensor(6),
        "unrelated": "preserved",
    }
    if include_ref:
        encoded["ref_frame_latents"] = _tensor(7) if ref is None else ref
    return encoded


def test_gridnormal_resolver_enforces_ref_capability_gate() -> None:
    assert (
        tuple(variant.name for variant in resolve_gridnormal_variants({}))
        == RUNNABLE_GRIDNORMAL_VARIANTS
    )
    assert resolve_gridnormal_variants({"variant": "wan22_a14b"}) == A14B_GRIDNORMAL_VARIANTS
    assert [variant.name for variant in A14B_GRIDNORMAL_VARIANTS] == [
        "full",
        "normal_to_video",
        "normal_fg_to_video",
        "normal_bg_to_video",
        "grid_to_video",
    ]
    assert A14B_GRIDNORMAL_VARIANTS[0].ref is False
    with pytest.raises(GridNormalCapabilityError, match="wan22_ti2v_5b_ref"):
        resolve_gridnormal_variants({}, ("full",))
    assert (
        resolve_gridnormal_variants({"variant": "wan22_ti2v_5b_ref"}) == TAICHU_GRIDNORMAL_VARIANTS
    )
    assert (
        tuple(
            variant.name
            for variant in resolve_gridnormal_variants(
                {"variant": "wan22_ti2v_5b_ref"},
                RUNNABLE_GRIDNORMAL_VARIANTS,
            )
        )
        == RUNNABLE_GRIDNORMAL_VARIANTS
    )


@pytest.mark.parametrize(
    ("name", "grid", "normal", "keeps_ref"),
    [
        ("full", 1, 2, True),
        ("normal_to_video", 3, 2, False),
        ("normal_fg_to_video", 3, 5, False),
        ("normal_bg_to_video", 3, 6, False),
        ("grid_to_video", 1, 4, False),
        ("ref_normal_to_video", 3, 2, True),
        ("ref_grid_to_video", 1, 4, True),
        ("ti2v", 3, 4, False),
    ],
)
def test_apply_gridnormal_variant_covers_all_eight_condition_selections(
    name: str,
    grid: float,
    normal: float,
    keeps_ref: bool,
) -> None:
    encoded = _bundle()
    variant = next(item for item in GRIDNORMAL_VARIANTS if item.name == name)

    transformed = apply_gridnormal_variant(encoded, variant)

    assert transformed["condition_latents_grid"].item() == grid
    assert transformed["condition_latents_normal"].item() == normal
    assert ("ref_frame_latents" in transformed) is keeps_ref
    assert transformed["unrelated"] == "preserved"
    assert encoded["condition_latents_grid"].item() == 1
    assert encoded["condition_latents_normal"].item() == 2
    assert "ref_frame_latents" in encoded


def test_ref_variant_distinguishes_malformed_missing_ref_from_online_k_zero() -> None:
    ref_variant = GRIDNORMAL_VARIANTS[0]

    with pytest.raises(GridNormalCapabilityError, match="ref_frame_latents"):
        apply_gridnormal_variant(
            _bundle(include_ref=False),
            ref_variant,
        )

    online_k_zero = _bundle()
    online_k_zero["ref_frame_latents"] = None
    transformed = apply_gridnormal_variant(online_k_zero, ref_variant)
    assert "ref_frame_latents" not in transformed


@pytest.mark.parametrize("name", ["normal_fg_to_video", "normal_bg_to_video"])
def test_missing_partial_normal_uses_neutral_fallback(name: str) -> None:
    encoded = _bundle()
    encoded.pop("normal_fg_body")
    encoded.pop("normal_bg_body")
    variant = next(item for item in GRIDNORMAL_VARIANTS if item.name == name)

    transformed = apply_gridnormal_variant(encoded, variant)

    assert transformed["condition_latents_normal"].item() == 4


def test_partial_normal_preserves_a14b_mask_channels() -> None:
    encoded = _bundle()
    encoded["condition_latents_normal"] = torch.cat(
        [torch.full((1, 4, 1, 1, 1), 9.0), torch.full((1, 16, 1, 1, 1), 2.0)],
        dim=1,
    )
    encoded["normal_fg_body"] = torch.full((1, 16, 1, 1, 1), 5.0)
    variant = next(item for item in GRIDNORMAL_VARIANTS if item.name == "normal_fg_to_video")

    transformed = apply_gridnormal_variant(encoded, variant)

    normal = transformed["condition_latents_normal"]
    assert normal.shape[1] == 20
    assert torch.all(normal[:, :4] == 9)
    assert torch.all(normal[:, 4:] == 5)


class _FallbackEncoder:
    def __init__(self) -> None:
        self.pixel_values: list[float] = []

    def encode_constant_condition(
        self,
        *,
        pixel_value: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.pixel_values.append(pixel_value)
        return torch.full(
            (kwargs["batch_size"], 1, 1, 1, 1),
            pixel_value + 10,
            dtype=kwargs["dtype"],
            device=kwargs["device"],
        )


def test_prepare_gridnormal_encoded_uses_frozen_encoder_and_fixed_field_names() -> None:
    encoder = _FallbackEncoder()
    prepared = prepare_gridnormal_encoded(
        {
            "condition_latents_grid": _tensor(1),
            "condition_latents_normal": _tensor(2),
        },
        encoder,
        height=16,
        width=16,
        num_frames=5,
    )

    assert encoder.pixel_values == [0.0, 0.5]
    assert prepared["condition_fallback_latents_grid"].item() == 10
    assert prepared["condition_fallback_latents_normal"].item() == 10.5
