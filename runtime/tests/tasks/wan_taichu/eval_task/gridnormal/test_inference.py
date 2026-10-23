from __future__ import annotations

from typing import Any

import torch

from vforge.models.wan_taichu.inference import WAN_TAICHU_GENERATOR_STATE_FIELD
from vforge.tasks.wan_taichu.eval_task.gridnormal import (
    GRIDNORMAL_VARIANTS,
    GridNormalInferencePipeline,
    resolve_gridnormal_variants,
)
from vforge.tasks.wan_taichu.eval_task.reference import REF_SELECTION_FIELD


def _tensor(value: float) -> torch.Tensor:
    return torch.full((1, 1, 1, 1, 1), value)


def _encoded(*, with_ref: bool) -> dict[str, Any]:
    encoded = {
        "condition_latents_grid": _tensor(1),
        "condition_latents_normal": _tensor(2),
        "condition_fallback_latents_grid": _tensor(3),
        "condition_fallback_latents_normal": _tensor(4),
        "normal_fg_body": _tensor(5),
        "normal_bg_body": _tensor(6),
    }
    if with_ref:
        encoded["ref_frame_latents"] = _tensor(7)
    return encoded


class _RecordingInference:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, Any]]] = []
        self.generator_states: list[torch.Tensor | None] = []

    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        self.calls.append((seed, sample["__encoded__"]))
        self.generator_states.append(sample.get(WAN_TAICHU_GENERATOR_STATE_FIELD))
        return {
            "video": torch.zeros(1, 3, 2, 2),
            "metadata": {"base": True},
        }


def test_pipeline_uses_preserved_seed_offset_and_reuses_seed_across_variants() -> None:
    base = _RecordingInference()
    pipeline = GridNormalInferencePipeline(
        base=base,
        variants=resolve_gridnormal_variants({}),
    )
    sample = {
        "sample_id": "sample",
        "__vforge_eval_index__": 9,
        "__vforge_eval_seed_offset__": 114,
        "__encoded__": _encoded(with_ref=False),
        "video": "/data/gt.mp4",
        "grid": "/data/grid.mp4",
        "normal": "/data/normal.mp4",
    }
    resolved_seed = pipeline.resolve_sample_seed(sample, 100)

    result = pipeline.infer_sample(
        sample,
        seed=resolved_seed,
    )

    assert resolved_seed == 214
    assert [seed for seed, _bundle in base.calls] == [214] * 5
    assert list(result["variants"]) == [
        "normal_to_video",
        "normal_fg_to_video",
        "normal_bg_to_video",
        "grid_to_video",
        "ti2v",
    ]
    assert result["shared_sources"] == {
        "gt": "/data/gt.mp4",
        "normal": "/data/normal.mp4",
        "grid": "/data/grid.mp4",
    }
    normal_only = result["variants"]["normal_to_video"]["metadata"]["gridnormal_condition"]
    assert normal_only["grid"]["fallback_pixel_value"] == 0.0
    assert normal_only["normal"]["input"] == "/data/normal.mp4"


def test_online_ref_k_zero_runs_ref_variants_without_fabricating_ref_latents() -> None:
    base = _RecordingInference()
    pipeline = GridNormalInferencePipeline(
        base=base,
        variants=resolve_gridnormal_variants({"variant": "wan22_ti2v_5b_ref"}),
    )

    result = pipeline.infer_sample(
        {
            "sample_id": "sample",
            "__vforge_eval_index__": 0,
            "__encoded__": _encoded(with_ref=False),
            REF_SELECTION_FIELD: {
                "source": "target_video",
                "indices": [],
                "count": 0,
            },
        },
        seed=5,
    )

    expected = GRIDNORMAL_VARIANTS[:5]
    assert list(result["variants"]) == [variant.name for variant in expected]
    assert len(base.calls) == 5
    assert all("ref_frame_latents" not in encoded for _seed, encoded in base.calls)
    assert result["metadata"]["ref_selection"]["count"] == 0


def test_only_ref_variants_inherit_online_ref_generator_state() -> None:
    base = _RecordingInference()
    pipeline = GridNormalInferencePipeline(
        base=base,
        variants=resolve_gridnormal_variants({"variant": "wan22_ti2v_5b_ref"}),
    )
    state = torch.Generator().manual_seed(42).get_state()

    pipeline.infer_sample(
        {
            "sample_id": "sample",
            "__vforge_eval_index__": 0,
            "__encoded__": _encoded(with_ref=True),
            WAN_TAICHU_GENERATOR_STATE_FIELD: state,
        },
        seed=42,
    )

    assert [generator_state is not None for generator_state in base.generator_states] == [
        variant.ref for variant in GRIDNORMAL_VARIANTS[:5]
    ]
