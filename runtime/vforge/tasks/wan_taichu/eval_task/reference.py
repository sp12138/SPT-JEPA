"""Reference-frame preparation for Wan Taichu evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vforge.models.wan_taichu.inference import WAN_TAICHU_GENERATOR_STATE_FIELD
from vforge.tasks.wan_taichu.variants import WanTaichuVariant

REF_SELECTION_FIELD = "__wan_taichu_ref_selection__"


def _prepare_reference_batch(
    batch: dict[str, Any],
    sample: Mapping[str, Any],
    *,
    base_seed: int,
    spec: WanTaichuVariant,
    extras: Mapping[str, Any],
    device: torch.device,
    num_frames_override: int | None = None,
    frame_indices_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Attach explicit or sampled references and return their provenance."""
    if not spec.ref_token:
        return {}
    explicit = batch.get("reference_condition")
    if isinstance(explicit, torch.Tensor):
        return {
            REF_SELECTION_FIELD: {
                "source": "explicit",
                "indices": None,
                "count": int(explicit.shape[1]),
            }
        }

    cap = int(extras.get("ref_num_frames_online", 0))
    has_override = num_frames_override is not None or frame_indices_override is not None
    if cap <= 0 and not has_override:
        return {}
    max_frames = extras.get("max_frames")
    video_source = sample.get("video_frames")
    if isinstance(video_source, torch.Tensor):
        if video_source.ndim != 4:
            raise ValueError(
                f"online reference video_frames must be [T,C,H,W], got {tuple(video_source.shape)}"
            )
        video = video_source
        if max_frames is not None:
            video = video[: int(max_frames)]
        source_description = "decoded target video"
    else:
        video_path = sample.get("video")
        if not video_path:
            raise ValueError(
                "online reference sampling requires validation input with a target video "
                "or an explicit ref/reference"
            )
        from vforge.data.video_io import read_video_tensor

        video = read_video_tensor(
            str(video_path),
            max_frames=None if max_frames is None else int(max_frames),
        )
        source_description = str(video_path)
    frame_count = int(video.shape[0])
    if frame_count < 1:
        raise ValueError(f"online reference source has no frames: {source_description}")
    seed_offset = int(
        sample.get(
            "__vforge_eval_seed_offset__",
            sample.get("__vforge_eval_index__", 0),
        )
    )
    seed = int(base_seed) + seed_offset
    generator = torch.Generator(device=device).manual_seed(seed)

    if frame_indices_override is not None:
        indices = [int(index) for index in frame_indices_override]
        invalid = [index for index in indices if index >= frame_count]
        if invalid:
            raise ValueError(
                "evaluation.online_ref.frame_indices exceed the target video: "
                f"indices={invalid} frames={frame_count}"
            )
        selection_mode = "frame_indices"
    else:
        if num_frames_override is not None:
            count = int(num_frames_override)
            if count > frame_count:
                raise ValueError(
                    "evaluation.online_ref.num_frames exceeds the target video: "
                    f"requested={count} frames={frame_count}"
                )
            selection_mode = "num_frames"
        else:
            maximum = min(frame_count // 80, cap)
            count = (
                0
                if maximum <= 0
                else int(
                    torch.randint(
                        0,
                        maximum + 1,
                        (1,),
                        generator=generator,
                        device=device,
                    ).item()
                )
            )
            selection_mode = "random_up_to_cap"
        indices = (
            _select_stratified_indices(frame_count, count, generator, device=device)
            if count
            else []
        )
    if indices:
        batch["reference_condition"] = video[indices].unsqueeze(0).contiguous()
    return {
        WAN_TAICHU_GENERATOR_STATE_FIELD: generator.get_state(),
        REF_SELECTION_FIELD: {
            "source": "target_video",
            "indices": indices,
            "count": len(indices),
            "seed": seed,
            "seed_offset": seed_offset,
            "cap": cap,
            "mode": selection_mode,
        },
    }


def _select_stratified_indices(
    frame_count: int,
    count: int,
    generator: torch.Generator,
    *,
    device: torch.device,
) -> list[int]:
    segment = frame_count / count
    indices = []
    for slot in range(count):
        lower = int(slot * segment)
        upper = max(int((slot + 1) * segment), lower + 1)
        offset = int(
            torch.randint(
                0,
                upper - lower,
                (1,),
                generator=generator,
                device=device,
            ).item()
        )
        indices.append(lower + offset)
    return indices


__all__ = ["REF_SELECTION_FIELD", "_prepare_reference_batch", "_select_stratified_indices"]
