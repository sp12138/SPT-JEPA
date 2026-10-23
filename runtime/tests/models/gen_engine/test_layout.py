from __future__ import annotations

import torch

from vforge.models.gen_engine.codec import (
    PeriodicOffsetsTemporalMapping,
    StrideTemporalMapping,
)
from vforge.models.gen_engine.layout import (
    AUDIO_REFERENCE,
    IMAGE_REFERENCE,
    VIDEO_REFERENCE,
    AudioGeometry,
    ReferenceGeometry,
    TargetGeometry,
    VisualGeometry,
    build_packed_layout,
)


def _image() -> VisualGeometry:
    return VisualGeometry((1, 1, 2, 2), None, is_image=True)


def _video(frames: int) -> VisualGeometry:
    return VisualGeometry((1, frames, 2, 2), 1.0)


def _audio(frames: int, *, channels: int = 1, fps: float = 1.0) -> AudioGeometry:
    return AudioGeometry((channels, 2, frames), fps)


def test_golden_reference_timeline_and_modality_identities() -> None:
    layout = build_packed_layout(
        2,
        [
            ReferenceGeometry("image", visual=_image()),
            ReferenceGeometry("image", visual=_image()),
            ReferenceGeometry(
                "video_audio",
                visual=_video(3),
                audio=_audio(3),
            ),
            ReferenceGeometry("video", visual=_video(3)),
        ],
        TargetGeometry(
            visual=_video(2),
            audio=_audio(2, fps=2.0),
        ),
    )

    # Image 1 T=0, Image 2 T=1, paired AV T=2..4, Video 2 T=5..7,
    # and both target branches begin at T=8.
    video_times = layout.position_ids[layout.video_indices, 0]
    audio_times = layout.position_ids[layout.audio_indices, 0]
    torch.testing.assert_close(
        video_times,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float32),
    )
    torch.testing.assert_close(
        audio_times,
        torch.tensor([2, 3, 4, 8, 8.5], dtype=torch.float32),
    )
    assert layout.target_start == 8

    expected_video_modalities = torch.tensor(
        [
            IMAGE_REFERENCE,
            IMAGE_REFERENCE,
            VIDEO_REFERENCE,
            VIDEO_REFERENCE,
            VIDEO_REFERENCE,
            VIDEO_REFERENCE,
            VIDEO_REFERENCE,
            VIDEO_REFERENCE,
            0,
            0,
        ]
    )
    expected_video_indices = torch.tensor([1, 2, 1, 1, 1, 2, 2, 2, 0, 0])
    assert torch.equal(
        layout.reference_modality_ids[layout.video_indices],
        expected_video_modalities,
    )
    assert torch.equal(
        layout.reference_index_ids[layout.video_indices],
        expected_video_indices,
    )
    assert torch.equal(
        layout.reference_modality_ids[layout.audio_indices],
        torch.tensor([AUDIO_REFERENCE, AUDIO_REFERENCE, AUDIO_REFERENCE, 0, 0]),
    )
    assert torch.equal(
        layout.reference_index_ids[layout.audio_indices],
        torch.tensor([1, 1, 1, 0, 0]),
    )
    assert torch.equal(layout.reference_item_types, torch.tensor([0, 0, 1, 2]))
    assert torch.equal(layout.reference_item_image_indices, torch.tensor([1, 2, 0, 0]))
    assert torch.equal(layout.reference_item_video_indices, torch.tensor([0, 0, 1, 2]))
    assert torch.equal(layout.reference_item_audio_indices, torch.tensor([0, 0, 1, 0]))
    torch.testing.assert_close(
        layout.reference_item_time_ranges,
        torch.tensor([[0, 0], [1, 1], [2, 4], [5, 7]], dtype=torch.float32),
    )


def test_physical_order_text_positions_and_visual_grid_reset() -> None:
    visual = VisualGeometry((1, 1, 4, 4), None, is_image=True)
    target = VisualGeometry((1, 1, 4, 4), 1.0)
    layout = build_packed_layout(
        3,
        [
            ReferenceGeometry("image", visual=visual),
            ReferenceGeometry("image", visual=visual),
        ],
        TargetGeometry(visual=target),
    )

    assert torch.equal(layout.text_indices, torch.arange(3))
    assert int(layout.video_indices[0]) == 3
    assert layout.audio_indices.shape == (0,)
    assert torch.count_nonzero(layout.position_ids[layout.text_indices]) == 0
    expected_grid = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=torch.float32,
    )
    first = layout.position_ids[layout.video_indices[:4], 1:]
    second = layout.position_ids[layout.video_indices[4:8], 1:]
    target_grid = layout.position_ids[layout.video_indices[8:], 1:]
    assert torch.equal(first, expected_grid)
    assert torch.equal(second, expected_grid)
    assert torch.equal(target_grid, expected_grid)


def test_audio_channel_rows_repeat_local_time_and_av_shares_start() -> None:
    layout = build_packed_layout(
        0,
        [
            ReferenceGeometry(
                "video_audio",
                visual=VisualGeometry((1, 2, 2, 2), 2.0),
                audio=AudioGeometry((2, 4, 3), 4.0),
            )
        ],
        TargetGeometry(audio=AudioGeometry((2, 4, 2), 2.0)),
    )

    torch.testing.assert_close(
        layout.position_ids[layout.video_indices, 0],
        torch.tensor([0.0, 0.5]),
    )
    torch.testing.assert_close(
        layout.position_ids[layout.audio_indices, 0],
        torch.tensor([0.0, 0.25, 0.5, 0.0, 0.25, 0.5, 2.0, 2.5, 2.0, 2.5]),
    )
    assert layout.target_start == 2
    assert layout.num_reference_audio_rows == 6
    assert layout.num_target_audio_rows == 4
    assert torch.equal(
        layout.audio_plane_ids,
        torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2]),
    )


def test_video_only_and_audio_only_layouts_keep_zero_row_ranks() -> None:
    video_only = build_packed_layout(
        1,
        [],
        TargetGeometry(visual=VisualGeometry((2, 2, 4, 4), 1.0)),
    )
    assert video_only.audio_indices.shape == (0,)
    assert video_only.audio_loss_mask.shape == (0,)
    assert video_only.position_ids.shape == (9, 3)

    audio_only = build_packed_layout(
        1,
        [],
        TargetGeometry(audio=AudioGeometry((2, 3, 2), 1.0)),
    )
    assert audio_only.video_indices.shape == (0,)
    assert audio_only.video_loss_mask.shape == (0,)
    assert audio_only.position_ids.shape == (5, 3)


def test_h3_paired_target_uses_video_5_0_and_audio_5_15_second_endpoints() -> None:
    layout = build_packed_layout(
        0,
        [],
        TargetGeometry(
            visual=VisualGeometry((1, 37, 2, 2), 24.0),
            audio=AudioGeometry((2, 3, 207), 32_000.0),
        ),
        visual_temporal_mapping=PeriodicOffsetsTemporalMapping(
            input_period=17,
            latent_offsets=(0, 1, 5, 9, 13),
            valid_length_remainder=5,
        ),
        audio_temporal_mapping=StrideTemporalMapping(800),
    )

    video_times = layout.position_ids[layout.video_indices, 0]
    audio_times = layout.position_ids[layout.audio_indices, 0]
    torch.testing.assert_close(video_times[-1], torch.tensor(5.0))
    torch.testing.assert_close(audio_times[206], torch.tensor(5.15))
    assert layout.target_end == audio_times[206].item()
