"""Original fixed-size video clips behind Native's resumable DP loader.

No latent cache or implicit replacement of corrupt/short samples: an invalid
record fails with its sample ID. Worker RNG and data cursors belong to Native.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from vforge.data.dataset import load_manifest_records, manifest_file_digest


def frame_indices(length, frames, interval, *, random_clip):
    required = (frames - 1) * interval + 1
    if length < required:
        raise ValueError(f"Video has {length} frames; this clip requires {required}")
    fraction = float(torch.rand(())) if random_clip else 0.0
    start = round(fraction * (length - required))
    return [start + index * interval for index in range(frames)]


def open_video(path):
    import decord

    # Preserve source resize semantics: 256x256, no crop/letterbox/FPS resample.
    return decord.VideoReader(str(path), ctx=decord.cpu(0), width=256, height=256, num_threads=1)


def read_video_clip(path, frames, interval, *, random_clip, exact_length=False, start_frame=None):
    """Decode one selected clip; cache pairing can require the audited fixed window."""
    reader = open_video(path)
    if exact_length and len(reader) != frames:
        raise ValueError("Cached RGB source length changed; refusing a different time window")
    if start_frame is None:
        indices = frame_indices(len(reader), frames, interval, random_clip=random_clip)
    else:
        if random_clip or type(start_frame) is not int or start_frame < 0:
            raise ValueError("A fixed clip window needs a nonnegative start and random_clip=False")
        indices = [start_frame + index * interval for index in range(frames)]
        if indices[-1] >= len(reader):
            raise ValueError("Fixed clip window extends beyond the source video")
    pixels = torch.from_numpy(reader.get_batch(indices).asnumpy())
    if pixels.shape != (frames, 256, 256, 3):
        raise ValueError(f"Unexpected decoded pixel shape {tuple(pixels.shape)}")
    return (pixels.float() / 255.0).mul(2).sub(1).permute(3, 0, 1, 2).contiguous()


class VideoClipDataset(Dataset):
    def __init__(self, manifest, *, frames, frame_interval=1, random_clip=True):
        if (
            type(frames) is not int
            or frames < 1
            or type(frame_interval) is not int
            or frame_interval < 1
        ):
            raise ValueError("frames and frame_interval must be positive integers")
        self.frames, self.frame_interval, self.random_clip = frames, frame_interval, random_clip
        manifest = Path(manifest).resolve()
        self.records = []
        for index, record in enumerate(load_manifest_records(manifest)):
            sample_id = str(record.get("sample_id", record.get("id", record.get("clip_id", index))))
            value = (
                record.get("rgb_path")
                or record.get("video")
                or record.get("video_path")
                or record.get("rgb")
            )
            if not isinstance(value, str) or not value:
                raise ValueError(f"Sample {sample_id}: requires rgb_path or video")
            path = Path(value)
            if not path.is_absolute():
                path = manifest.parent / path
            start = record.get("start_frame")
            end = record.get("end_frame_exclusive")
            if start is not None or end is not None:
                required = (frames - 1) * frame_interval + 1
                if (
                    type(start) is not int
                    or start < 0
                    or type(end) is not int
                    or end - start < required
                    or random_clip
                ):
                    raise ValueError(f"Sample {sample_id}: invalid fixed clip window")
            # Manifest-relative paths are already absolute here. Resolving each
            # video walks the shared filesystem for every sample on every rank;
            # defer file access (including symlink traversal) until __getitem__.
            self.records.append(dict(sample_id=sample_id, path=str(path), start_frame=start))
        if not self.records:
            raise ValueError("RepWorld video manifest is empty")
        if len({record["sample_id"] for record in self.records}) != len(self.records):
            raise ValueError("RepWorld manifest contains duplicate sample IDs")
        self.resume_contract = dict(
            dataset="rep_world.video.v1",
            manifest=str(manifest),
            manifest_digest=manifest_file_digest(manifest),
            frames=frames,
            frame_interval=frame_interval,
            random_clip=random_clip,
            resize=[256, 256],
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            rgb = read_video_clip(
                record["path"],
                self.frames,
                self.frame_interval,
                random_clip=self.random_clip,
                start_frame=record["start_frame"],
            )
        except Exception as exc:
            raise RuntimeError(f"Sample {record['sample_id']} ({record['path']}): {exc}") from exc
        return dict(sample_id=record["sample_id"], rgb=rgb)
