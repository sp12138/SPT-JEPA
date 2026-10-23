"""Shared substrate for the taichu raw-video datasets.

These datasets differ from the rest of ``vforge/data`` in one structural way:
they decode inside ``__getitem__`` rather than handing paths to the collate.
That is not a stylistic choice. A gameplay index entry names a frame window
inside a long chunk -- 149 frames out of 1797 for the 60s corpus -- and the
per-sample work that follows (camera intrinsics adjusted for the crop, one
rotation matrix drawn per clip, foreground/background normal split from a mask)
all depends on the decoded geometry. Deferring it to the collate would mean
passing the window plan along and rebuilding the same state there.

The base class, the geometry and timing helpers, prompt cleaning, and reference
frame selection all live in this one module: their only callers are the three
dataset classes in this package, which is not enough to justify splitting them
apart.
"""

from __future__ import annotations

import html
import json
import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

_WHITESPACE = re.compile(r"\s+")


def prompt_clean(text: str) -> str:
    """Repair mojibake, unescape HTML twice, and collapse whitespace.

    The double unescape is deliberate and matches taichu: these captions went
    through a pipeline that escaped them twice, so a single pass leaves
    ``&amp;quot;`` behind. Cleaning happens here rather than in the collate
    because the generic ``RawSampleDataset`` path does not clean, and quietly
    changing its tokenization would desynchronize old and new cache bundles
    without tripping the encoding fingerprint.
    """
    import ftfy

    fixed = ftfy.fix_text(text)
    fixed = html.unescape(html.unescape(fixed))
    return _WHITESPACE.sub(" ", fixed).strip()


def select_ref_frame_indices(
    count: int,
    picks: int,
    *,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Pick ``picks`` indices from ``[0, count)``: stratified, random within strata.

    The range splits into ``picks`` contiguous float-width segments and one
    index is drawn inside each. Because segment ``j + 1`` starts exactly where
    ``j`` ends, the result is strictly increasing without a dedup pass, and
    ``count / picks`` need not divide evenly.
    """
    if count < 1:
        raise ValueError(f"select_ref_frame_indices needs count >= 1, got {count}")
    picks = max(1, min(int(picks), int(count)))
    # torch.randint wants its device to match the generator's, so an inference
    # caller passing a CUDA generator would otherwise hit a device mismatch.
    device = generator.device if generator is not None else None
    width = count / picks
    indices: list[int] = []
    for step in range(picks):
        low = int(step * width)
        high = max(int((step + 1) * width), low + 1)
        offset = int(torch.randint(0, high - low, (1,), generator=generator, device=device).item())
        indices.append(low + offset)
    return indices


def resize_center_crop(frames: torch.Tensor, target_height: int, target_width: int):
    """Scale ``[T, C, H, W]`` to cover the target, then center crop to it.

    The scale is ``max`` of the two ratios, not ``min``: covering the target and
    cropping the overflow preserves the aspect ratio without letterboxing. A
    plain resize would distort, and the camera intrinsics adjustment downstream
    assumes exactly this cover-then-crop geometry.
    """
    _, _, height, width = frames.shape
    scale = max(target_height / height, target_width / width)
    scaled_height = max(int(round(height * scale)), target_height)
    scaled_width = max(int(round(width * scale)), target_width)
    resized = torch.nn.functional.interpolate(
        frames.float(),
        size=(scaled_height, scaled_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (scaled_height - target_height) // 2
    left = (scaled_width - target_width) // 2
    # contiguous() drops the reference to the full scaled tensor so it can be
    # collected; the crop is a view otherwise.
    return resized[:, :, top : top + target_height, left : left + target_width].contiguous()


def resize_exact(frames: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """Resize ``[T, C, H, W]`` straight to the target, aspect ratio be damned.

    This is what ``torchvision.transforms.Resize((h, w))`` does, which is what
    the self-forcing corpora were built with. Gameplay uses
    :func:`resize_center_crop` instead; the two are not interchangeable, since
    only the cropping variant keeps camera intrinsics meaningful.
    """
    return torch.nn.functional.interpolate(
        frames.float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def unit_normalize_normal(frames: torch.Tensor) -> torch.Tensor:
    """Renormalize an RGB-encoded normal clip in place, staying in ``[0, 1]``.

    Interpolation blends neighboring normals, which shortens the vectors, so a
    resized normal map is no longer unit length. Every step is in-place because
    at 149x3x704x1280 each intermediate copy is ~1.6 GB.
    """
    frames.mul_(2.0).sub_(1.0)
    norm = torch.linalg.norm(frames, dim=1, keepdim=True).clamp_(min=1e-6)
    frames.div_(norm)
    del norm
    return frames.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)


def align_4np1(length: int) -> int:
    """Round ``length`` down to the nearest ``4n + 1``, or 0 if too short.

    The causal Wan VAE consumes frames in groups of four plus a leading frame,
    so any other count would be silently truncated inside the encoder.
    """
    if length < 5:
        return 0
    return ((length - 1) // 4) * 4 + 1


def pad_or_trim(data: torch.Tensor, length: int) -> torch.Tensor:
    """Force dim 0 to ``length``, repeating the last entry when short."""
    have = data.shape[0]
    if have == length:
        return data
    if have > length:
        return data[:length]
    padding = data[-1:].expand(length - have, *data.shape[1:])
    return torch.cat([data, padding], dim=0)


def load_index_records(path: str | Path) -> list[Any]:
    """Read one index file as a list of records (``.pkl`` or ``.json``).

    Taichu writes gameplay indices as pickles (they carry numpy camera arrays)
    and the self-forcing indices as JSON. Both are operator-supplied local
    files from the same trust domain as the checkpoints.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"index file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".pkl":
        with file_path.open("rb") as handle:
            records = pickle.load(handle)
    elif suffix in {".json", ".jsonl"}:
        with file_path.open(encoding="utf-8") as handle:
            text = handle.read()
        records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if suffix == ".jsonl"
            else json.loads(text)
        )
    else:
        raise ValueError(
            f"unsupported index format {suffix!r} for {file_path}; expected .pkl or .json"
        )
    if not isinstance(records, list):
        raise ValueError(f"{file_path} must hold a list of records, got {type(records).__name__}")
    return records


def remap_path(
    path: str | Path,
    replacements: Sequence[Sequence[str]] = (),
) -> str:
    """Resolve one logical Taichu path through deployment mount aliases."""
    value = str(path)
    for replacement in replacements:
        if len(replacement) != 2:
            raise ValueError("path_replacements entries must contain source and target")
        source, target = (str(part) for part in replacement)
        source_root = source.rstrip("/")
        if value == source_root or value.startswith(source_root + "/"):
            return target.rstrip("/") + value[len(source_root) :]
    return value


class RawVideoDataset(Dataset):
    """Index-backed video dataset that decodes and preprocesses per sample.

    Subclasses implement :meth:`parse_record` (index entry to a typed record,
    or ``None`` to drop it) and :meth:`build_sample` (record to the sample dict
    the Wan collate expects).

    Reading prepared latents is deliberately absent: that is what
    ``EncodedCacheDataset`` does, reached by leaving ``kind`` empty on the data
    source. Two implementations of the same read path is how they drift.
    """

    def __init__(
        self,
        *,
        index_files: Sequence[str],
        frames: int,
        height: int,
        width: int,
        max_sequence_length: int = 512,
        name: str = "",
        mode: str = "train",
        extras: dict[str, Any] | None = None,
    ) -> None:
        if mode not in {"train", "val"}:
            raise ValueError(f"unsupported mode={mode!r}; expected 'train' or 'val'")
        if not index_files:
            raise ValueError(f"{type(self).__name__} requires at least one index file")
        for label, value in (("frames", frames), ("height", height), ("width", width)):
            if int(value) < 1:
                raise ValueError(f"{type(self).__name__} needs {label} >= 1, got {value}")

        self.mode = mode
        self.name = name
        self.frames = int(frames)
        self.height = int(height)
        self.width = int(width)
        self.max_sequence_length = int(max_sequence_length)
        self.extras = dict(extras or {})
        path_replacements = self.extras.get("path_replacements", ())

        # Taichu hangs reference-frame extraction off VideoDatasetBase, which
        # GameplayDataset bypasses by overriding the whole builder -- so a
        # gameplay source asking for reference frames there is silently
        # ignored. Hoisting it to the base makes the knob mean the same thing
        # everywhere, and the encoder already reads the key from any source.
        self.extract_reference_frame = bool(self.extras.get("extract_reference_frame", False))
        self.ref_num_frames = int(self.extras.get("ref_num_frames", 1))

        raw_entries: list[Any] = []
        for index_file in index_files:
            raw_entries.extend(load_index_records(remap_path(index_file, path_replacements)))
        records = self.build_records(raw_entries)

        source = name or "default"
        for record in records:
            record.source = source

        max_num = self.extras.get("max_num")
        if max_num is not None and int(max_num) > 0:
            records = records[: int(max_num)]
        if not records:
            raise ValueError(
                f"{type(self).__name__} loaded 0 usable records from {list(index_files)}; "
                "every index entry was dropped as unusable"
            )
        self.records = records

    def build_records(self, raw_entries: list[Any]) -> list[Any]:
        """Turn every index entry into a record, dropping the unusable ones.

        The override hook exists for datasets whose records are not one-to-one
        with index entries -- the long-video set groups entries by uuid.
        """
        parsed = (self.parse_record(raw) for raw in raw_entries)
        return [record for record in parsed if record is not None]

    def parse_record(self, raw: Any) -> Any | None:
        raise NotImplementedError

    def build_sample(self, record: Any) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def sample_ids(self) -> list[str]:
        return [str(record.id) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = self.build_sample(record)
        sample.setdefault("sample_id", str(record.id))
        if self.extract_reference_frame and "reference_frames" not in sample:
            frames = sample["video_frames"]
            picks = select_ref_frame_indices(int(frames.shape[0]), self.ref_num_frames)
            sample["reference_frames"] = frames[picks].contiguous()
        # Stamped so the collate can tokenize to this source's length even in a
        # mixed run, where one closure cannot speak for every sub-dataset.
        sample.setdefault("max_sequence_length", self.max_sequence_length)
        return sample


__all__ = [
    "RawVideoDataset",
    "align_4np1",
    "load_index_records",
    "pad_or_trim",
    "prompt_clean",
    "remap_path",
    "resize_center_crop",
    "resize_exact",
    "select_ref_frame_indices",
    "unit_normalize_normal",
]
