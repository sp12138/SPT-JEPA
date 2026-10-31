"""Past-only distant anchors for the TwoRoom and Cube windows (round-2 spec, Section 3.4).

`GapBenchmarkWindows` is `BenchmarkWindows` with one change: the renderer's anchor is a
phase-aligned frame of the same episode at least `gap` kept frames *before* window frame 0,
instead of window frame 0 itself. Every other field of a window (pixels, render, actions,
rows, episode, start) is byte-identical, and `gap = 0` reproduces `BenchmarkWindows` exactly.

The anchor never comes from the future: a future frame would show where slow objects end up,
and generation only ever has past frames. When an episode has no frame far enough in the past,
the earliest phase-aligned frame before the window is used (or window frame 0 itself when the
window starts the episode), and `anchor_fallback` is True.

`staleness_table` is the CPU check that chooses the gap (spec Section 5.6). For each anchor rule
it reports the share of windows that fall back, and, for each labelled quantity (agent position,
cube position and yaw, effector position), the share of rendered frames in which the anchor shows
that quantity more than a tolerance away from its value in the frame. Physical labels are read
only by this diagnostic; they never reach a model.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from as_jepa_tools.benchmark_data import BenchmarkWindows


def _anchor_offset(start, gap, stride, key):
    """Row of the anchor within its episode, and whether the gap rule had to fall back.

    Window frame 0 sits at episode row `start`. Eligible anchors are the phase-aligned rows
    `start % stride, start % stride + stride, ...` that lie at least `gap` kept frames before it.
    One is drawn uniformly with a generator seeded by `key`, as `FullPushTWindows` does.
    """
    if gap == 0:
        return start, False
    latest = start - gap * stride
    if latest < 0:
        return start % stride, True                     # earliest past frame; window frame 0 if none
    eligible = np.arange(start % stride, latest + 1, stride)
    generator = torch.Generator().manual_seed(hash(key) & 0x7FFFFFFF)
    return int(eligible[int(torch.randint(len(eligible), (1,), generator=generator))]), False


class GapBenchmarkWindows(BenchmarkWindows):
    def __init__(self, root, window=8, seed=0, split="train", gap=0):
        super().__init__(root, window=window, seed=seed, split=split)
        if gap < 0:
            raise ValueError("gap must be non-negative")
        self.gap = int(gap)

    def _frame(self, row):
        """One stored frame through exactly the transforms `BenchmarkWindows` applies to window frames."""
        pixels = torch.from_numpy(np.asarray(self.file["pixels"][row:row + 1]).copy())
        if pixels.shape[-1] in (3, 4):
            pixels = pixels[..., :3].permute(0, 3, 1, 2)
        if pixels.shape[1] != 3:
            raise ValueError("Expected RGB pixels")
        if pixels.shape[-2:] != (224, 224):
            pixels = F.interpolate(pixels.float(), size=(224, 224), mode="bilinear", align_corners=False,
                                   antialias=True).round().clamp(0, 255).byte()
        render = F.interpolate(pixels.float(), size=(64, 64), mode="area").round().byte()
        return pixels[0], render[0]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.gap == 0:
            return item
        epoch = int(index) // self.natural_length
        ep, start = self.locate(index)
        offset = int(self.offsets[ep])
        row, fallback = _anchor_offset(start, self.gap, self.stride, (ep, start, epoch))
        item["anchor224"], item["anchor"] = self._frame(offset + row)
        item["anchor_row"], item["anchor_fallback"] = offset + row, fallback
        return item

    def fallback_share(self):
        """Exact share of windows without an eligible past anchor: those whose window starts fewer
        than `gap` kept frames into its episode. It depends only on the data and the gap."""
        if self.gap == 0:
            return 0.
        return float(np.minimum(self.counts, self.gap * self.stride).sum() / self.natural_length)


def make_benchmark_dataset(cfg, split="train"):
    """The dataset a benchmark run trains or is diagnosed on: the gap rule when the config sets `gap`."""
    window, seed, gap = cfg["model"]["window"], cfg["seed"], int(cfg.get("gap", 0))
    if gap:
        return GapBenchmarkWindows(cfg["data"], window=window, seed=seed, split=split, gap=gap)
    return BenchmarkWindows(cfg["data"], window=window, seed=seed, split=split)


def _distance(kind, a, b):
    """Distance between label values: Euclidean for positions; for `yaw90`, the yaw difference
    modulo the cube's quarter-turn symmetry, in degrees."""
    if kind == "position":
        return np.linalg.norm(a - b, axis=-1)
    d = np.abs(a - b)[..., 0] % (math.pi / 2)
    return np.degrees(np.minimum(d, math.pi / 2 - d))


def staleness_table(root, labels, gaps=(0, 1, 2, 4, 8, 16), window=8, split="train", windows=20000, seed=0):
    """Share of fallback windows and of stale rendered frames, per anchor rule.

    `labels` maps a name to (HDF5 key, kind, tolerance), with kind "position" or "yaw90".
    A rendered frame is window frame 1..W-1; it is stale for a label when the anchor's value
    differs from the frame's by more than the tolerance. `gap = 0` is the first-frame anchor.
    A constant (zero) anchor shows nothing, so it is stale for every label by definition.
    """
    ds = BenchmarkWindows(root, window=window, seed=seed, split=split)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(ds), size=min(windows, len(ds)), replace=False)
    located = [ds.locate(int(i)) for i in picks]
    ds.open()                                           # the loader's own handle, with hdf5plugin's filters
    values = {name: np.asarray(ds.file[key], dtype=np.float64) for name, (key, _, _) in labels.items()}
    ds.file.close()
    table = {}
    for gap in gaps:
        fallback = 0
        stale = {name: 0 for name in labels}
        total = 0
        for ep, start in located:
            offset = int(ds.offsets[ep])
            row, fell = _anchor_offset(start, gap, ds.stride, (ep, start, 0))
            fallback += fell
            frames = offset + start + np.arange(1, window) * ds.stride
            total += len(frames)
            for name, (_, kind, tol) in labels.items():
                v = values[name]
                stale[name] += int((_distance(kind, v[frames], v[offset + row][None]) > tol).sum())
        table[str(gap)] = dict(fallback_share=fallback / len(located),
                               stale_share={name: count / total for name, count in stale.items()})
    return dict(windows=len(located), split=split, window=window, stride=ds.stride,
                labels={name: dict(key=k, kind=kind, tolerance=t) for name, (k, kind, t) in labels.items()},
                gaps=table)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="benchmark metadata directory (dataset_card.json, offsets.npy, ...)")
    p.add_argument("--label", action="append", required=True,
                   help="name:hdf5_key:kind:tolerance, kind in {position, yaw90}; yaw90 tolerance in degrees")
    p.add_argument("--gaps", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--windows", type=int, default=20000)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    labels = {}
    for spec in a.label:
        name, key, kind, tol = spec.split(":")
        if kind not in {"position", "yaw90"}:
            raise ValueError(f"unknown label kind {kind!r}")
        labels[name] = (key, kind, float(tol))
    result = staleness_table(a.data, labels, gaps=tuple(a.gaps), window=a.window, windows=a.windows)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gaps"], indent=2))


if __name__ == "__main__":
    main()
