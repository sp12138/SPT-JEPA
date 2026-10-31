"""Contracts for past-only distant anchors on benchmark windows (round-2 spec, Sections 3.4 and 11).

5 tests. CPU only. Run from the repository root:
    PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_benchmark_gap.py -v
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from as_jepa_tools.benchmark_data import BenchmarkWindows
from as_jepa_tools.benchmark_gap import GapBenchmarkWindows, make_benchmark_dataset, staleness_table

N = 60          # frames per episode
STRIDE = 5      # frameskip


def write_fixture(root):
    """Two training episodes and one test episode. Every pixel of row r has value r, so the row a
    frame came from can be read back from any pixel. The agent moves one unit per raw frame in
    episode 10 and stands still in episode 20; the yaw turns 1 degree per raw frame."""
    raw = root / "data.h5"
    rows = 3 * N
    with h5py.File(raw, "w") as f:
        f["pixels"] = np.broadcast_to(np.arange(rows, dtype=np.uint8)[:, None, None, None], (rows, 16, 16, 3))
        f["action"] = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        position = np.zeros((rows, 2))
        position[:N, 0] = np.arange(N)
        f["pos_agent"] = position
        f["yaw"] = np.radians(np.arange(rows, dtype=np.float64))[:, None]
    for k, v in dict(offsets=[0, N, 2 * N], lengths=[N, N, N], episode_ids=[10, 20, 30],
                     train_episodes=[10, 20], test_episodes=[30]).items():
        np.save(root / (k + ".npy"), v)
    (root / "dataset_card.json").write_text(json.dumps(dict(h5_path=str(raw), frameskip=STRIDE,
                                                            action_mean=[0, 0], action_std=[1, 1])))


class GapContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_gap_zero_reproduces_benchmark_windows(self):
        base, gap0 = BenchmarkWindows(self.root, window=4), GapBenchmarkWindows(self.root, window=4, gap=0)
        self.assertEqual(len(base), len(gap0))
        for i in range(len(base)):
            x, y = base[i], gap0[i]
            self.assertEqual(set(x), set(y))
            for k in x:
                if torch.is_tensor(x[k]):
                    torch.testing.assert_close(x[k], y[k], atol=0, rtol=0)
                else:
                    self.assertEqual(x[k], y[k])

    def test_anchor_is_past_phase_aligned_and_read_like_window_frames(self):
        gap, w = 2, 4
        ds, base = GapBenchmarkWindows(self.root, window=w, gap=gap), BenchmarkWindows(self.root, window=w)
        seen_eligible = seen_fallback = 0
        for i in range(len(ds)):
            x, b = ds[i], base[i]
            for k in ("pixels", "render", "action", "rows"):
                torch.testing.assert_close(x[k], b[k], atol=0, rtol=0)      # only the anchor changes
            ep, start = ds.locate(i)
            offset, row = int(ds.offsets[ep]), x["anchor_row"]
            self.assertTrue(offset <= row <= int(x["rows"][0]))               # same episode, never the future
            self.assertEqual((row - offset) % STRIDE, start % STRIDE)        # phase-aligned with the window
            if x["anchor_fallback"]:
                seen_fallback += 1
                self.assertLess(start, gap * STRIDE)
                self.assertEqual(row - offset, start % STRIDE)
            else:
                seen_eligible += 1
                self.assertLessEqual(row, int(x["rows"][0]) - gap * STRIDE)
            self.assertTrue(bool((x["anchor224"] == row % 256).all()))     # the anchor's own pixels
            self.assertTrue(bool((x["anchor"] == row % 256).all()))
            self.assertEqual(tuple(x["anchor224"].shape), (3, 224, 224))
            self.assertEqual(tuple(x["anchor"].shape), (3, 64, 64))
        self.assertGreater(seen_eligible, 0)
        self.assertGreater(seen_fallback, 0)

    def test_anchor_draw_is_deterministic_and_changes_across_epochs(self):
        ds = GapBenchmarkWindows(self.root, window=4, gap=1)
        late = [i for i in range(len(ds)) if ds.locate(i)[1] >= 6 * STRIDE]
        self.assertEqual([ds[i]["anchor_row"] for i in late], [ds[i]["anchor_row"] for i in late])
        moved = sum(ds[i]["anchor_row"] != ds[i + ds.natural_length]["anchor_row"] for i in late)
        self.assertGreater(moved, 0)

    def test_dataset_dispatch_follows_the_config(self):
        cfg = dict(data=str(self.root), seed=1, model=dict(window=4))
        self.assertIs(type(make_benchmark_dataset(cfg)), BenchmarkWindows)
        self.assertIs(type(make_benchmark_dataset(dict(cfg, gap=0))), BenchmarkWindows)
        made = make_benchmark_dataset(dict(cfg, gap=3), split="test")
        self.assertIsInstance(made, GapBenchmarkWindows)
        self.assertEqual(made.gap, 3)
        self.assertEqual(int(made.episode_ids[made.episodes[0]]), 30)

    def test_staleness_table_counts_moving_and_static_labels(self):
        labels = dict(agent=("pos_agent", "position", 4.0), yaw=("yaw", "yaw90", 12.0))
        table = staleness_table(self.root, labels, gaps=(0, 2), window=4, windows=10_000)
        windows = table["windows"]
        first = table["gaps"]["0"]
        self.assertEqual(first["fallback_share"], 0.0)
        # First-frame anchor: frames 1..3 are 5, 10, 15 raw frames after the anchor. The agent moves
        # 1 unit per raw frame in episode 10 only, so frames 1, 2 and 3 are stale there (5, 10, 15 > 4):
        # half the windows, every rendered frame.
        self.assertAlmostEqual(first["stale_share"]["agent"], 0.5)
        # Yaw turns 1 degree per raw frame in both episodes: frames 2 and 3 (10 and 15 degrees mod 90)
        # exceed 12 degrees only for frame 3.
        self.assertAlmostEqual(first["stale_share"]["yaw"], 1 / 3)
        second = table["gaps"]["2"]
        n_windows = N - 3 * STRIDE
        self.assertAlmostEqual(second["fallback_share"], (2 * STRIDE) / n_windows)
        self.assertAlmostEqual(GapBenchmarkWindows(self.root, window=4, gap=2).fallback_share(), (2 * STRIDE) / n_windows)
        self.assertEqual(GapBenchmarkWindows(self.root, window=4, gap=0).fallback_share(), 0.)
        self.assertEqual(windows, 2 * n_windows)
        self.assertGreaterEqual(second["stale_share"]["agent"], first["stale_share"]["agent"])


if __name__ == "__main__":
    unittest.main()
