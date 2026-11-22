"""Contracts for the gap-bucket diagnostics (as_jepa_tools/gap_buckets.py).

20 tests. CPU only. Tiny models, except the last test: it runs the command line through the RoboCasa trainer's own
`build` and `Windows` on synthetic files, with the full-size model (about a minute on four CPU threads).
Run from the repository root:
    PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_gap_buckets.py -v
"""
import contextlib
import hashlib
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.models.as_jepa.flow import draw_timesteps
from vforge.models.as_jepa.pixels import pack
from as_jepa_tools.benchmark_data import BenchmarkWindows
from as_jepa_tools.benchmark_gap import GapBenchmarkWindows
from as_jepa_tools.data import FullPushTWindows, PushTWindows
from as_jepa_tools.gap_buckets import (KEYS, RobocasaWindows, add_probes, cluster_mean, collect, distortion_quantities,
                                       frame_flow_loss, frame_offsets, gap_buckets, load_labels, main, spearman, summarize,
                                       time_derangement, window_derangement, window_quantities)

N = 60          # frames per episode
STRIDE = 5      # frameskip
W = 4           # frames per window
NOISE = .1      # size of the planted renderer's gap-independent error
TINY = dict(mode="L2", beta=0.25, sigma0=0.1, window=W, encoder_depth=1, renderer_width=64,
            renderer_depth=1, renderer_heads=1, encoder_size=56)
EPISODES, FRAMES = 6, 17        # RoboCasa fixture: episodes, stored frames per episode (the trainer reads windows of 8)
REACH = dict(TINY, window=8, action_dim=14, rho=.8, forecast_seed=7, rate_prior="forecast")


def setUpModule():
    reset_ops()
    initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=torch.device("cpu"), parallel_state=None)


def write_benchmark(root, random_pixels=False):
    """Two training episodes (10, 20) and one validation episode (30) of N raw frames. Unless `random_pixels`,
    every pixel of stored row r has value r, so a frame shows the row it came from. Labels: `ramp` is linear in
    the row, `noise` is unrelated to it, `yaw` turns one degree per row."""
    raw, rows, rng = root / "data.h5", 3 * N, np.random.default_rng(0)
    with h5py.File(raw, "w") as f:
        f["pixels"] = (rng.integers(0, 255, (rows, 16, 16, 3), dtype=np.uint8) if random_pixels else
                       np.broadcast_to(np.arange(rows, dtype=np.uint8)[:, None, None, None], (rows, 16, 16, 3)))
        f["action"] = rng.normal(size=(rows, 2)).astype(np.float32)
        f["ramp"] = np.stack([2. * np.arange(rows) + 1, -np.arange(rows)], 1)
        f["noise"] = rng.normal(size=(rows, 2))
        f["yaw"] = np.radians(np.arange(rows, dtype=np.float64))[:, None]
    for k, v in dict(offsets=[0, N, 2 * N], lengths=[N, N, N], episode_ids=[10, 20, 30],
                     train_episodes=[10, 20], validation_episodes=[30]).items():
        np.save(root / (k + ".npy"), v)
    (root / "dataset_card.json").write_text(json.dumps(dict(h5_path=str(raw), frameskip=STRIDE,
                                                            action_mean=[0, 0], action_std=[1, 1])))


def write_pusht(root):
    """Three PushT episodes of N raw frames; episodes 1 and 2 are the test split. Encoder frames are stored at
    the tiny models' 56 pixels."""
    rows, rng = 3 * N, np.random.default_rng(0)
    np.save(root / "pixels224.npy", rng.integers(0, 255, (rows, 56, 56, 3), dtype=np.uint8))
    np.save(root / "pixels64.npy", rng.integers(0, 255, (rows, 64, 64, 3), dtype=np.uint8))
    np.save(root / "actions.npy", rng.normal(size=(rows, 10)).astype(np.float32))
    np.save(root / "states.npy", rng.normal(size=(rows, 5)).astype(np.float32))
    for k, v in dict(offsets=[0, N, 2 * N], lengths=[N, N, N], test_episodes=[1, 2], raw_rows=np.arange(rows)).items():
        np.save(root / (k + ".npy"), v)
    (root / "dataset_card.json").write_text(json.dumps(dict(frameskip=5)))


def write_robocasa(root):
    """Six Reach3D-style episodes in the files the RoboCasa trainer's `Windows` reads: per episode a feature array
    [17 frames, 2 views, 3 patches, 1664] and an archive with render [17, 2, 3, 64, 64] and commands [17, 7].
    Every render pixel of episode e, frame f, view v has value 17 e + f + 120 v, so a frame shows where it came
    from. Episodes 0..3 are the training split, 4 and 5 the validation split. Returns the records file."""
    rng, records = np.random.default_rng(0), []
    for e in range(EPISODES):
        files = root / f"episode-{e}.npy", root / f"episode-{e}.npz"
        np.save(files[0], rng.normal(size=(FRAMES, 2, 3, 1664)).astype(np.float32))
        value = 17 * e + np.arange(FRAMES)[:, None] + 120 * np.arange(2)[None]
        np.savez(files[1], render=np.broadcast_to(value[..., None, None, None], (FRAMES, 2, 3, 64, 64)).astype(np.uint8),
                 commands=rng.normal(size=(FRAMES, 7)).astype(np.float32))
        records.append(dict(id=f"episode-{e}", split="train" if e < 4 else "validation",
                            local_features=str(files[0]), local_observations=str(files[1])))
    (root / "data.json").write_text(json.dumps(dict(records=records)))
    return root / "data.json"


def randomize_zeros(model):
    """Zero-initialized layers (identity carry, AdaLN-zero, renderer output) hide dependencies at initialization,
    so they get small random values."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().sum() == 0:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))
    return model


def tiny_model(cls, **kw):
    """A tiny model in eval mode."""
    torch.manual_seed(0)
    return randomize_zeros(cls(**{**TINY, **kw})).eval()


def reach_model(objective="future-mean", joint=False, point=False, profile=None, forecast_fraction=.5):
    """A tiny model of the RoboCasa family, assembled as the trainer's `build` assembles the full one: the
    cached-feature model of `encoder_ablation.build` (frozen features in; an adapter and the pool make the state),
    then the same method patches. objective "flow" keeps `AnalyticResidualSPT.distortion`."""
    from experiments.robocasa_refine.analytic_model import AnalyticResidualSPT
    from experiments.robocasa_stable_scaling import train
    from experiments.robocasa_stable_scaling.pixel_likelihood import install_pixel_likelihood

    class Cached(AnalyticResidualSPT):                               # `CachedSPT` of encoder_ablation.build
        def __init__(self):
            super().__init__(**REACH, forecast_fraction=forecast_fraction)
            self.code_chart = "global"
            self.encoder = nn.Identity()
            self.feature_adapter = nn.Sequential(nn.Linear(1664, 192), nn.LayerNorm(192))

        def prepare(self, batch):
            self.cached = batch["features"]
            render = batch["render"].float() / 127.5 - 1.
            return render, render, render[:, 0]

        def encode(self, info):
            b, w = self.cached.shape[:2]
            tokens = self.feature_adapter(self.cached.float().flatten(0, 1))
            info["emb"] = self.pool(tokens).reshape(b, w, self.state_dim)
            if "action" in info:
                info["act_emb"] = self.action_encoder(info["action"])
            return info

        def loss(self, batch, step=0, ramp_steps=0):
            try:
                return super().loss(batch, step, ramp_steps)
            finally:
                self.cached = None

    torch.manual_seed(0)
    model = Cached()
    if objective != "flow":
        model.pixel_objective = objective
        model.distortion = types.MethodType(train.pixel_prediction_distortion, model)
    if joint:
        model.joint_views = True
        model.joint_view_embedding = nn.Parameter(torch.zeros(2, 192))
        model.encode = types.MethodType(train.joint_pixel_encode, model)
        model.original_pixel_distortion = model.distortion
        model.distortion = types.MethodType(train.joint_pixel_distortion, model)
        model.loss = types.MethodType(train.joint_pixel_loss, model)
    if profile:
        install_pixel_likelihood(model, profile)
    if point:
        model.state_mean_observation, model.identity_observation = True, True
    return randomize_zeros(model).eval()


def frame_anchor_distortion(self, batch, render, anchor, condition):
    """A planted `distortion`: the squared difference between the rendered frames and the anchor. It ignores the state."""
    return (render[:, 1:] - anchor[:, None]).square().mean()


def save_checkpoint(model, folder, target="as_jepa_lewm.model.ASJepaLeWM", **kw):
    folder.mkdir()
    torch.save(model.state_dict(), folder / "weights.pt")
    (folder / "config.json").write_text(json.dumps(dict(_target_=target, **TINY, **kw)))
    return folder


def run_main(*argv):
    with mock.patch.object(sys, "argv", ["gap_buckets", *map(str, argv), "--device", "cpu"]), \
            contextlib.redirect_stdout(None):
        main()


def fingerprint(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class PlantedRenderer:
    """Stands in for the renderer on the fixture whose pixels show their row. It reads the frame's pixel value
    from the state (with use_state=False it assumes the frame equals the anchor instead), which lets it undo
    the flow interpolation exactly. Its velocity is then off by NOISE * eps + slope * gap, where the gap is the
    distance it sees between that value and the anchor's: its loss grows with the gap exactly when slope > 0."""
    def __init__(self, slope, use_state):
        self.slope, self.use_state, self._anchor = slope, use_state, None

    @contextlib.contextmanager
    def anchored(self, anchor):
        self._anchor = anchor
        try:
            yield self
        finally:
            self._anchor = None

    def __call__(self, x, t, condition):
        tv = t.view(-1, 1, 1, 1, 1)
        shown = self._anchor.mean((1, 2, 3, 4), keepdim=True)
        value = shown
        if self.use_state:                                           # rounding to the 8-bit grid removes the posterior noise
            value = torch.round((condition.mean((1, 2)) + 1.) * 127.5).view(-1, 1, 1, 1, 1) / 127.5 - 1.
        eps = (x - (1 - tv) * value) / tv
        gap = (value - shown).abs() * 127.5 / STRIDE
        return value - eps - (NOISE * eps + self.slope * gap)


class PlantedModel:
    """The interface `collect` reads: every state dimension holds the frame's pixel value, the prior copies the
    last state, and the renderer is the planted one."""
    mode, sigma0, window, objective, training_shift, training = "L2", 1e-3, W, "flow", 1., False
    state_slots, state_width, state_dim = 4, 16, 64

    def __init__(self, slope=0., use_state=True):
        self.renderer = PlantedRenderer(slope, use_state)

    def prepare(self, batch):
        return batch["pixels"].float(), batch["render"].float() / 127.5 - 1., batch["anchor"].float() / 127.5 - 1.

    def encode(self, info):
        value = info["pixels"].mean((2, 3, 4)) / 127.5 - 1.
        return dict(info, emb=value[..., None].expand(-1, -1, self.state_dim), act_emb=info["action"])

    def predict(self, emb, act_emb):
        return emb


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bench, self.pusht, self.reach = self.root / "bench", self.root / "pusht", self.root / "reach"
        self.bench.mkdir(); self.pusht.mkdir(); self.reach.mkdir()
        write_benchmark(self.bench); write_pusht(self.pusht)
        self.records = write_robocasa(self.reach)

    def tearDown(self):
        self.tmp.cleanup()

    def offsets(self, data, indices):
        batch = default_collate([data[i] for i in indices])
        return frame_offsets(batch).tolist(), batch["anchor_fallback"].tolist()


class GapContracts(Fixtures):
    def test_first_frame_anchor_gap_is_the_frame_index(self):
        data = BenchmarkWindows(self.bench, window=W, split="validation")
        offsets, fallback = self.offsets(data, range(len(data)))
        self.assertEqual(offsets, [[0, 1, 2, 3]] * (N - (W - 1) * STRIDE))
        self.assertFalse(any(fallback))
        same, _ = self.offsets(GapBenchmarkWindows(self.bench, window=W, split="validation", gap=0), range(len(data)))
        self.assertEqual(same, offsets)

    def test_past_gap_anchor_by_hand(self):
        """gap = 2, stride 5. Window index = raw start within the one validation episode.
        start 10: the only phase-aligned row at least 10 raw rows back is row 0, so frame t is 2 + t kept frames away.
        start 12: likewise row 2. start 3: no past frame, the anchor is window frame 0 (fallback).
        start 7: the earliest past frame is row 2, one kept frame back (fallback)."""
        data = GapBenchmarkWindows(self.bench, window=W, split="validation", gap=2)
        offsets, fallback = self.offsets(data, [10, 12, 3, 7])
        self.assertEqual(offsets, [[2, 3, 4, 5], [2, 3, 4, 5], [0, 1, 2, 3], [1, 2, 3, 4]])
        self.assertEqual(fallback, [False, False, True, True])
        every, fell = self.offsets(data, range(len(data)))
        for row, f in zip(every, fell):                              # past only, at least `gap` back unless it fell back
            self.assertEqual([x - row[0] for x in row], [0, 1, 2, 3])
            self.assertTrue(0 <= row[0] < 2 if f else row[0] >= 2)

    def test_pusht_full_windows_by_hand(self):
        """gap = 8, raw stride 5, episodes of 60 raw rows. Window index = raw start within test episode 1.
        start 0: the only aligned row at least 8 kept frames after the last window frame (row 15) is row 55: a
          future anchor, 11 - t kept frames from frame t.
        start 44: the only aligned row at least 8 kept frames before row 44 is row 4: a past anchor, 8 + t away.
        start 20: no eligible row on either side; the rule falls back to the last aligned row, 55."""
        data = FullPushTWindows(self.pusht, window=W, gap=8, split="test")
        offsets, fallback = self.offsets(data, [0, 44, 20])
        self.assertEqual(offsets, [[-11, -10, -9, -8], [8, 9, 10, 11], [-7, -6, -5, -4]])
        self.assertEqual(fallback, [False, False, True])
        every, fell = self.offsets(data, range(len(data)))
        kept = [row for row, f in zip(every, fell) if not f]
        self.assertTrue(all(min(abs(x) for x in row) >= 8 for row in kept))
        self.assertTrue(any(row[0] > 0 for row in kept) and any(row[0] < 0 for row in kept))     # both sides occur

    def test_pusht_kept_frame_windows_and_misaligned_anchors(self):
        data = PushTWindows(self.pusht, window=W, gap=8, split="test")
        for i in range(0, len(data), 7):
            item = data[i]
            (row,), _ = self.offsets(data, [i])
            self.assertEqual(row, [int(r) - item["anchor_row"] for r in item["rows"]])    # stored rows are kept frames
            self.assertTrue(item["anchor_fallback"] or min(abs(x) for x in row) >= 8)
        with self.assertRaises(ValueError):
            frame_offsets(dict(rows=torch.tensor([[10, 15, 20]]), anchor_row=torch.tensor([3])))
        with self.assertRaises(ValueError):
            frame_offsets(dict(rows=torch.tensor([[10, 15, 25]]), anchor_row=torch.tensor([0])))


class SummaryContracts(unittest.TestCase):
    def test_buckets_are_exact_up_to_twelve_gaps_then_quantiles(self):
        gap = np.repeat(np.arange(12), 3)
        index, rule = gap_buckets(gap)
        self.assertEqual((rule, index.tolist()), ("exact", gap.tolist()))
        gap = np.random.default_rng(0).permutation(np.repeat(np.arange(8, 48), 5))      # 40 distinct gaps, 5 frames each
        index, rule = gap_buckets(gap)
        self.assertEqual((rule, sorted(set(index))), ("quantile", list(range(8))))
        self.assertEqual(np.bincount(index).tolist(), [25] * 8)
        bounds = [(gap[index == i].min(), gap[index == i].max()) for i in range(8)]
        self.assertTrue(all(lo <= hi for lo, hi in bounds) and all(a[1] < b[0] for a, b in zip(bounds, bounds[1:])))

    def test_cluster_standard_error_and_rank_correlations_by_hand(self):
        one = cluster_mean(np.array([1., 2., 3., 6., np.nan]), np.arange(5))            # one finite value per window
        self.assertEqual((one["frames"], one["mean"]), (4, 3.))
        self.assertAlmostEqual(one["sem"], np.std([1., 2., 3., 6.], ddof=1) / 2)
        two = cluster_mean(np.array([1., 1., 3., 3.]), np.array([0, 0, 1, 1]))          # two windows, each repeated twice:
        self.assertAlmostEqual(two["sem"], 1.)                                           # the error of two values, 1 and 3
        self.assertEqual(cluster_mean(np.array([np.nan]), np.array([0])), dict(frames=0, mean=None, sem=None))
        x, y = np.array([1., 2., 3., 4., np.nan]), np.array([1., 3., 2., 4., 9.])
        self.assertAlmostEqual(spearman(x, y), .8)                                       # 1 - 6 * 2 / (4 * 15)
        # ties share their mean rank: ranks (1.5, 1.5, 3, 4) and (1, 2, 3.5, 3.5), covariance 4, both sums of squares 4.5
        self.assertAlmostEqual(spearman(np.array([1., 1., 2., 3.]), np.array([1., 2., 3., 3.])), 4 / 4.5)
        self.assertIsNone(spearman(np.ones(5), np.arange(5.)))
        g = np.random.default_rng(0)
        z, e = g.normal(size=4000), g.normal(size=(2, 4000))
        self.assertGreater(spearman(z + .3 * e[0], z + .3 * e[1]), .8)                   # related only through z
        self.assertAlmostEqual(spearman(z + .3 * e[0], z + .3 * e[1], given=z), 0., delta=.05)
        self.assertIsNone(spearman(e[0], z, given=z))                                    # nothing of z is left beyond z


class PlantedContracts(Fixtures):
    def planted(self, **kw):
        data = GapBenchmarkWindows(self.bench, window=W, split="train", gap=2)          # gap and frame index vary separately
        records, states = collect(PlantedModel(**kw), data, range(len(data)), seed=0)
        return records, states, data

    def test_planted_gap_dependence_is_found_and_its_absence_too(self):
        records, _, _ = self.planted(slope=.3)
        gaps = records["gap"].reshape(-1, W)
        self.assertTrue((gaps - gaps[:, :1] == np.arange(W)).all())  # past-only anchors: one kept frame further per frame
        summary = summarize(records)
        self.assertEqual(summary["bucketing"], "exact")
        self.assertGreater(summary["spearman"]["render_loss"]["gap"], .95)
        means = [b["quantities"]["render_loss"]["mean"] for b in summary["buckets"]]
        self.assertIsNone(means[0])                                  # gap 0 holds window frames 0 only: not rendered
        self.assertGreaterEqual(len(means), 8)
        self.assertTrue(all(a < b for a, b in zip(means[1:], means[2:])))
        for b in summary["buckets"]:                                 # the loss the stub was built to have
            if b["gap_min"]:
                self.assertAlmostEqual(b["quantities"]["render_loss"]["mean"], NOISE ** 2 + (.3 * b["gap_min"]) ** 2, delta=.05)
        self.assertEqual(sum(b["frames"] for b in summary["buckets"]), len(records["gap"]))
        self.assertEqual(summary["overall"]["quantities"]["entry_nats"]["frames"], len(records["gap"]) // W)

        flat = summarize(self.planted(slope=0.)[0])
        self.assertLess(abs(flat["spearman"]["render_loss"]["gap"]), .2)
        self.assertLess(abs(flat["spearman"]["render_loss"]["gap_given_anchor_pixel_mse"] or 0.), .2)
        spread = [b["quantities"]["render_loss"]["mean"] for b in flat["buckets"][1:]]
        self.assertLess(max(spread) - min(spread), 1e-3)

    def test_state_swaps_matter_only_when_the_renderer_reads_the_state(self):
        reads, _, _ = self.planted()
        rendered = reads["frame"] > 0
        for name in ("state_reliance", "state_reliance_time"):
            self.assertTrue(np.isnan(reads[name][~rendered]).all())
            self.assertGreater(reads[name][rendered].min(), 0.)
        ignores, _, _ = self.planted(use_state=False)
        for name in ("render_loss_swapped", "render_loss_swapped_time"):
            self.assertEqual(ignores[name][rendered].tolist(), ignores["render_loss"][rendered].tolist())
        self.assertEqual(np.abs(ignores["state_reliance"][rendered]).max(), 0.)
        self.assertGreater(np.abs(ignores["anchor_reliance"][rendered]).min(), 0.)       # it copies the anchor instead

    def test_derangements_have_no_fixed_point_and_avoid_the_same_episode(self):
        self.assertIsNone(window_derangement(torch.tensor([3])))
        self.assertIsNone(time_derangement(1))
        for frames in range(2, 9):
            shift = time_derangement(frames)
            self.assertEqual(sorted(shift.tolist()), list(range(frames)))
            self.assertTrue(bool((shift != torch.arange(frames)).all()))
        self.assertEqual(window_derangement(torch.tensor([1, 2, 3, 4])).tolist(), [1, 2, 3, 0])
        self.assertEqual(window_derangement(torch.tensor([1, 1, 2, 2])).tolist(), [2, 3, 0, 1])     # not the roll by one
        records, _, _ = self.planted()
        rendered = records["frame"] > 0
        self.assertEqual(set(records["swap_same_episode"][rendered]), {0., 1.})         # batches inside one episode are flagged
        self.assertTrue(np.isnan(records["swap_same_episode"][~rendered]).all())

    def test_probes_recover_a_label_the_state_holds_and_not_an_unrelated_one(self):
        records, states, data = self.planted()
        labels = load_labels(data, ["ramp:ramp", "noise:noise:position:1.0", "yaw:yaw:yaw90"])
        self.assertEqual([labels[k].shape for k in ("ramp", "noise", "yaw")], [(3 * N, 2)] * 3)
        info = add_probes(records, states, labels)
        self.assertEqual(info["split_by"], "episode")                # two training episodes: one fits, one is held out
        fitted = records["episode"] == 10
        errors = records["probe_error_ramp"]
        self.assertTrue(np.isnan(errors[fitted]).all() and np.isfinite(errors[~fitted]).all())
        summary = summarize(records)
        held_out = [b for b in summary["buckets"] if b["quantities"]["probe_error_ramp"]["frames"]]
        self.assertGreaterEqual(len(held_out), 8)
        for b in held_out + [summary["overall"]]:
            self.assertGreater(b["probe_r2"]["ramp"], .999)
        self.assertLess(summary["overall"]["probe_r2"]["noise"], .2)
        self.assertEqual(summary["overall"]["quantities"]["probe_error_ramp"]["frames"], len(records["gap"]) // 2)
        with self.assertRaises(ValueError):
            load_labels(data, ["yaw:yaw:angle"])


class ModelContracts(Fixtures):
    def batch(self, count=3):
        data = FullPushTWindows(self.pusht, window=W, gap=8, split="test")
        return default_collate([data[i] for i in (3, 41, 77)[:count]])

    def test_frame_losses_average_to_the_models_own_batch_loss(self):
        from as_jepa_lewm.model import ASJepaLeWM
        from as_jepa_lewm.robust import RobustLeWM
        from as_jepa_lewm.forecast_scored import ForecastScoredLeWM
        batch, n = self.batch(), 3 * (W - 1)
        for cls, kw in ((ASJepaLeWM, {}), (RobustLeWM, dict(rho=.8)), (ForecastScoredLeWM, dict(rho=.8))):
            model = tiny_model(cls, **kw)
            torch.manual_seed(11)
            _, reported = model.loss(batch)
            torch.manual_seed(11)                                    # the loss's own draws, in its order
            noise = torch.randn(3, W, 4, 16)
            uniform = torch.rand(n, 1).clamp_(1e-5, 1 - 1e-5)
            eps = torch.randn(n, 1, 48, 1, 16, 16)
            out = window_quantities(model, batch, noise, uniform, eps)
            for ours, theirs in (("render_loss", "fm"), ("surprise_nats", "surprise")):
                self.assertTrue(bool(out[ours][:, 0].isnan().all()))
                torch.testing.assert_close(out[ours][:, 1:].mean(), reported[theirs].detach(), rtol=1e-5, atol=0)
            torch.testing.assert_close(out["entry_nats"][:, 0].mean(), reported["entry"].detach(), rtol=1e-5, atol=0)
            self.assertTrue(bool(out["entry_nats"][:, 1:].isnan().all()))
            for other in ("render_loss_swapped", "render_loss_swapped_time", "render_loss_zero_anchor"):
                self.assertTrue(bool((out[other][:, 1:] != out["render_loss"][:, 1:]).all()))    # it reads its state and anchor
        # A grid of times: every condition's loss is the mean of its losses at the single times, under the same draws.
        g = torch.Generator().manual_seed(3)
        uniform, eps = torch.rand(n, 2, generator=g), torch.randn(n, 2, 48, 1, 16, 16, generator=g)
        grid = window_quantities(model, batch, noise, uniform, eps)
        single = [window_quantities(model, batch, noise, uniform[:, j:j + 1], eps[:, j:j + 1]) for j in range(2)]
        for name in ("render_loss", "render_loss_swapped", "render_loss_swapped_time", "render_loss_zero_anchor"):
            torch.testing.assert_close(grid[name], (single[0][name] + single[1][name]) / 2, rtol=1e-5, atol=0, equal_nan=True)
        self.assertTrue(torch.equal(grid["surprise_nats"][:, 1:], single[1]["surprise_nats"][:, 1:]))   # `noise` is not consumed

    def test_the_tool_changes_nothing_and_repeats_for_a_seed(self):
        from as_jepa_lewm.robust import RobustLeWM
        model = tiny_model(RobustLeWM, rho=.8, forecast_fraction=.5)
        data = FullPushTWindows(self.pusht, window=W, gap=8, split="test")
        indices = np.random.default_rng(0).permutation(len(data))[:6].tolist()
        probe = self.batch()
        before = fingerprint(model), model.encode(dict(pixels=model.prepare(probe)[0]))["emb"].clone()
        torch.manual_seed(123); np.random.seed(123)
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()[1].copy()
        first, states = collect(model, data, indices, seed=5, batch_size=4)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))                # the global generators did not move
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state))
        self.assertEqual(fingerprint(model), before[0])
        self.assertTrue(torch.equal(model.encode(dict(pixels=model.prepare(probe)[0]))["emb"], before[1]))
        self.assertFalse(model.training)
        again, states_again = collect(model, data, indices, seed=5, batch_size=4)
        for k in first:
            self.assertTrue(np.array_equal(first[k], again[k], equal_nan=True), k)
        self.assertTrue(np.array_equal(states, states_again))
        other, _ = collect(model, data, indices, seed=6, batch_size=4)
        rendered = first["frame"] > 0
        self.assertTrue((first["render_loss"][rendered] != other["render_loss"][rendered]).all())
        for k in KEYS + ("anchor_pixel_mse",):                       # what does not depend on the draws
            self.assertTrue(np.array_equal(first[k], other[k]), k)
        with self.assertRaises(ValueError):
            collect(model.train(), data, indices[:2], seed=5)


class DistortionFamilyContracts(Fixtures):
    """Models that own their `distortion` (the RoboCasa Reach3D family), on windows of the trainer's `Windows`."""
    def windows(self, split="validation", seed=3):
        records = [r for r in json.loads(self.records.read_text())["records"] if r["split"] == split]
        return RobocasaWindows(self.reach, records, dict(seed=seed))

    def batch(self, indices=(0, 5)):
        data = self.windows()
        return data.model_batch(default_collate([data[i] for i in indices]))

    def test_windows_recover_episode_and_start_and_the_gap_is_the_frame_index(self):
        from experiments.robocasa_stable_scaling import train
        data = self.windows()
        items = [data[i] for i in (0, 5, 123456, 7)]
        for item in items:                                           # validation records 0 and 1 are episodes 4 and 5
            for view in range(2):
                shown = 17 * (4 + item["episode"]) + item["start"] + torch.arange(8) + 120 * view
                self.assertEqual(item["render"][view, :, 0, 0, 0].tolist(), shown.tolist())
            self.assertEqual(item["rows"].tolist(), list(range(item["start"], item["start"] + 8)))
            self.assertEqual(item["anchor_row"], item["start"])
        self.assertGreater(len({(item["episode"], item["start"]) for item in items}), 1)
        raw = default_collate(items)
        self.assertEqual(frame_offsets(raw).tolist(), [list(range(8))] * 4)
        raw["pixels"] = torch.arange(4 * 2 * 8 * 3).reshape(4, 2, 8, 3, 1, 1)            # raw-pixel models also get these
        ours = data.model_batch(raw)
        with mock.patch.object(torch.Tensor, "cuda", lambda self, *a, **k: self):       # the trainer's prepare moves to CUDA
            theirs = train.prepare(raw)
        self.assertEqual(set(ours), set(theirs) | {"episode"})
        for k, v in theirs.items():
            self.assertTrue(torch.equal(ours[k], v) if torch.is_tensor(v) else ours[k] == v, k)
        self.assertEqual(ours["episode"].tolist(), raw["episode"].repeat_interleave(2).tolist())
        picked = data.distinct(50, seed=0)                           # two episodes, ten starts each
        self.assertEqual(len({data.locate(i) for i in picked}), len(picked))
        self.assertEqual(len(picked), 20)
        self.assertEqual(picked[:5], data.distinct(5, seed=0))

    def test_frame_values_average_to_the_loss_for_every_variant_without_draws(self):
        """The trainer's patched variants whose distortion is a function of the condition alone. Under the loss's own
        posterior noise, and with its forecast coin replayed, the per-frame values average to the loss's `fm`."""
        batch, step = self.batch(), 3
        variants = dict(future_mean={}, joint_views=dict(joint=True), point_states=dict(point=True),
                        profiled_likelihood=dict(profile="profile-global"))
        for name, kw in variants.items():
            model = reach_model(**kw)
            units = 2 if kw.get("joint") else 4                      # two windows; a joint-view state covers both views
            torch.manual_seed(11)
            with torch.no_grad():
                _, reported = model.loss(batch, step=step)
            torch.manual_seed(11)                                    # the loss's own noise draw
            noise = torch.randn(units, 8, 4, 16)
            seeds = torch.arange(units * 7 * 2).reshape(units, 7, 2)
            out = distortion_quantities(model, batch, noise, seeds, step=step)
            with torch.random.fork_rng(devices=[]):                  # the loss's coin: which frames read the forecast
                torch.manual_seed(model.forecast_seed + step)
                forecast = torch.rand(units, 7, 1, 1)[..., 0, 0] < model.forecast_fraction
            self.assertTrue(bool(forecast.any()) and not bool(forecast.all()), name)
            mixed = torch.where(forecast, out["render_loss_forecast"][:, 1:], out["render_loss"][:, 1:])
            torch.testing.assert_close(mixed.mean(), reported["fm"].float(), rtol=1e-5, atol=0, msg=name)
            torch.testing.assert_close(out["surprise_nats"][:, 1:].mean(), reported["surprise"], rtol=1e-5, atol=0, msg=name)
            torch.testing.assert_close(out["entry_nats"][:, 0].mean(), reported["entry"], rtol=1e-5, atol=0, msg=name)
            for key in ("render_loss", "render_loss_forecast", "surprise_nats"):
                self.assertTrue(bool(out[key][:, 0].isnan().all()) and bool(out[key][:, 1:].isfinite().all()), (name, key))
            self.assertEqual(tuple(out["state"].shape), (units, 8, 64))
            self.assertTrue(bool((out["forecast_excess"][:, 1:] != 0).all()), name)
            for key in ("state_reliance", "state_reliance_time", "anchor_reliance"):
                self.assertTrue(bool((out[key][:, 1:] != 0).all()), (name, key))    # the renderer reads state and anchor
            self.assertIsNone(model.cached)                          # as after the model's own loss
            self.assertIsNone(getattr(model, "_profiled_pixel_metrics", None))
            again = distortion_quantities(model, batch, noise, seeds + 1, step=step)
            self.assertTrue(torch.equal(out["render_loss"][:, 1:], again["render_loss"][:, 1:]), name)   # no draw enters

    def test_a_distortion_that_draws_is_paired_and_averaged_over_seeds(self):
        """The family's default distortion is `flow_loss`, which draws a time and noise per call. Under a seed, the
        value of a frame is the flow loss of that frame with exactly that seed's draws."""
        model, batch = reach_model(objective="flow"), self.batch()
        g = torch.Generator().manual_seed(0)
        noise, seeds = torch.randn(4, 8, 4, 16, generator=g), torch.arange(4 * 7 * 2).reshape(4, 7, 2) + 100
        out = distortion_quantities(model, batch, noise, seeds)
        _, render, anchor = model.prepare(batch)
        for i, t in ((0, 1), (1, 3), (3, 7)):
            y, a = pack(render[i:i + 1, t]), pack(anchor[i:i + 1])
            values = []
            for seed in seeds[i, t - 1].tolist():
                torch.manual_seed(seed)                              # flow_loss's draws, in its order
                uniform = torch.rand(1, 1).clamp_(1e-5, 1 - 1e-5)
                eps = torch.randn_like(y)
                time = draw_timesteps(1, y.device, model.training_shift, uniform=uniform)
                values.append(frame_flow_loss(model, y, out["state"][i, t].reshape(1, 4, 16), a, time, eps))
            torch.testing.assert_close(out["render_loss"][i, t], torch.cat(values).mean(), rtol=1e-5, atol=0)
        self.assertTrue(bool((out["render_loss_swapped"][:, 1:] != out["render_loss"][:, 1:]).all()))
        again = distortion_quantities(model, batch, noise, seeds)
        other = distortion_quantities(model, batch, noise, seeds + 1)
        for key in ("render_loss", "state_reliance", "anchor_reliance", "forecast_excess"):
            self.assertTrue(torch.equal(out[key][:, 1:], again[key][:, 1:]), key)
            self.assertTrue(bool((out[key][:, 1:] != other[key][:, 1:]).all()), key)
        self.assertTrue(torch.equal(out["surprise_nats"][:, 1:], other["surprise_nats"][:, 1:]))   # the rate draws nothing

    def test_a_planted_distortion_shows_its_gap_trend_and_no_state_reliance(self):
        model = reach_model()
        model.distortion = types.MethodType(frame_anchor_distortion, model)
        data = self.windows()
        indices = data.distinct(20, seed=0)
        records, states = collect(model, data, indices, seed=0, batch_size=5)
        self.assertEqual(len(records["gap"]), 20 * 2 * 8)            # windows x views x frames
        self.assertEqual(records["gap"].tolist(), records["frame"].tolist())
        self.assertEqual(records["view"].reshape(-1, 8)[:, 0].tolist(), [0, 1] * 20)
        self.assertEqual(records["window"].reshape(-1, 16)[:, 0].tolist(), indices)
        self.assertEqual(records["row"].tolist(), (records["start"] + records["frame"]).tolist())
        self.assertEqual(tuple(states.shape), (20 * 2 * 8, 64))
        summary = summarize(records)
        self.assertEqual((summary["bucketing"], len(summary["buckets"])), ("exact", 8))
        for g, b in enumerate(summary["buckets"]):                   # frame g shows the anchor's value plus g
            q = b["quantities"]
            self.assertEqual((b["gap_min"], b["frames"], b["windows"]), (g, 40, 20))
            self.assertAlmostEqual(q["anchor_pixel_mse"]["mean"], (g / 127.5) ** 2, places=12)
            if g:
                self.assertAlmostEqual(q["render_loss"]["mean"], (g / 127.5) ** 2, places=9)
                self.assertGreater(abs(q["anchor_reliance"]["mean"]), 1e-3)             # a zeroed anchor is a different image
                for name in ("state_reliance", "state_reliance_time", "forecast_excess"):
                    self.assertEqual((q[name]["frames"], q[name]["mean"]), (40, 0.), name)
        self.assertGreater(summary["spearman"]["render_loss"]["gap"], .999)
        self.assertIsNone(summary["spearman"]["render_loss"]["gap_given_anchor_pixel_mse"])   # the gap is that covariate here
        self.assertIsNone(summary["spearman"]["state_reliance"]["gap"])

        joint = reach_model(joint=True)                              # one state per window renders both views
        joint.original_pixel_distortion = types.MethodType(frame_anchor_distortion, joint)
        records, states = collect(joint, data, indices[:3], seed=0, batch_size=2)
        self.assertEqual((len(records["gap"]), tuple(states.shape)), (3 * 8, (3 * 8, 64)))
        self.assertEqual(set(records["view"]), {-1})
        self.assertEqual(records["window"][::8].tolist(), indices[:3])
        rendered = records["frame"] > 0
        np.testing.assert_allclose(records["render_loss"][rendered], (records["gap"][rendered] / 127.5) ** 2, rtol=1e-6)
        np.testing.assert_allclose(records["anchor_pixel_mse"], (records["gap"] / 127.5) ** 2, rtol=1e-12)

    def test_the_distortion_path_changes_nothing_and_repeats_for_a_seed(self):
        model, data = reach_model(objective="flow"), self.windows()
        pool = data.distinct(20, seed=1)                             # two windows of different episodes, then a third
        indices = [next(i for i in pool if data.locate(i)[0] == e) for e in (0, 1)]
        indices.append(next(i for i in pool if i not in indices))
        before = fingerprint(model)
        torch.manual_seed(123); np.random.seed(123); random.seed(123)
        kept = torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()
        first, states = collect(model, data, indices, seed=5, batch_size=2, draws=2)
        self.assertTrue(torch.equal(torch.get_rng_state(), kept[0]))                    # no generator moved
        self.assertTrue(np.array_equal(np.random.get_state()[1], kept[1]))
        self.assertEqual(random.getstate(), kept[2])
        self.assertEqual(fingerprint(model), before)
        self.assertFalse(model.training)
        self.assertIsNone(model.cached)
        again, states_again = collect(model, data, indices, seed=5, batch_size=2, draws=2)
        for k in first:
            self.assertTrue(np.array_equal(first[k], again[k], equal_nan=True), k)
        self.assertTrue(np.array_equal(states, states_again))
        other, _ = collect(model, data, indices, seed=6, batch_size=2, draws=2)
        rendered = first["frame"] > 0
        self.assertTrue((first["render_loss"][rendered] != other["render_loss"][rendered]).all())
        for k in KEYS + ("anchor_pixel_mse", "surprise_nats", "entry_nats"):            # what does not depend on the draws
            self.assertTrue(np.array_equal(first[k], other[k], equal_nan=True), k)
        # the last batch holds one window: its two views are each other's partner and are flagged
        self.assertEqual(first["swap_same_episode"].reshape(-1, 8)[:, 1].tolist(), [0.] * 4 + [1.] * 2)
        with self.assertRaises(ValueError):
            collect(model.train(), data, indices[:1], seed=5)


class CommandLineContract(Fixtures):
    def report(self, *argv):
        run_main(*argv, "--output", self.root / "out")
        text = (self.root / "out" / "gap_buckets.json").read_text()
        self.assertNotIn("NaN", text)                                # strict JSON: missing values are null
        return json.loads(text)

    def checkpoint(self, name="ckpt", **kw):
        from as_jepa_lewm.robust import RobustLeWM
        return save_checkpoint(tiny_model(RobustLeWM, rho=.8, forecast_fraction=.5, **kw), self.root / name,
                               "as_jepa_lewm.robust.RobustLeWM", rho=.8, forecast_fraction=.5, **kw)

    def test_benchmark_windows_with_and_without_the_gap_rule(self):
        ckpt = self.checkpoint()
        labels = ("--label", "ramp:ramp", "--label", "yaw:yaw:yaw90:12")
        common = ("--checkpoint", ckpt, "--data", self.bench, "--dataset", "benchmark", "--cases", 12, "--seed", 3)
        first = self.report(*common, *labels)
        self.assertEqual([first[k] for k in ("dataset", "anchor_rule_gap", "window", "rho")], ["BenchmarkWindows", 0, W, .8])
        self.assertEqual(first["bucketing"], "exact")
        self.assertEqual(first["probes"]["split_by"], "window")      # one validation episode
        self.assertEqual(len(first["flow_times"]), 5)
        records = first["records"]
        self.assertEqual(records["gap"], records["frame"])           # first-frame anchor: the gap is the frame index
        self.assertEqual(records["gap"], [0, 1, 2, 3] * 12)
        for g, b in enumerate(first["buckets"]):
            self.assertEqual((b["gap_min"], b["gap_max"], b["frames"], b["windows"]), (g, g, 12, 12))
            q = b["quantities"]
            self.assertEqual(q["entry_nats"]["frames"], 12 if g == 0 else 0)
            for name in ("render_loss", "state_reliance", "state_reliance_time", "anchor_reliance", "surprise_nats"):
                self.assertEqual(q[name]["frames"], 0 if g == 0 else 12)
                self.assertTrue(g == 0 or (np.isfinite(q[name]["mean"]) and q[name]["sem"] > 0))
            self.assertEqual(q["anchor_pixel_mse"]["frames"], 12)
            # every pixel differs from the anchor's by STRIDE * g of 255 levels, in [-1, 1] units
            self.assertAlmostEqual(q["anchor_pixel_mse"]["mean"], (STRIDE * g / 127.5) ** 2, places=12)
            self.assertEqual(set(b["probe_r2"]), {"ramp", "yaw"})
        self.assertEqual(len(set(records["anchor_pixel_mse"])), W)   # equal pixel differences tie exactly
        self.assertEqual(set(first["spearman"]["render_loss"]), {"frames", "gap", "gap_given_anchor_pixel_mse"})
        self.assertEqual(first["spearman"]["render_loss"]["frames"], 36)
        self.assertIsNone(first["spearman"]["entry_nats"]["gap"])    # every entry frame has gap 0

        gapped = self.report(*common, "--gap", 2, "--times", .5)
        self.assertEqual([gapped[k] for k in ("dataset", "anchor_rule_gap", "time_quantiles")], ["GapBenchmarkWindows", 2, [.5]])
        self.assertIsNone(gapped["probes"])
        data, records = GapBenchmarkWindows(self.bench, window=W, split="validation", gap=2), gapped["records"]
        self.assertEqual(records["window"], first["records"]["window"])                 # the seed picks the same windows
        for i in range(12):
            item = data[records["window"][i * W]]
            expected = [(int(r) - item["anchor_row"]) // STRIDE for r in item["rows"]]
            self.assertEqual(records["offset"][i * W:(i + 1) * W], expected)
            self.assertEqual(records["gap"][i * W:(i + 1) * W], expected)                # past only: no sign to drop
            self.assertEqual(records["anchor_fallback"][i * W], float(item["anchor_fallback"]))
        self.assertGreater(max(records["gap"]), W - 1)
        self.assertEqual(gapped["overall"]["quantities"]["anchor_future"]["mean"], 0.)

    def test_pusht_full_and_kept_frame_windows(self):
        ckpt = self.checkpoint()
        full = self.report("--checkpoint", ckpt, "--data", self.pusht, "--cases", 16, "--label", "agent:0-2", "--batch-size", 5)
        self.assertEqual([full[k] for k in ("dataset", "split", "anchor_rule_gap", "cases")], ["FullPushTWindows", "test", 8, 16])
        data, records = FullPushTWindows(self.pusht, window=W, gap=8, split="test"), full["records"]
        self.assertEqual(records["window"][::W], np.random.default_rng(0).permutation(len(data))[:16].tolist())
        for i in range(16):
            item = data[records["window"][i * W]]
            expected = [(int(r) - item["anchor_row"]) // 5 for r in item["rows"]]
            self.assertEqual(records["offset"][i * W:(i + 1) * W], expected)
            self.assertEqual(records["gap"][i * W:(i + 1) * W], [abs(x) for x in expected])
            self.assertEqual(records["anchor_future"][i * W:(i + 1) * W], [float(x < 0) for x in expected])
        self.assertEqual(sum(b["frames"] for b in full["buckets"]), 16 * W)
        self.assertEqual(full["bucketing"], "exact" if len(set(records["gap"])) <= 12 else "quantile")
        self.assertEqual(full["probes"]["split_by"], "episode")
        self.assertIn("agent", full["overall"]["probe_r2"])
        self.assertEqual(full["overall"]["quantities"]["render_loss"]["frames"], 16 * (W - 1))
        # the last batch of five holds one window: it has no partner window, its other quantities stay
        self.assertEqual(full["overall"]["quantities"]["state_reliance"]["frames"], 15 * (W - 1))
        self.assertEqual(full["overall"]["quantities"]["state_reliance_time"]["frames"], 16 * (W - 1))

        kept = self.report("--checkpoint", ckpt, "--data", self.pusht, "--dataset", "pusht", "--cases", 6, "--gap", 3)
        self.assertEqual((kept["dataset"], kept["anchor_rule_gap"]), ("PushTWindows", 3))
        self.assertGreaterEqual(min(kept["records"]["gap"]), 0)
        with self.assertRaises(ValueError):                          # longer windows than the model was trained on
            run_main("--checkpoint", ckpt, "--data", self.pusht, "--window", 8, "--output", self.root / "out")

    def test_robocasa_checkpoint_through_the_trainers_build_and_windows(self):
        """The loader path on synthetic files, with the trainer's full-size model (the slow test): the campaign
        layout <root>/training/<run>/<label>/weights.pt and <run>/config.json, records from a data.json."""
        from experiments.robocasa_stable_scaling.train import build
        root = self.root / "campaign"
        label = root / "training" / "run-a" / "h01"
        label.mkdir(parents=True)
        cfg = dict(variant="spt-future-mean", encoder="vjepa", objective="spt", seed=0)
        (label.parent / "config.json").write_text(json.dumps(cfg))
        model = randomize_zeros(build(root, cfg))
        self.assertEqual(model.distortion.__func__.__name__, "pixel_prediction_distortion")    # a patched distortion
        torch.save(model.state_dict(), label / "weights.pt")
        common = ("--checkpoint", label, "--data", self.records, "--dataset", "robocasa", "--cases", 2, "--batch-size", 2)
        report = self.report(*common)
        self.assertEqual([report[k] for k in ("dataset", "split", "anchor_rule_gap", "cases", "window", "rho")],
                         ["RobocasaWindows", "validation", 0, 2, 8, .8])
        self.assertEqual([report[k] for k in ("render_loss", "time_quantiles", "flow_times")], ["model.distortion", None, None])
        records = report["records"]
        self.assertEqual(records["gap"], list(range(8)) * 4)         # two windows, two views; the gap is the frame index
        self.assertEqual(records["gap"], records["frame"])
        self.assertEqual(records["view"][::8], [0, 1, 0, 1])
        self.assertTrue(set(records["episode"]) <= {0, 1})           # positions in the two validation records
        self.assertEqual((report["bucketing"], len(report["buckets"])), ("exact", 8))
        for g, b in enumerate(report["buckets"]):
            q = b["quantities"]
            self.assertEqual((b["gap_min"], b["gap_max"], b["frames"], b["windows"]), (g, g, 4, 2))
            self.assertAlmostEqual(q["anchor_pixel_mse"]["mean"], (g / 127.5) ** 2, places=12)
            self.assertEqual(q["entry_nats"]["frames"], 4 if g == 0 else 0)
            for name in ("render_loss", "state_reliance", "state_reliance_time", "anchor_reliance", "render_loss_forecast",
                         "forecast_excess", "surprise_nats"):
                self.assertEqual(q[name]["frames"], 0 if g == 0 else 4, name)
                self.assertTrue(g == 0 or (np.isfinite(q[name]["mean"]) and q[name]["mean"] != 0), name)
        self.assertEqual(set(report["spearman"]["forecast_excess"]), {"frames", "gap", "gap_given_anchor_pixel_mse"})
        for extra in (("--label", "agent:0-2"), ("--gap", 2), ("--split", "test")):     # no labels, one anchor rule, no such split
            with self.assertRaises(ValueError):
                run_main(*common, *extra, "--output", self.root / "out")


if __name__ == "__main__":
    unittest.main()
