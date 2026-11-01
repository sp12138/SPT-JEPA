"""Contracts for the rollout diagnostics (round-2 spec, Sections 5.3 and 10).

4 tests. CPU only. Run from the repository root:
    PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_rollout_gain.py -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from as_jepa_tools.rollout_gain import displacement_gain, main, one_step_gain, rollout_metrics

TINY = dict(mode="L2", beta=0.25, sigma0=0.1, window=4, encoder_depth=1, renderer_width=64,
            renderer_depth=1, renderer_heads=1, encoder_size=56)


class MetricContracts(unittest.TestCase):
    def test_metrics_recover_a_motion_attenuation(self):
        """Predicted displacement = a * true displacement, levels carried exactly."""
        g = torch.Generator().manual_seed(0)
        last = torch.randn(4096, 32, generator=g)
        moves = torch.randn(4096, 5, 32, generator=g)
        for a in (.5, 1., 1.25):
            out = rollout_metrics(last[:, None] + moves, last[:, None] + a * moves, last)
            for k in range(5):
                self.assertAlmostEqual(out["motion_gain"][k], a, places=6)
                self.assertAlmostEqual(out["error_vs_copy"][k], (1 - a) ** 2, places=6)
                self.assertAlmostEqual(out["carry_ratio"][k], 1., delta=.03)
                self.assertAlmostEqual(out["spread_ratio"][k], (1 + a * a) / 2, delta=.03)

    def test_metrics_detect_a_centre_pull(self):
        """The rate-optimal prior of spec Appendix A.8: m = a * s + motion, with a = r^2 / (1 + r^2) per dimension."""
        g = torch.Generator().manual_seed(1)
        r = torch.tensor([.5, .5, 1., 1., 2., 2., 4., 4.]); a = r ** 2 / (1 + r ** 2); q, n, h = .5, 40000, 5
        level = r * torch.randn(n, 8, generator=g)
        moves = q * torch.randn(n, h, 8, generator=g)
        truth = level[:, None] + moves.cumsum(1)
        state, rolled = level, []
        for k in range(h):
            state = a * state + moves[:, k]
            rolled.append(state)
        out = rollout_metrics(truth, torch.stack(rolled, 1), level)
        for k in range(h):
            for i, ai in enumerate(a[::2]):                          # quartiles hold the dimension pairs, lowest spread first
                self.assertAlmostEqual(out["by_spread_quartile"][f"q{i + 1}"]["carry_ratio"][k], float(ai ** (k + 1)), delta=.03)
            lag = float(sum(a ** (k - j) for j in range(k + 1)).mean() / (k + 1))
            self.assertAlmostEqual(out["motion_gain"][k], lag, delta=.02)
            self.assertLess(out["spread_ratio"][k], 1.)
            self.assertLess(out["by_spread_quartile"]["q1"]["spread_ratio"][k], out["by_spread_quartile"]["q4"]["spread_ratio"][k])
        # Revision 3's first reading (true on predicted displacement) falls below 1 here, not above it.
        self.assertLess(displacement_gain(truth[:, -1], rolled[-1], level), 1.)

    def test_one_step_gain_clean_and_noisy(self):
        g = torch.Generator().manual_seed(2)
        mu = torch.randn(64, 8, 64, generator=g)

        class Shrinking:                                             # predicts mu_t moved toward by a factor k
            window, rho = 8, 0.
            def __init__(self, k, target): self.k, self.target = k, target
            def predict(self, emb, act): return emb + self.k * (self.target[:, 1:emb.shape[1] + 1] - emb)

        for k in (.5, 1.):
            self.assertAlmostEqual(one_step_gain(Shrinking(k, mu), mu, torch.zeros(64, 8, 4)), 1 / k, places=6)
        noise = .1 * torch.randn(mu.shape, generator=g)              # noisy inputs, AR(1) target with rho = 0.5
        model = Shrinking(1., mu + .5 * torch.cat([torch.zeros_like(noise[:, :1]), noise[:, :-1]], 1))
        model.rho = .5
        self.assertAlmostEqual(one_step_gain(model, mu, torch.zeros(64, 8, 4), noise), 1., places=6)


class CommandLineContract(unittest.TestCase):
    def test_runs_end_to_end_on_benchmark_windows(self):
        from as_jepa_lewm.model import ASJepaLeWM
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, n = root / "data.h5", 60
            with h5py.File(raw, "w") as f:
                f["pixels"] = np.random.default_rng(0).integers(0, 255, (2 * n, 16, 16, 3), dtype=np.uint8)
                f["action"] = np.random.default_rng(1).normal(size=(2 * n, 2)).astype(np.float32)
            for k, v in dict(offsets=[0, n], lengths=[n, n], episode_ids=[10, 20],
                             train_episodes=[10], validation_episodes=[20]).items():
                np.save(root / (k + ".npy"), v)
            (root / "dataset_card.json").write_text(json.dumps(dict(h5_path=str(raw), frameskip=5,
                                                                    action_mean=[0, 0], action_std=[1, 1])))
            torch.manual_seed(0)
            model = ASJepaLeWM(**TINY)
            with torch.no_grad():                                    # make the zero-initialized carry move
                for p in model.parameters():
                    if p.abs().sum() == 0:
                        p.copy_(0.02 * torch.randn(p.shape))
            ckpt = root / "ckpt"; ckpt.mkdir()
            torch.save(model.state_dict(), ckpt / "weights.pt")
            (ckpt / "config.json").write_text(json.dumps(dict(_target_="as_jepa_lewm.model.ASJepaLeWM", **TINY)))
            argv = ["rollout_gain", "--checkpoint", str(ckpt), "--data", str(root), "--output", str(root / "out"),
                    "--dataset", "benchmark", "--cases", "6", "--window", "8", "--context", "3", "--device", "cpu"]
            with mock.patch.object(sys, "argv", argv):
                main()
            report = json.loads((root / "out" / "rollout_gain.json").read_text())
        self.assertEqual(report["cases"], 6)
        ol = report["open_loop"]
        self.assertEqual(len(ol["spread_ratio"]), 5)
        self.assertEqual(set(ol["by_spread_quartile"]), {"q1", "q2", "q3", "q4"})
        self.assertEqual(len(report["one_step_clean"]["motion_gain"]), 1)
        values = [report["one_step_noisy_gain"], report["state_spread_over_sigma0"],
                  report["noisy_state_information_upper_bound_nats"],
                  *ol["motion_gain"], *ol["carry_ratio"], *ol["spread_ratio"], *ol["error_vs_copy"]]
        self.assertTrue(all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
