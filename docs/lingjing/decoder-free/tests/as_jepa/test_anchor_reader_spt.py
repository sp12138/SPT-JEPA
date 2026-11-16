"""Contracts for the anchor-reader adapter of the AnalyticResidualSPT family (experiments/anchor_reader_spt.py).
CPU, tiny models, raw pixels, the initial frame as anchor (prepare_pixels of robocasa_stable_scaling/raw_pixels.py).

Run from the repository root:
    PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_anchor_reader_spt.py -v
"""
import math
import types
import unittest
from unittest import mock

import torch

from vforge.ops import OpsConfig, initialize_ops, reset_ops

W, B, ENC = 4, 3, 56
TINY = dict(mode="L2", beta=.25, sigma0=.1, window=W, anchor_gap=0, encoder_depth=1, encoder_size=ENC,
            renderer_width=48, renderer_depth=2, renderer_heads=3, action_dim=14,
            rho=.8, forecast_fraction=.5, forecast_seed=61001, rate_prior="forecast")
READER = dict(beta_nats=0.3, reader_width=32, reader_depth=1, reader_draws=2)


def setUpModule():
    reset_ops()
    initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=torch.device("cpu"), parallel_state=None)


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(pixels=torch.randint(0, 256, (B, W, 3, ENC, ENC), generator=g).float(),
                render=torch.randint(0, 256, (B, W, 3, 64, 64), generator=g).float(),
                action=torch.randn(B, W, 14, generator=g))


def perturb(model):
    """Zero-initialized layers (identity carry, AdaLN-zero) hide gradients at initialization."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.abs().sum() == 0:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))


def build(attach=True, **reader):
    from experiments.robocasa_refine.analytic_model import AnalyticResidualSPT
    from experiments.robocasa_stable_scaling.raw_pixels import prepare_pixels
    from experiments.anchor_reader_spt import attach_anchor_reader
    torch.manual_seed(0)
    m = AnalyticResidualSPT(**TINY)
    m.raw_render_size = 64
    m.prepare = types.MethodType(prepare_pixels, m)            # the anchor image is the initial frame
    if attach:
        attach_anchor_reader(m, **{**READER, **reader})
    perturb(m); m.train()
    return m


def record_first_call(model, name):
    """Wrap a bound method so the positional arguments of its first call are kept."""
    seen, original = [], getattr(model, name)

    def wrapper(*args, **kwargs):
        if not seen:
            seen.append([a.detach().clone() for a in args if torch.is_tensor(a)])
        return original(*args, **kwargs)
    setattr(model, name, wrapper)
    return seen


class AnchorReaderSPTTest(unittest.TestCase):
    def setUp(self):
        self.batch = make_batch()

    def test_attach_keeps_tensors_and_the_main_stream(self):
        torch.manual_seed(0); ref = build(attach=False); a = torch.rand(4)
        torch.manual_seed(0); m = build(); b = torch.rand(4)
        self.assertTrue(torch.equal(a, b))
        ref_sd, sd = ref.state_dict(), m.state_dict()
        self.assertEqual([k for k in sd if not k.startswith("reader.")], list(ref_sd))
        for k, v in ref_sd.items():
            self.assertTrue(torch.equal(v, sd[k]), k)
        self.assertTrue(any(k.startswith("reader.frame.") for k in sd))
        self.assertEqual((m.beta_nats, m.reader_loss_weight, m.reader_lr), (0.3, 0.01, 5e-4))

    def test_rate_terms_equal_the_original_loss_bitwise(self):
        from experiments.robocasa_refine.analytic_model import AnalyticResidualSPT
        for geometry in ("markov", "root-marginal"):
            ref, m = build(attach=False), build()
            ref.forecast_rate_geometry = m.forecast_rate_geometry = geometry
            ref_predict = record_first_call(ref, "predict"); m_predict = record_first_call(m, "predict")
            m_reader = record_first_call(m.reader, "forward")
            torch.manual_seed(5); _, met_r = AnalyticResidualSPT.loss(ref, self.batch, 3)
            torch.manual_seed(5); _, met_m = m.loss(self.batch, 3)
            for k in ("entry", "surprise", "rate", "feature_std", "active_dims", "temporal_noise_rho",
                      "clean_rollout_mse", "analytic_coding_noise"):      # clean_rollout_mse pins `means`
                self.assertTrue(torch.equal(met_r[k], met_m[k]), (geometry, k))
            # `noisy`: the reference forward reads noisy[:, :-1] in both; the reader reads noisy[:, 1:]
            self.assertTrue(torch.equal(ref_predict[0][0], m_predict[0][0]), geometry)
            noisy_head, noisy_tail = m_predict[0][0], m_reader[0][0].flatten(2)
            self.assertTrue(torch.equal(noisy_head[:, 1:], noisy_tail[:, :-1]), geometry)
            self.assertEqual(set(met_r) - set(met_m), {"fm", "rate_share", "forecast_fraction"})
            self.assertEqual(set(met_m) - set(met_r), {"reader_loss", "reader_mse", "score_rms"})
            for k, v in met_m.items():
                self.assertTrue(math.isfinite(float(v.detach())), k)

    def test_loss_is_the_stated_combination(self):
        from as_jepa_lewm.anchor_reader import encoder_entropy_term, reader_denoising_term, reader_stream_seed
        m = build(reader_loss_weight=0.05)
        m_predict = record_first_call(m, "predict"); m_reader = record_first_call(m.reader, "forward")
        torch.manual_seed(5); loss, met = m.loss(self.batch, 3)
        noisy_tail, cond = m_reader[0]
        with torch.no_grad():
            pixels, _, anchor = m.prepare(self.batch)
            mu = m.encode(dict(pixels=pixels))["emb"].reshape(B, W, 4, 16)
            self.assertTrue(torch.equal(cond, m.reader.condition(mu[:, 0], anchor)))   # initial code and initial frame
            ent, stats = encoder_entropy_term(m.reader, mu[:, 1:], noisy_tail, cond, m.sigma0)
            reader_loss = reader_denoising_term(m.reader, mu[:, 1:], cond, m.sigma0, m.reader_draws,
                                                reader_stream_seed(m.reader_seed, 3)).mean()
        T = W - 1
        nats = met["entry"] + T * met["surprise"]                        # true nats per window
        expect = (m.beta_nats * nats + ent.sum(1).mean()) / T + 0.05 * reader_loss
        self.assertTrue(torch.allclose(loss, expect, rtol=1e-4, atol=1e-4), (loss, expect))
        self.assertTrue(torch.allclose(met["reader_loss"], reader_loss, rtol=1e-6))
        self.assertTrue(torch.equal(met["reader_mse"], stats["reader_mse"]))
        # each frame's marginal noise is sigma0 although the noise is correlated across frames
        self.assertAlmostEqual(float(((noisy_tail - mu[:, 1:]) / m.sigma0).std()), 1., delta=0.15)

    def test_gradient_routing(self):
        """Two weights on the reader's loss: the encoder side must not notice (it never receives that loss),
        and the reader's gradient must scale with the weight (it receives nothing else)."""
        grads = {}
        for weight in (0.01, 10.):
            m = build(reader_loss_weight=weight)
            torch.manual_seed(5); loss, _ = m.loss(self.batch, 3); loss.backward()
            grads[weight] = {n: p.grad for n, p in m.named_parameters()}
        small, large = grads[0.01], grads[10.]
        norm = lambda prefix, g: float(sum(v.abs().sum() for n, v in g.items() if v is not None and n.startswith(prefix)))
        for prefix in ("encoder.", "pool.", "predictor.", "action_encoder.", "reader.", "reader.frame."):
            self.assertGreater(norm(prefix, small), 0, prefix)
        self.assertEqual(norm("renderer.", small), 0.)                     # nothing trains the renderer
        for n, g in small.items():
            if n.startswith("reader."):
                # proportional up to float32 rounding, measured against the tensor's largest entry
                self.assertLessEqual(float((large[n] - 1000. * g).abs().max()), 1e-5 * float(large[n].abs().max()), n)
            elif g is None:
                self.assertIsNone(large[n], n)
            else:
                self.assertTrue(torch.equal(large[n], g), n)

    def test_the_renderer_is_never_called(self):
        m = build()
        forbidden = AssertionError("the anchor-reader loss must not render")
        with mock.patch.object(m.renderer, "forward", side_effect=forbidden), \
                mock.patch.object(m, "distortion", side_effect=forbidden):
            torch.manual_seed(5); loss, _ = m.loss(self.batch, 3)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_strict_reload_and_validation(self):
        from experiments.anchor_reader_spt import attach_anchor_reader
        m = build()
        torch.manual_seed(9); other = build(reader_seed=7)
        self.assertFalse(torch.equal(other.reader.out.weight, m.reader.out.weight))
        other.load_state_dict(m.state_dict(), strict=True); other.reader_seed = m.reader_seed
        torch.manual_seed(5); loss_a, _ = m.loss(self.batch, 3)
        torch.manual_seed(5); loss_b, _ = other.loss(self.batch, 3)
        self.assertTrue(torch.equal(loss_a, loss_b))
        with self.assertRaises(RuntimeError):
            build(attach=False).load_state_dict(m.state_dict(), strict=True)
        with self.assertRaises(ValueError):
            m.loss(self.batch, 3, ramp_steps=10)
        for kw in (dict(beta_nats=0.), dict(beta_nats=0.3, reader_anchor="pixels"), dict(beta_nats=0.3, reader_draws=0),
                   dict(beta_nats=0.3, reader_loss_weight=0.)):
            with self.assertRaises(ValueError, msg=kw):
                attach_anchor_reader(build(attach=False), **kw)
        with self.assertRaises(TypeError):
            attach_anchor_reader(build(attach=False))                      # beta_nats has no default
        laplace = build(attach=False); laplace.coding_family = "laplace"
        with self.assertRaises(ValueError):
            attach_anchor_reader(laplace, beta_nats=0.3)


if __name__ == "__main__":
    unittest.main()
