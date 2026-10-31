"""Round-2 contracts: forecast-scored renderer (lambda), learned variances, reconstruction-free variant.

11 tests.

CPU only, tiny models. Run from the repository root:
    PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_round2.py -v
"""
import math
import unittest

import torch
from torch import nn

from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.models.as_jepa.codes import entry_nats, surprise_nats

W, B, ENC = 4, 3, 56
TINY = dict(mode="L2", beta=0.25, sigma0=0.1, window=W, encoder_depth=1, renderer_width=64,
            renderer_depth=1, renderer_heads=1, encoder_size=ENC)


def setUpModule():
    reset_ops()
    initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=torch.device("cpu"), parallel_state=None)


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(pixels=torch.randint(0, 256, (B, W, 3, ENC, ENC), generator=g).float(),
                anchor224=torch.randint(0, 256, (B, 3, ENC, ENC), generator=g).float(),
                render=torch.randint(0, 256, (B, W, 3, 64, 64), generator=g).float(),
                anchor=torch.randint(0, 256, (B, 3, 64, 64), generator=g).float(),
                action=torch.randn(B, W, 10, generator=g))


def perturb(model):
    """Zero-initialized layers (identity carry, AdaLN-zero, renderer output) hide gradients and
    dependencies at initialization; give them small random values for contract tests."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.abs().sum() == 0:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))


def build(cls, **kw):
    torch.manual_seed(0)
    m = cls(**{**TINY, **kw}); perturb(m); m.train()
    return m


def flow_grad_split(model, batch, step=0):
    model.zero_grad()
    torch.manual_seed(5)
    _, met = model.loss(batch, step=step)
    met["fm"].backward()
    def total(prefixes):
        return float(sum(p.grad.abs().sum() for n, p in model.named_parameters()
                         if p.grad is not None and n.startswith(prefixes)))
    return total(("encoder.", "pool.")), total(("predictor.", "action_encoder."))


class ForecastScoredTest(unittest.TestCase):
    def setUp(self):
        from as_jepa_lewm.model import ASJepaLeWM
        from as_jepa_lewm.forecast_scored import ForecastScoredLeWM
        self.Base, self.Lam = ASJepaLeWM, ForecastScoredLeWM
        self.base = build(ASJepaLeWM)
        self.batch = make_batch()

    def paired(self, **kw):
        m = build(self.Lam, **kw); m.load_state_dict(self.base.state_dict())
        return m

    def test_lambda_zero_reproduces_l2_loss(self):
        for kw in (dict(lam=0.), dict(lam=0., horizon="rollout"), dict(lam=0., forecast_grad="encoder_only")):
            m = self.paired(**kw)
            torch.manual_seed(5); l0, _ = self.base.loss(self.batch, step=3)
            torch.manual_seed(5); l1, _ = m.loss(self.batch, step=3)
            self.assertTrue(torch.equal(l0, l1), kw)

    def test_main_random_stream_is_paired_across_lambda(self):
        for kw in (dict(lam=1.), dict(lam=.5), dict(lam=1., horizon="rollout"), dict(lam=1., forecast_grad="encoder_only")):
            m = self.paired(**kw)
            torch.manual_seed(5); self.base.loss(self.batch, step=3); a = torch.rand(4)
            torch.manual_seed(5); m.loss(self.batch, step=3); b = torch.rand(4)
            self.assertTrue(torch.equal(a, b), kw)

    def test_forecast_share(self):
        m = build(self.Lam, lam=.5)
        share = sum(m.loss(self.batch, step=k)[1]["forecast_share"].item() for k in range(40)) / 40
        self.assertLess(abs(share - .5), .06)
        self.assertEqual(build(self.Lam, lam=1.).loss(self.batch, step=0)[1]["forecast_share"].item(), 1.)

    def test_gradient_routing(self):
        enc, pri = flow_grad_split(build(self.Lam, lam=0.), self.batch)
        self.assertGreater(enc, 0); self.assertEqual(pri, 0)        # lambda = 0: flow loss never reaches the prior
        for kw in (dict(lam=1.), dict(lam=1., horizon="rollout")):
            enc, pri = flow_grad_split(build(self.Lam, **kw), self.batch)
            self.assertGreater(enc, 0, kw); self.assertGreater(pri, 0, kw)
        enc, pri = flow_grad_split(build(self.Lam, lam=1., forecast_grad="encoder_only"), self.batch)
        self.assertGreater(enc, 0); self.assertEqual(pri, 0)        # fallback: states yes, prior parameters no

    def test_lambda_one_never_renders_a_frame_from_its_own_encoding(self):
        m = build(self.Lam, lam=1.)
        pix = self.batch["pixels"].clone().requires_grad_(True)
        torch.manual_seed(5); _, met = m.loss(dict(self.batch, pixels=pix), step=0); met["fm"].backward()
        self.assertEqual(pix.grad[:, -1].abs().sum().item(), 0.)    # last frame: only forecasts of later frames could use it
        self.assertGreater(pix.grad[:, 0].abs().sum().item(), 0.)   # first frame feeds the forecast of frame 1

    def test_calibration_slope(self):
        from as_jepa_lewm.forecast_scored import calibration_slope
        m = build(self.Lam)
        mu = torch.randn(64, W, 4, 16); s = mu + .1 * torch.randn_like(mu)
        class Scaled(nn.Module):
            def __init__(self, k): super().__init__(); self.k = k
            def forward(self, emb, act): return self.k * (mu[:, 1:].flatten(2) - emb)
        for k in (1., 1.2):
            m.predictor = Scaled(k)
            self.assertAlmostEqual(calibration_slope(m, s, mu, torch.zeros(64, W, 64)), 1 / k, places=5)


class LearnedVarianceTest(unittest.TestCase):
    def test_kl_functions(self):
        from as_jepa_lewm.learned_variance import entry_nats_lv, surprise_nats_lv
        mu0, mu, m = torch.randn(5, 4, 16), torch.randn(5, 3, 4, 16), torch.randn(5, 3, 4, 16)
        lv = torch.full_like(mu, 2 * math.log(.1))
        self.assertTrue(torch.allclose(entry_nats_lv(mu0, lv[:, 0]), entry_nats(mu0, .1), rtol=1e-6, atol=1e-4))
        self.assertTrue(torch.allclose(surprise_nats_lv(mu, lv, m, lv), surprise_nats(mu, m, .1), rtol=1e-6, atol=1e-4))
        lvq, lvp = torch.randn_like(mu) - 3, torch.randn_like(mu) - 3
        ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu, (.5 * lvq).exp()),
                                                torch.distributions.Normal(m, (.5 * lvp).exp())).sum((-2, -1))
        self.assertTrue(torch.allclose(surprise_nats_lv(mu, lvq, m, lvp), ref, rtol=1e-5, atol=1e-3))

    def test_initial_loss_equals_fixed_variance(self):
        from as_jepa_lewm.forecast_scored import ForecastScoredLeWM
        from as_jepa_lewm.learned_variance import LearnedVarianceLeWM
        x = make_batch()
        fixed = build(ForecastScoredLeWM)
        torch.manual_seed(0); lvm = LearnedVarianceLeWM(**TINY)
        sd = lvm.state_dict()
        halves = ("pool.head.weight", "pool.head.bias",
                  "predictor.transformer.output_proj.weight", "predictor.transformer.output_proj.bias")
        for k, v in fixed.state_dict().items():
            (sd[k][: v.shape[0]] if k in halves else sd[k]).copy_(v)
        lvm.load_state_dict(sd); lvm.train()
        torch.manual_seed(5); l0, _ = fixed.loss(x, step=2)
        torch.manual_seed(5); l1, met = lvm.loss(x, step=2)
        self.assertTrue(torch.allclose(l0, l1, rtol=1e-5, atol=1e-6))
        self.assertAlmostEqual(met["posterior_sd"].item(), .1, places=6)
        self.assertAlmostEqual(met["prior_sd"].item(), .1, places=6)
        lvm.zero_grad(); torch.manual_seed(5); lvm.loss(x, step=2)[0].backward()
        self.assertGreater(lvm.pool.head.bias.grad[16:].abs().sum().item(), 0)
        self.assertGreater(lvm.predictor.transformer.output_proj.bias.grad[64:].abs().sum().item(), 0)
        self.assertEqual(tuple(lvm.predict(torch.randn(2, 3, 64), torch.randn(2, 3, 64)).shape), (2, 3, 64))


    def test_forecast_gradient_routing(self):
        from as_jepa_lewm.learned_variance import LearnedVarianceLeWM
        x = make_batch()
        enc, pri = flow_grad_split(build(LearnedVarianceLeWM, lam=1.), x)
        self.assertGreater(enc, 0); self.assertGreater(pri, 0)
        enc, pri = flow_grad_split(build(LearnedVarianceLeWM, lam=1., forecast_grad="encoder_only"), x)
        self.assertGreater(enc, 0); self.assertEqual(pri, 0)


class MetricsTest(unittest.TestCase):
    """Every class logs the same core metrics, each convertible the way train.py converts them."""
    CORE = {"loss", "rate", "entry", "surprise", "feature_std", "active_dims", "forecast_share", "calib_slope"}

    def test_core_metrics(self):
        from as_jepa_lewm.forecast_scored import ForecastScoredLeWM
        from as_jepa_lewm.learned_variance import LearnedVarianceLeWM
        from as_jepa_lewm.nce import NCELeWM
        x = make_batch()
        extra = {ForecastScoredLeWM: {"fm", "rate_share"},
                 LearnedVarianceLeWM: {"fm", "rate_share", "prior_var_ratio", "posterior_sd", "prior_sd"},
                 NCELeWM: {"nce", "info_bound"}}
        for cls, keys in extra.items():
            _, met = build(cls, lam=1.).loss(x, step=1)
            self.assertTrue(self.CORE | keys <= set(met), (cls.__name__, sorted(self.CORE | keys - set(met))))
            values = {k: float(v.detach()) for k, v in met.items()}
            self.assertTrue(all(map(math.isfinite, values.values())), cls.__name__)


class ReconstructionFreeTest(unittest.TestCase):
    def test_no_renderer_and_gradients(self):
        """Also checks the gradient contract the trainer's audit reads."""
        from as_jepa_lewm.nce import NCELeWM
        x = make_batch()
        for lam in (0., 1.):
            m = build(NCELeWM, lam=lam)
            self.assertIsNone(m.renderer)
            self.assertFalse(any(n.startswith("renderer.") for n, _ in m.named_parameters()))
            m.zero_grad(); torch.manual_seed(5); loss, _ = m.loss(x, step=1); loss.backward()
            pool = sum(p.grad.abs().sum() for n, p in m.named_parameters() if p.grad is not None and n.startswith("pool."))
            prior = sum(p.grad.abs().sum() for n, p in m.named_parameters() if p.grad is not None and n.startswith("predictor."))
            self.assertTrue(torch.isfinite(loss)); self.assertGreater(pool, 0); self.assertGreater(prior, 0)
            self.assertNotIn("renderer", m.audit_expected_gradients())


if __name__ == "__main__":
    unittest.main()
