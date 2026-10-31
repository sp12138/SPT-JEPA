"""Contracts for the decoder-free AS-JEPA (as_jepa_lewm/decoder_free.py). 14 tests, CPU, tiny models.

Run from the repository root:
    PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_decoder_free.py -v
"""
import math
import unittest

import torch

from vforge.ops import OpsConfig, initialize_ops, reset_ops

W, B, ENC = 4, 3, 56
TINY = dict(mode="L2", beta=0.3, sigma0=0.1, window=W, encoder_depth=1, renderer_width=64,
            renderer_depth=1, renderer_heads=1, encoder_size=ENC)
VARIANTS = (dict(lam=0.), dict(lam=1.), dict(lam=1., forecast_noise=True), dict(lam=1., horizon="rollout"),
            dict(lam=1., horizon="rollout", forecast_noise=True))


def setUpModule():
    reset_ops()
    initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=torch.device("cpu"), parallel_state=None)


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(pixels=torch.randint(0, 256, (B, W, 3, ENC, ENC), generator=g).float(),
                anchor224=torch.randint(0, 256, (B, 3, ENC, ENC), generator=g).float(),
                render=torch.randint(0, 256, (B, W, 3, 64, 64), generator=g).float(),
                anchor=torch.randint(0, 256, (B, 3, 64, 64), generator=g).float(),
                action=torch.randn(B, W, 10, generator=g),
                episode=torch.arange(B), rows=torch.arange(B * W).reshape(B, W) * 7 + 3,
                anchor_row=torch.arange(B) * 11 + 1)


def perturb(model):
    """Zero-initialized layers (identity carry, AdaLN-zero) hide gradients at initialization."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.abs().sum() == 0:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))


def build(**kw):
    from as_jepa_lewm.decoder_free import DecoderFreeLeWM
    torch.manual_seed(0)
    m = DecoderFreeLeWM(**{**TINY, **kw}); perturb(m); m.train()
    return m


def grad_total(model, batch, prefixes):
    model.zero_grad()
    torch.manual_seed(5)
    loss, _ = model.loss(batch, step=1)
    loss.backward()
    return float(sum(p.grad.abs().sum() for n, p in model.named_parameters()
                     if p.grad is not None and n.startswith(prefixes)))


class DecoderFreeTest(unittest.TestCase):
    def setUp(self):
        self.batch = make_batch()

    def test_no_renderer_and_matching_initialization(self):
        from as_jepa_lewm.model import ASJepaLeWM
        from as_jepa_lewm.decoder_free import DecoderFreeLeWM
        torch.manual_seed(0); ref = ASJepaLeWM(**TINY)
        torch.manual_seed(0); m = DecoderFreeLeWM(**TINY)
        self.assertIsNone(m.renderer)
        self.assertNotIn("renderer", m.audit_expected_gradients())
        ref_sd = ref.state_dict()
        for k, v in m.state_dict().items():                  # every non-renderer tensor equals L2's at the same seed
            self.assertTrue(torch.equal(v, ref_sd[k]), k)

    def test_objective_weights(self):
        for lam in (0., 1.):
            m = build(lam=lam)
            torch.manual_seed(5); loss, met = m.loss(self.batch, step=1)
            T = W - 1
            expect = (m.beta * met["entry"] + (m.beta + lam) * T * met["surprise"] - T * met["info"])
            self.assertTrue(torch.allclose(loss.float(), expect.float(), rtol=1e-4, atol=1e-4), (lam, loss, expect))

    def test_anchor_reference_cancels_episode_constants(self):
        g = torch.Generator().manual_seed(2)
        mu = torch.randn(64, 5, 4, 16, generator=g); mu_A = torch.randn(64, 4, 16, generator=g)
        offset = 3 * torch.randn(64, 1, 4, 16, generator=g)          # a per-episode constant (appearance)
        for estimator in ("logdet", "logdet_window"):
            m = build(estimator=estimator)
            i0 = m.information(mu, mu_A); i1 = m.information(mu + offset, mu_A + offset[:, 0])
            self.assertAlmostEqual(float(i0), float(i1), places=4, msg=estimator)   # float32 inputs
        mb = build(conditioning="batch")
        self.assertGreater(float(mb.information(mu + offset, mu_A + offset[:, 0])) - float(mb.information(mu, mu_A)), 1.0)

    def test_per_window_estimator_resists_episode_keyed_rotation(self):
        """Rotating each episode's change by an episode-level key inflates the pooled log-det (the Jensen gap
        the encoder could exploit) but leaves the per-window estimate unchanged."""
        g = torch.Generator().manual_seed(4)
        n, T, D = 256, 7, 16
        base = torch.zeros(n, T, D); base[..., :2] = torch.randn(n, T, 2, generator=g)   # change lives in 2 dimensions
        mu_A = torch.zeros(n, D)
        rot = torch.linalg.qr(torch.randn(n, D, D, generator=g))[0]                     # one rotation per episode
        rotated = torch.einsum("btd,bed->bte", base, rot)
        shape = lambda x: x.reshape(n, T, 4, 4)
        pooled, window = build(estimator="logdet"), build(estimator="logdet_window")
        a = shape(base); b = shape(rotated); A = mu_A.reshape(n, 4, 4)
        self.assertGreater(float(pooled.information(b, A)) - float(pooled.information(a, A)), 5.0)
        self.assertAlmostEqual(float(window.information(b, A)), float(window.information(a, A)), places=4)

    def test_pooled_logdet_and_pairwise_agree_for_gaussian_states(self):
        m = build()
        g = torch.Generator().manual_seed(3)
        n, D = 200000, 4
        A = torch.randn(D, D, generator=g, dtype=torch.float64) * 0.15
        mu = (torch.randn(n, 1, D, generator=g, dtype=torch.float64) @ A.T).reshape(n, 1, 1, D)
        mu_A = (torch.randn(n, D, generator=g, dtype=torch.float64) @ A.T).reshape(n, 1, D)
        diag = m.information_diagnostics(mu, mu_A)
        # for Gaussian states the pairwise value equals the log-det minus (D/2) log(e/2) (population, small spread)
        self.assertAlmostEqual(float(diag["info_pairwise"]), float(diag["info_pooled"]) - 0.5 * D * math.log(math.e / 2),
                               delta=0.03)

    def test_forecast_variants(self):
        x = self.batch
        m1 = build(lam=1., horizon="one_step", forecast_noise=False)
        torch.manual_seed(5); _, a = m1.loss(x, step=1)
        self.assertTrue(torch.allclose(a["forecast"], a["surprise"]))   # decoder N(m, sigma0^2) is the surprise
        mr = build(lam=1., horizon="rollout", forecast_noise=False)
        mr.load_state_dict(m1.state_dict()); mr.eval()                     # no predictor dropout in this comparison
        # the open-loop forecast's first step equals the one-step forecast
        torch.manual_seed(5)
        enc = mr.encode(dict(pixels=mr.frames(x), action=x["action"]))
        flat, act = enc["emb"], enc["act_emb"]
        mu = flat.reshape(B, W + 1, 4, 16)[:, :W]
        s = mu + mr.sigma0 * torch.randn_like(mu)
        pred = mr.predict(s.flatten(2)[:, :-1], act[:, :-1]).reshape_as(mu[:, 1:])
        from vforge.models.as_jepa.codes import surprise_nats
        sur = surprise_nats(mu[:, 1:], pred, mr.sigma0)
        f = mr.forecast_term(mu, s, s, pred, act, sur, step=1)
        self.assertTrue(torch.allclose(f[:, 0], sur[:, 0], rtol=1e-5, atol=1e-5))
        for kw in VARIANTS[2:]:
            _, met = build(**kw).loss(x, step=1)
            self.assertTrue(torch.isfinite(met["forecast"]), kw)

    def test_main_random_stream_is_paired(self):
        base = build(lam=1.)
        for kw in VARIANTS[2:]:
            m = build(**kw); m.load_state_dict(base.state_dict())
            torch.manual_seed(5); base.loss(self.batch, step=3); a = torch.rand(4)
            torch.manual_seed(5); m.loss(self.batch, step=3); b = torch.rand(4)
            self.assertTrue(torch.equal(a, b), kw)

    def test_gradient_routing_in_every_forecast_variant(self):
        for kw in VARIANTS:
            m = build(**kw)
            self.assertGreater(grad_total(m, self.batch, ("encoder.", "pool.")), 0, kw)
            self.assertGreater(grad_total(m, self.batch, ("predictor.", "action_encoder.")), 0, kw)

    def test_anchor_enters_only_through_the_information_term(self):
        for conditioning, expect_nonzero in (("anchor", True), ("batch", False)):
            m = build(conditioning=conditioning)
            anchor = self.batch["anchor224"].clone().requires_grad_(True)
            torch.manual_seed(5); loss, _ = m.loss(dict(self.batch, anchor224=anchor), step=1); loss.backward()
            g = anchor.grad.abs().sum().item() if anchor.grad is not None else 0.
            self.assertEqual(g > 0, expect_nonzero, conditioning)

    def test_domain_transforms_match_l2_for_window_and_anchor(self):
        from as_jepa_lewm.model import ASJepaLeWM
        for domain in ("color", "dot"):
            torch.manual_seed(0); ref = ASJepaLeWM(**{**TINY, "domain": domain})
            m = build(domain=domain)
            frames = m.frames(self.batch)
            self.assertTrue(torch.allclose(frames[:, :W], ref.prepare(self.batch)[0]), domain)
            # the anchor through L2's own transform: present it to prepare as a one-frame window
            as_window = dict(self.batch, pixels=self.batch["anchor224"][:, None], rows=self.batch["anchor_row"][:, None])
            self.assertTrue(torch.allclose(frames[:, W], ref.prepare(as_window)[0][:, 0]), domain)

    def test_prior_input_and_estimator_variants(self):
        x = self.batch
        for kw in (dict(prior_input="sample"), dict(prior_input="mean"), dict(estimator="logdet_window")):
            _, met = build(**kw).loss(x, step=1)
            self.assertTrue(torch.isfinite(met["loss"]), kw)
        with self.assertRaises(ValueError):
            build(estimator="logdet_window", conditioning="batch")

    def test_metrics_and_planning_interface(self):
        m = build()
        _, met = m.loss(self.batch, step=1)
        for k in ("loss", "info", "entry", "surprise", "forecast", "rate", "feature_std", "spread_over_sigma0",
                  "active_dims", "calib_slope", "info_pooled", "info_window", "info_pairwise", "participation", "info_dirs"):
            self.assertIn(k, met)
            self.assertTrue(math.isfinite(float(met[k].detach())), k)
        with torch.no_grad():   # LeWM's rollout appends predict(...)[:, -1]; the planner reads these means
            self.assertEqual(tuple(m.predict(torch.randn(2, 3, 64), torch.randn(2, 3, 64)).shape), (2, 3, 64))
        with self.assertRaises(RuntimeError):
            m.render_states(torch.zeros(1, 64), torch.zeros(1, 3, 64, 64))

    def test_no_schedule(self):
        with self.assertRaises(ValueError):
            build().loss(self.batch, step=1, ramp_steps=10)

    def test_info_is_the_trained_estimate(self):
        for estimator, key in (("logdet", "info_pooled"), ("logdet_window", "info_window")):
            _, met = build(estimator=estimator).loss(self.batch, step=1)
            self.assertAlmostEqual(float(met["info"]), float(met[key]), places=5, msg=estimator)


if __name__ == "__main__":
    unittest.main()
