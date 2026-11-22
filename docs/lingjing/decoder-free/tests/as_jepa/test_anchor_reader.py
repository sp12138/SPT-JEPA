"""Contracts for the anchor-reader AS-JEPA (as_jepa_lewm/anchor_reader.py). CPU, tiny models.

Run from the repository root:
    PYTHONPATH=.:runtime:<stable-worldmodel checkout> python -B -m unittest tests/as_jepa/test_anchor_reader.py -v
"""
import math
import unittest

import torch

from vforge.ops import OpsConfig, initialize_ops, reset_ops

W, B, ENC = 4, 3, 56
TINY = dict(mode="L2", beta=0.3, sigma0=0.1, window=W, encoder_depth=1, renderer_width=64,
            renderer_depth=1, renderer_heads=1, encoder_size=ENC)
SMALL_READER = dict(reader_width=32, reader_depth=1, reader_draws=2)
VARIANTS = (dict(lam=0.), dict(lam=1.), dict(lam=1., forecast_noise=True), dict(lam=1., horizon="rollout"),
            dict(lam=1., horizon="rollout", forecast_noise=True))
PARENT_KEYS = ("loss", "info", "entry", "surprise", "forecast", "rate", "feature_std", "spread_over_sigma0",
               "active_dims", "calib_slope", "info_pooled", "info_window", "info_pairwise", "participation", "info_dirs")
ENCODER_SIDE = ("encoder.", "pool.", "predictor.", "action_encoder.")


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
    """Zero-initialized layers (identity carry, AdaLN-zero) hide gradients at initialization. The reader is
    registered last, so the draws for every other parameter are the same with and without a reader."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.abs().sum() == 0:
                p.copy_(0.02 * torch.randn(p.shape, generator=g))


def build(**kw):
    from as_jepa_lewm.anchor_reader import AnchorReaderLeWM
    torch.manual_seed(0)
    m = AnchorReaderLeWM(**{**TINY, **SMALL_READER, **kw}); perturb(m); m.train()
    return m


def build_parent(**kw):
    from as_jepa_lewm.decoder_free import DecoderFreeLeWM
    torch.manual_seed(0)
    m = DecoderFreeLeWM(**{**TINY, **kw}); perturb(m); m.train()
    return m


def total(grads, names, prefixes, invert=False):
    return float(sum(g.abs().sum() for g, n in zip(grads, names)
                     if g is not None and n.startswith(prefixes) != invert))


def proportional(a, b, factor):
    """a == factor * b up to float32 rounding, measured against the tensor's largest entry."""
    return bool((a - factor * b).abs().max() <= 1e-5 * a.abs().max())


def trainer_default(model, max_steps):
    """The default path of TrainModule.configure_optimizers in as_jepa_lewm/train.py, copied verbatim."""
    renderer=[];other=[]
    for name,param in model.named_parameters():
        if param.requires_grad:(renderer if name.startswith("renderer.") else other).append(param)
    groups=[dict(params=other,lr=5e-5,weight_decay=1e-3,betas=(.9,.999))]
    if renderer:groups.append(dict(params=renderer,lr=1e-4,weight_decay=0.,betas=(.9,.95)))
    optimizer=torch.optim.AdamW(groups,foreach=True)
    total=max_steps;warmup=max(1,int(.01*total))
    def schedule(step):
        if step<warmup:return max(1,step)/warmup
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=scheduler,interval="step"))


class AnchorReaderModelTest(unittest.TestCase):
    def setUp(self):
        self.batch = make_batch()

    # 1 ---------------------------------------------------------------- parent estimators
    def test_parent_estimators_are_bitwise_decoder_free(self):
        from as_jepa_lewm.anchor_reader import AnchorReaderLeWM
        from as_jepa_lewm.decoder_free import DecoderFreeLeWM
        for estimator in ("logdet", "logdet_window"):
            torch.manual_seed(0); ref = DecoderFreeLeWM(**TINY, estimator=estimator); a = torch.rand(4)
            torch.manual_seed(0); m = AnchorReaderLeWM(**TINY, estimator=estimator); b = torch.rand(4)
            self.assertTrue(torch.equal(a, b), estimator)                      # construction draws the same stream
            self.assertIsNone(m.reader); self.assertIsNone(m.renderer)
            ref_sd, sd = ref.state_dict(), m.state_dict()
            self.assertEqual(list(sd), list(ref_sd))
            for k, v in sd.items():
                self.assertTrue(torch.equal(v, ref_sd[k]), k)
            self.assertEqual(m.audit_expected_gradients(), ref.audit_expected_gradients())
            for kw in (dict(lam=1.), dict(lam=1., horizon="rollout", forecast_noise=True)):
                ref, m = build_parent(estimator=estimator, **kw), build(estimator=estimator, **kw)
                torch.manual_seed(5); loss_r, met_r = ref.loss(self.batch, step=3); a = torch.rand(4)
                torch.manual_seed(5); loss_m, met_m = m.loss(self.batch, step=3); b = torch.rand(4)
                self.assertTrue(torch.equal(loss_r, loss_m), (estimator, kw))
                self.assertTrue(torch.equal(a, b), (estimator, kw))
                self.assertEqual(list(met_m), list(met_r))
                for k in met_r:
                    self.assertTrue(torch.equal(met_r[k], met_m[k]), (estimator, kw, k))

    # 2 ---------------------------------------------------------------- new estimators against the parent
    def test_new_estimators_keep_parent_tensors_rate_terms_and_main_stream(self):
        from as_jepa_lewm.anchor_reader import AnchorReaderLeWM
        from as_jepa_lewm.decoder_free import DecoderFreeLeWM
        torch.manual_seed(0); ref = DecoderFreeLeWM(**TINY); a = torch.rand(4)
        ref_sd = ref.state_dict()
        for estimator in ("denoiser", "nce"):
            torch.manual_seed(0); m = AnchorReaderLeWM(**TINY, **SMALL_READER, estimator=estimator); b = torch.rand(4)
            self.assertTrue(torch.equal(a, b), estimator)                      # the reader's weights use their own stream
            sd = m.state_dict()
            self.assertEqual([k for k in sd if not k.startswith("reader.")], list(ref_sd))
            self.assertEqual(any(k.startswith("reader.") for k in sd), estimator == "denoiser")
            for k, v in ref_sd.items():
                self.assertTrue(torch.equal(v, sd[k]), (estimator, k))
            for kw in VARIANTS:
                ref_v, m = build_parent(**kw), build(estimator=estimator, **kw)
                torch.manual_seed(5); _, met_r = ref_v.loss(self.batch, step=3); r_after = torch.rand(4)
                torch.manual_seed(5); _, met_m = m.loss(self.batch, step=3); m_after = torch.rand(4)
                self.assertTrue(torch.equal(r_after, m_after), (estimator, kw))   # reader noise never shifts the main stream
                for k in ("entry", "surprise", "forecast", "rate", "feature_std", "spread_over_sigma0", "active_dims",
                          "calib_slope", "info_pooled", "info_window", "info_pairwise", "participation", "info_dirs"):
                    self.assertTrue(torch.equal(met_r[k], met_m[k]), (estimator, kw, k))
                # the posterior sample itself: the parent's first lines, replicated
                torch.manual_seed(5); terms = m.window_terms(self.batch, step=3)
                torch.manual_seed(5)
                enc = ref_v.encode(dict(pixels=ref_v.frames(self.batch), action=self.batch["action"]))
                mu = enc["emb"].reshape(B, W + 1, 4, 16)[:, :W]
                s = mu + ref_v.sigma0 * torch.randn_like(mu)
                self.assertTrue(torch.equal(mu, terms["mu"]) and torch.equal(s, terms["s"]), (estimator, kw))

    # 3 ---------------------------------------------------------------- gradient routing
    def test_gradient_routing_between_the_two_players(self):
        from as_jepa_lewm.anchor_reader import encoder_entropy_term, reader_denoising_term, reader_stream_seed
        for reader_anchor in ("frame", "state"):
            m = build(estimator="denoiser", reader_anchor=reader_anchor)
            names = [n for n, _ in m.named_parameters()]; params = [p for _, p in m.named_parameters()]
            torch.manual_seed(5); t = m.window_terms(self.batch, step=1)
            cond = m.reader.condition(t["mu_A"], t["anchor_frame"])
            ent, _ = encoder_entropy_term(m.reader, t["mu"][:, 1:], t["s"][:, 1:], cond, m.sigma0)
            reader_loss = reader_denoising_term(m.reader, t["mu"][:, 1:], cond, m.sigma0, m.reader_draws,
                                                reader_stream_seed(m.reader_seed, 1)).mean()
            rate = m.beta * t["entry"].mean() + m.beta * t["surprise"].sum(1).mean() + m.lam * t["forecast"].sum(1).mean()
            g_reader = torch.autograd.grad(reader_loss, params, retain_graph=True, allow_unused=True)
            g_ent = torch.autograd.grad(ent.sum(1).mean(), params, retain_graph=True, allow_unused=True)
            g_rest = torch.autograd.grad(rate + ent.sum(1).mean(), params, retain_graph=True, allow_unused=True)
            # the reader's loss reaches the reader only
            self.assertEqual(total(g_reader, names, ("reader.",), invert=True), 0., reader_anchor)
            self.assertGreater(total(g_reader, names, ("reader.",)), 0, reader_anchor)
            if reader_anchor == "frame":
                self.assertGreater(total(g_reader, names, ("reader.frame.",)), 0)
            # every other term leaves the reader alone and reaches the encoder side
            self.assertEqual(total(g_rest, names, ("reader.",)), 0., reader_anchor)
            self.assertGreater(total(g_rest, names, ("predictor.", "action_encoder.")), 0, reader_anchor)
            # the entropy term alone trains the encoder and the pool, and nothing else
            self.assertGreater(total(g_ent, names, ("encoder.",)), 0, reader_anchor)
            self.assertGreater(total(g_ent, names, ("pool.",)), 0, reader_anchor)
            self.assertEqual(total(g_ent, names, ("encoder.", "pool."), invert=True), 0., reader_anchor)
            # the class's loss is exactly these pieces, with the reader's loss weighted
            torch.manual_seed(5); loss, met = m.loss(self.batch, step=1)
            expect = rate + ent.sum(1).mean() + m.reader_loss_weight * reader_loss
            self.assertTrue(torch.allclose(loss, expect, rtol=1e-5, atol=1e-5), (loss, expect))
            self.assertTrue(torch.allclose(met["reader_loss"], reader_loss, rtol=1e-6))    # logged unweighted
            m.zero_grad(); loss.backward()
            for n, p, g in zip(names, params, g_reader):
                if n.startswith("reader."):
                    self.assertTrue(proportional(p.grad, g, m.reader_loss_weight), n)

    def test_reader_loss_weight_scales_the_reader_gradient_only(self):
        grads = {}
        for weight in (0.01, 10.):
            m = build(estimator="denoiser", reader_loss_weight=weight)
            torch.manual_seed(5); loss, met = m.loss(self.batch, step=1); loss.backward()
            grads[weight] = ({n: p.grad for n, p in m.named_parameters() if p.grad is not None}, loss, met)
        (small, loss_s, met_s), (large, loss_l, met_l) = grads[0.01], grads[10.]
        self.assertTrue(torch.equal(met_s["reader_loss"], met_l["reader_loss"]))
        self.assertTrue(torch.allclose(loss_l - loss_s, 9.99 * met_s["reader_loss"], rtol=1e-3, atol=1e-4))
        for n, g in small.items():
            if n.startswith("reader."):
                self.assertTrue(proportional(large[n], g, 1000.), n)
            else:
                self.assertTrue(torch.equal(large[n], g), n)

    def test_nce_trains_the_encoder_side_without_a_reader(self):
        m = build(estimator="nce")
        self.assertIsNone(m.reader)
        self.assertEqual([n for n, _ in m.named_parameters() if n.startswith("reader.")], [])
        torch.manual_seed(5); loss, met = m.loss(self.batch, step=1); loss.backward()
        for prefixes in (("encoder.", "pool."), ("predictor.", "action_encoder.")):
            self.assertGreater(sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
                                   if p.grad is not None and n.startswith(prefixes)), 0)
        T = W - 1
        self.assertLessEqual(float(met["info"]), math.log(T + 1) + 1e-6)             # logged in float32
        expect = m.beta * met["entry"] + (m.beta + m.lam) * T * met["surprise"] - T * met["info"]
        self.assertTrue(torch.allclose(loss.float(), expect.float(), rtol=1e-4, atol=1e-4))   # info is the trained value

    # 9 ---------------------------------------------------------------- optimizer
    def test_optimizer_config_matches_the_trainer_default_plus_a_reader_group(self):
        steps = 1000
        for estimator in ("logdet", "logdet_window", "nce", "denoiser"):
            m = build(estimator=estimator, reader_lr=3e-4)
            ours, ref = m.training_optimizer_config(steps), trainer_default(m, steps)
            opt, ref_opt = ours["optimizer"], ref["optimizer"]
            self.assertIs(type(opt), type(ref_opt))
            self.assertEqual(opt.defaults, ref_opt.defaults)
            self.assertEqual(ours["lr_scheduler"]["interval"], "step")
            self.assertEqual(set(ours), set(ref)); self.assertEqual(set(ours["lr_scheduler"]), set(ref["lr_scheduler"]))
            self.assertIs(type(ours["lr_scheduler"]["scheduler"]), type(ref["lr_scheduler"]["scheduler"]))
            schedule, ref_schedule = (c["lr_scheduler"]["scheduler"].lr_lambdas[0] for c in (ours, ref))
            for step in list(range(0, 40)) + [499, 500, 999, 1000, 1500]:
                self.assertEqual(schedule(step), ref_schedule(step), step)
            strip = lambda group: {k: v for k, v in group.items() if k != "params"}
            reader = [p for n, p in m.named_parameters() if n.startswith("reader.")]
            other = [p for n, p in m.named_parameters() if not n.startswith("reader.")]
            self.assertEqual(len(opt.param_groups), 2 if estimator == "denoiser" else 1)
            self.assertEqual([id(p) for p in opt.param_groups[0]["params"]], [id(p) for p in other])
            self.assertEqual(strip(opt.param_groups[0]), strip(ref_opt.param_groups[0]))
            if estimator == "denoiser":
                group = opt.param_groups[1]
                self.assertEqual([id(p) for p in group["params"]], [id(p) for p in reader])
                self.assertGreater(len(reader), 0)
                self.assertEqual((group["initial_lr"], group["weight_decay"], group["betas"]), (3e-4, 0., (.9, .95)))
            else:                                   # no reader: exactly the trainer's default path
                self.assertEqual(len(ref_opt.param_groups), 1)
                self.assertEqual([id(p) for p in ref_opt.param_groups[0]["params"]], [id(p) for p in other])
        try:                                        # with the trainer importable, also against its own code
            from as_jepa_lewm.train import TrainModule
        except ImportError:
            return
        from types import SimpleNamespace
        parent = build_parent()
        real = TrainModule.configure_optimizers(SimpleNamespace(model=parent, cfg=dict(max_steps=steps)))
        ours = build(estimator="logdet").training_optimizer_config(steps)
        strip = lambda group: {k: v for k, v in group.items() if k != "params"}
        self.assertEqual([strip(g) for g in ours["optimizer"].param_groups],
                         [strip(g) for g in real["optimizer"].param_groups])
        for step in range(0, 1200, 7):
            self.assertEqual(ours["lr_scheduler"]["scheduler"].lr_lambdas[0](step),
                             real["lr_scheduler"]["scheduler"].lr_lambdas[0](step))

    # 10 --------------------------------------------------------------- interface
    def test_metrics_reload_audit_and_inference_without_the_reader(self):
        from as_jepa_lewm.anchor_reader import AnchorReaderLeWM
        from as_jepa_lewm.factory import gradient_expectations
        m = build(estimator="denoiser")
        self.assertIsNone(m.renderer)
        self.assertFalse(any(k.startswith("renderer.") for k in m.state_dict()))
        torch.manual_seed(5); _, met = m.loss(self.batch, step=1)
        self.assertEqual(list(met)[:len(PARENT_KEYS)], list(PARENT_KEYS))
        self.assertEqual(set(met) - set(PARENT_KEYS), {"reader_loss", "reader_mse", "score_rms"})
        for k, v in met.items():
            self.assertTrue(math.isfinite(float(v.detach())), k)             # the trainer logs float(v.detach())
        self.assertTrue(torch.equal(met["info"], met["info_pooled"].float()))
        torch.manual_seed(5); _, met_nce = build(estimator="nce").loss(self.batch, step=1)
        self.assertEqual(list(met_nce), list(PARENT_KEYS))
        # audit
        parent = build_parent().audit_expected_gradients()
        self.assertEqual(m.audit_expected_gradients(), {**parent, "reader": "active"})
        self.assertEqual(gradient_expectations(m), {**parent, "reader": "active"})
        self.assertEqual(build(estimator="nce").audit_expected_gradients(), parent)
        # strict reload into a model built from other seeds
        torch.manual_seed(9)
        other = AnchorReaderLeWM(**{**TINY, **SMALL_READER}, estimator="denoiser", reader_seed=7).train()
        self.assertFalse(torch.equal(other.reader.out.weight, m.reader.out.weight))
        other.load_state_dict(m.state_dict(), strict=True)
        torch.manual_seed(5); loss_a, met_a = m.loss(self.batch, step=1)
        torch.manual_seed(5); _, met_b = other.loss(self.batch, step=1)
        self.assertFalse(torch.equal(met_a["reader_loss"], met_b["reader_loss"]))   # reader_seed also seeds its noise stream
        self.assertTrue(torch.equal(met_a["reader_mse"], met_b["reader_mse"]))      # same weights, same posterior sample
        other.reader_seed = m.reader_seed
        torch.manual_seed(5); loss_b, _ = other.loss(self.batch, step=1)
        self.assertTrue(torch.equal(loss_a, loss_b))
        with self.assertRaises(RuntimeError):
            build(estimator="nce").load_state_dict(m.state_dict(), strict=True)
        # inference never touches the reader
        m.eval()
        with torch.no_grad():
            frames = m.frames(self.batch)
            emb = m.encode(dict(pixels=frames, action=self.batch["action"]))
            pred = m.predict(emb["emb"][:, :3], emb["act_emb"][:, :3])
            before = (emb["emb"].clone(), emb["act_emb"].clone(), pred.clone())
            m.reader = None
            emb = m.encode(dict(pixels=frames, action=self.batch["action"]))
            pred = m.predict(emb["emb"][:, :3], emb["act_emb"][:, :3])
        for x, y in zip(before, (emb["emb"], emb["act_emb"], pred)):
            self.assertTrue(torch.equal(x, y))
        with self.assertRaises(RuntimeError):
            m.render_states(torch.zeros(1, 64), torch.zeros(1, 3, 64, 64))

    def test_reader_seed_changes_the_reader_only(self):
        a, b = build(estimator="denoiser", reader_seed=0), build(estimator="denoiser", reader_seed=1)
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if n.startswith("reader.") and p.ndim > 1:
                self.assertFalse(torch.equal(p, q), n)
            elif not n.startswith("reader."):
                self.assertTrue(torch.equal(p, q), n)

    def test_argument_validation(self):
        for kw in (dict(reader_anchor="pixels"), dict(reader_width=0), dict(reader_depth=0), dict(reader_draws=0),
                   dict(reader_draws=1.5), dict(reader_lr=0.), dict(reader_lr=float("inf")), dict(reader_loss_weight=0.),
                   dict(estimator="score"), dict(estimator="denoiser", conditioning="batch"),
                   dict(estimator="nce", conditioning="batch"), dict(estimator="logdet_window", conditioning="batch")):
            with self.assertRaises(ValueError, msg=kw):
                build(**kw)
        for estimator in ("denoiser", "nce"):
            with self.assertRaises(ValueError):
                build(estimator=estimator).loss(self.batch, step=1, ramp_steps=10)

    # 11 --------------------------------------------------------------- correlated posterior noise (rho)
    def test_rho_argument_validation(self):
        for kw in (dict(rho=1.), dict(rho=-.1), dict(estimator="logdet", rho=.8, prior_input="mean"),
                   dict(estimator="logdet_window", rho=.8, lam=1., horizon="rollout"), dict(rho=.8, prior_input="mean"),
                   dict(rho=.8, lam=1., forecast_noise=True), dict(rho=.8, lam=1., horizon="rollout"),
                   dict(estimator="nce", reader_relative=True), dict(estimator="logdet", reader_relative=True)):
            with self.assertRaises(ValueError, msg=kw):
                build(**kw)
        for kw in (dict(rho=.8, lam=0.), dict(rho=.8, lam=0., horizon="rollout"), dict(rho=.8, lam=1.),
                   dict(estimator="nce", rho=.8, lam=1.), dict(rho=.8, lam=1., reader_relative=True),
                   dict(estimator="logdet", rho=.8, lam=1.), dict(estimator="logdet_window", rho=.8, lam=0.)):
            build(**kw)                                     # accepted

    def test_rho_zero_is_bitwise_the_default(self):
        for estimator in ("denoiser", "nce"):
            for kw in VARIANTS:
                a, b = build(estimator=estimator, **kw), build(estimator=estimator, rho=0., **kw)
                torch.manual_seed(5); loss_a, met_a = a.loss(self.batch, step=3); after_a = torch.rand(4)
                torch.manual_seed(5); loss_b, met_b = b.loss(self.batch, step=3); after_b = torch.rand(4)
                self.assertTrue(torch.equal(loss_a, loss_b) and torch.equal(after_a, after_b), (estimator, kw))
                self.assertEqual(list(met_a), list(met_b))

    def test_rho_gives_the_ar1_posterior_and_its_exact_kl(self):
        from as_jepa_lewm.forecast_scored import ar1
        from vforge.models.as_jepa.codes import surprise_nats
        rho = .8
        # the noise: unit marginal variance, correlation rho between consecutive frames
        noise = ar1(torch.randn(20000, 6, 2, generator=torch.Generator().manual_seed(3)), rho)
        self.assertAlmostEqual(float(noise.var()), 1., delta=.03)
        corr = float((noise[:, 1:] * noise[:, :-1]).mean() / noise.var())
        self.assertAlmostEqual(corr, rho, delta=.02)
        for estimator in ("denoiser", "nce"):
            for lam in (0., 1.):
                m, ref = build(estimator=estimator, rho=rho, lam=lam), build(estimator=estimator, lam=lam)
                torch.manual_seed(5); t = m.window_terms(self.batch, step=3); after = torch.rand(4)
                torch.manual_seed(5); t0 = ref.window_terms(self.batch, step=3); after0 = torch.rand(4)
                self.assertTrue(torch.equal(after, after0), (estimator, lam))    # same number of main-stream draws
                self.assertTrue(torch.equal(t["mu"], t0["mu"]) and torch.equal(t["entry"], t0["entry"]))
                xi0 = (t0["s"] - t0["mu"]) / m.sigma0
                expect_s = t["mu"] + m.sigma0 * ar1(xi0.clone(), rho)
                self.assertTrue(torch.allclose(t["s"], expect_s, atol=1e-6), (estimator, lam))
                target = t["mu"][:, 1:] + rho * (t["s"][:, :-1] - t["mu"][:, :-1])
                self.assertTrue(torch.equal(t["target"], target))
                const = m.state_dim * .5 * (-rho ** 2 - math.log1p(-rho ** 2))
                expect = surprise_nats(target, t["predicted"], m.sigma0) + const
                self.assertTrue(torch.equal(t["surprise"], expect), (estimator, lam))
                if lam:                                     # one-step noiseless forecast: Forecast_t = Surprise_t
                    self.assertTrue(torch.equal(t["forecast"], t["surprise"]))
                torch.manual_seed(5); loss, met = m.loss(self.batch, step=3)
                self.assertTrue(math.isfinite(float(loss)))
                self.assertAlmostEqual(float(met["temporal_noise_rho"]), rho, places=6)
                loss.backward()
                self.assertGreater(sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
                                       if p.grad is not None and n.startswith(("encoder.", "pool."))), 0)

    def test_rho_with_the_log_det_estimators_is_the_parent_reward_on_the_ar1_terms(self):
        rho = .8
        for estimator in ("logdet", "logdet_window"):
            for lam in (0., 1.):
                m = build(estimator=estimator, rho=rho, lam=lam)
                ref = build(estimator="denoiser", rho=rho, lam=lam)
                self.assertIsNone(m.reader)
                torch.manual_seed(5); t = m.window_terms(self.batch, step=3)
                torch.manual_seed(5); t0 = ref.window_terms(self.batch, step=3)
                for k in ("mu", "s", "entry", "surprise", "forecast"):      # the same AR(1) terms as the reader arms
                    self.assertTrue(torch.equal(t[k], t0[k]), (estimator, lam, k))
                torch.manual_seed(5); loss, met = m.loss(self.batch, step=3)
                info = m.information(t["mu"][:, 1:], t["mu_A"])
                expect = (m.beta * t["entry"].mean() + (m.beta + lam) * t["surprise"].sum(1).mean()
                          - (W - 1) * info.to(t["entry"].dtype))
                self.assertTrue(torch.allclose(loss, expect, rtol=1e-6, atol=1e-6), (estimator, lam))
                self.assertTrue(torch.equal(met["info"], info.detach().float()))
                key = "info_window" if estimator == "logdet_window" else "info_pooled"
                self.assertAlmostEqual(float(met["info"]), float(met[key]), places=5)
                self.assertAlmostEqual(float(met["temporal_noise_rho"]), rho, places=6)
                self.assertFalse(any(k.startswith("reader") for k in met))
                loss.backward()
                for prefix in ENCODER_SIDE:
                    self.assertGreater(sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
                                           if p.grad is not None and n.startswith(prefix)), 0, (estimator, lam, prefix))

    # 12 --------------------------------------------------------------- anchor-relative coordinates
    def test_reader_relative_gives_a_common_shift_exactly_zero_gradient(self):
        """With beta = lam = 0 the loss is the encoder term plus the reader's (detached) loss, so its gradient
        with respect to a constant c added to every code (frames and anchor) is the entropy term's alone."""
        grads = {}
        for relative in (False, True):
            m = build(estimator="denoiser", beta=0., lam=0., reader_relative=relative)
            c = torch.zeros(m.state_dim, requires_grad=True)
            encode = m.encode

            def shifted(info, encode=encode, c=c):
                out = encode(info); out["emb"] = out["emb"] + c
                return out
            m.encode = shifted
            torch.manual_seed(5); loss, met = m.loss(self.batch, step=1)
            grads[relative] = torch.autograd.grad(loss, c)[0]
            self.assertEqual("reader_relative" in met, relative)
        absolute = float(grads[False].abs().max())
        self.assertGreater(absolute, 0)
        self.assertLessEqual(float(grads[True].abs().max()), 1e-5 * absolute)

    def test_reader_relative_denoises_the_change_from_the_anchor(self):
        from as_jepa_lewm.anchor_reader import encoder_entropy_term, reader_denoising_term, reader_stream_seed
        m = build(estimator="denoiser", reader_relative=True)
        torch.manual_seed(5); t = m.window_terms(self.batch, step=1)
        cond = m.reader.condition(t["mu_A"], t["anchor_frame"])
        state, sample = t["mu"][:, 1:] - t["mu_A"][:, None], t["s"][:, 1:] - t["mu_A"][:, None]
        ent, stats = encoder_entropy_term(m.reader, state, sample, cond, m.sigma0)
        reader_loss = reader_denoising_term(m.reader, state, cond, m.sigma0, m.reader_draws,
                                            reader_stream_seed(m.reader_seed, 1)).mean()
        rate = m.beta * t["entry"].mean() + m.beta * t["surprise"].sum(1).mean() + m.lam * t["forecast"].sum(1).mean()
        torch.manual_seed(5); loss, met = m.loss(self.batch, step=1)
        self.assertTrue(torch.allclose(loss, rate + ent.sum(1).mean() + m.reader_loss_weight * reader_loss,
                                       rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.equal(met["reader_mse"], stats["reader_mse"]))   # the noise is the same either way


class AnchorReaderMechanismTest(unittest.TestCase):
    """The plain functions, without the encoder: AnchorReader, encoder_entropy_term, reader_denoising_term,
    window_nce."""
    SIGMA0 = 0.1

    # 4 ---------------------------------------------------------------- direction
    def test_encoder_term_is_the_gradient_of_minus_conditional_entropy(self):
        """States mu ~ N(m_A, a^2 I) given the anchor, with the exact Gaussian denoiser. The entropy of s given
        the anchor is 0.5 D log(2 pi e (theta^2 a^2 + sigma0^2)) for a scale theta on mu - m_A, so the term's
        gradient at theta = 1 must be -D a^2 / (a^2 + sigma0^2): negative, descent increases the spread."""
        from as_jepa_lewm.anchor_reader import encoder_entropy_term
        sigma0, n, T, K, d = self.SIGMA0, 4096, 3, 4, 16
        D = K * d
        for a in (0.03, 0.1, 0.5):
            g = torch.Generator().manual_seed(11)
            m_A = torch.randn(n, K, d, generator=g)
            e = torch.randn(n, T, K, d, generator=g); xi = torch.randn(n, T, K, d, generator=g)
            theta = torch.ones((), requires_grad=True)
            mu = m_A[:, None] + theta * a * e
            s = mu + sigma0 * xi
            exact = lambda s, cond: sigma0 * (s - cond.reshape(n, 1, K, d)) / (a ** 2 + sigma0 ** 2)
            ent, stats = encoder_entropy_term(exact, mu, s, m_A.flatten(1), sigma0)
            self.assertEqual(tuple(ent.shape), (n, T))
            grad = float(torch.autograd.grad(ent.mean(), theta)[0])
            expect = -D * a ** 2 / (a ** 2 + sigma0 ** 2)          # d/dtheta of -0.5 D log(theta^2 a^2 + sigma0^2)
            self.assertLess(grad, 0)
            self.assertAlmostEqual(grad / expect, 1., delta=0.03, msg=a)
            # the optimal denoiser's error is the share of the variance the anchor leaves open
            self.assertAlmostEqual(float(stats["reader_mse"]), a ** 2 / (a ** 2 + sigma0 ** 2), delta=0.01, msg=a)
            self.assertAlmostEqual(float(stats["score_rms"]), (a ** 2 + sigma0 ** 2) ** -0.5, delta=0.02 / sigma0, msg=a)

    # 5 ---------------------------------------------------------------- nothing for what the anchor determines
    def test_no_reward_for_states_the_anchor_determines(self):
        from as_jepa_lewm.anchor_reader import encoder_entropy_term
        sigma0, n, T, K, d = self.SIGMA0, 16384, 3, 4, 16
        g = torch.Generator().manual_seed(12)
        m_A = torch.randn(n, K, d, generator=g); xi = torch.randn(n, T, K, d, generator=g)
        determined = lambda anchor: 0.3 * torch.tanh(anchor)                 # the state is a function of the anchor
        theta = torch.ones((), requires_grad=True)
        mu = (theta * determined(m_A))[:, None].expand(n, T, K, d)
        s = mu + sigma0 * xi
        exact = lambda s, cond: (s - determined(cond.reshape(n, 1, K, d))) / sigma0
        ent, stats = encoder_entropy_term(exact, mu, s, m_A.flatten(1), sigma0)
        grad = float(torch.autograd.grad(ent.mean(), theta)[0])
        # Monte Carlo error only (standard deviation about 0.07); an undetermined state of the same spread
        # would give about -K d * 0.035 / (0.035 + sigma0^2), near -50
        self.assertLess(abs(grad), 0.5)
        self.assertLess(float(stats["reader_mse"]), 1e-8)

    # 6 ---------------------------------------------------------------- mechanism
    def run_vector_problem(self, anchor_visible, steps=300, seed=0):
        """A linear encoder on frames x_t = (z_t, a): z_t is fresh every frame (the anchor says nothing about
        it), a is shared by the window's frames and its anchor (appearance). The reader's trunk is the real
        module; it gets the anchor's state mean and, in place of its CNN's features, the anchor vector itself.
        The price is the entry code length 0.5 ||mu||^2 on every frame. Returns the spread per unit factor of
        the variation and of the appearance, in units of sigma0, after every step."""
        from as_jepa_lewm.anchor_reader import (AnchorReader, encoder_entropy_term, reader_denoising_term,
                                                READER_FRAME_FEATURES)
        sigma0, n, T, p, q, price = self.SIGMA0, 256, 3, 4, 4, 1.
        D = p + q
        g = torch.Generator().manual_seed(seed)
        reader = AnchorReader(2, D // 2, frame=True, width=64, depth=2, seed=seed)
        weight = torch.nn.Parameter(0.01 * sigma0 * torch.eye(D))               # the encoder: mu = weight x
        optimizer = torch.optim.Adam([dict(params=[weight], lr=1e-3),
                                      dict(params=list(reader.parameters()), lr=1e-2, betas=(.9, .95))])
        history = []
        for step in range(steps):
            z = torch.randn(n, T, p, generator=g); a = torch.randn(n, 1, q, generator=g).expand(n, T, q)
            x = torch.cat([z, a], -1); x_A = torch.cat([torch.randn(n, p, generator=g), a[:, 0]], -1)
            mu = (x @ weight.T).reshape(n, T, 2, D // 2); mu_A = (x_A @ weight.T).reshape(n, 2, D // 2)
            s = mu + sigma0 * torch.randn(mu.shape, generator=g)
            features = torch.cat([x_A, torch.zeros(n, READER_FRAME_FEATURES - D)], 1)
            cond = torch.cat([mu_A.detach().flatten(1), features], 1)
            if not anchor_visible:
                cond = torch.zeros_like(cond)
            ent, _ = encoder_entropy_term(reader, mu, s, cond, sigma0)
            reader_loss = reader_denoising_term(reader, mu, cond, sigma0, 4, 1000 + step).mean()
            loss = ent.sum(1).mean() + price * 0.5 * mu.square().sum((1, 2, 3)).mean() + 0.01 * reader_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            columns = weight.detach().norm(dim=0) / sigma0
            history.append((float(columns[:p].mean()), float(columns[p:].mean())))
        return history

    def test_variation_grows_and_appearance_does_not(self):
        history = self.run_vector_problem(anchor_visible=True)
        variation, appearance = history[-1]
        self.assertAlmostEqual(history[0][0], 0.01, delta=0.02)                  # starts at 0.01 sigma0
        self.assertGreater(variation, 3.)                                        # observed about 6 sigma0
        self.assertLess(max(a for _, a in history), 0.6)                         # observed about 0.25 sigma0
        self.assertGreater(variation, 10 * appearance)
        # control: a reader that cannot see the anchor rewards the appearance like any other variation
        _, blind = self.run_vector_problem(anchor_visible=False)[-1]
        self.assertGreater(blind, 3.)

    # 8 ---------------------------------------------------------------- separability
    def test_denoiser_terms_are_separable_across_windows(self):
        from as_jepa_lewm.anchor_reader import AnchorReader, encoder_entropy_term, reader_denoising_term
        sigma0, n, T, K, d = self.SIGMA0, 4, 3, 4, 16
        for frame in (True, False):
            reader = AnchorReader(K, d, frame=frame, width=32, depth=2, seed=3)
            g = torch.Generator().manual_seed(13)
            mu = 0.2 * torch.randn(n, T, K, d, generator=g); s = mu + sigma0 * torch.randn(n, T, K, d, generator=g)
            mu_A = 0.2 * torch.randn(n, K, d, generator=g); image = torch.randn(n, 3, 56, 56, generator=g)

            def terms(mu, s, mu_A, image):
                cond = reader.condition(mu_A, image if frame else None)
                ent, _ = encoder_entropy_term(reader, mu, s, cond, sigma0)
                return ent, reader_denoising_term(reader, mu, cond, sigma0, 3, seed=17)
            ent, err = terms(mu, s, mu_A, image)
            self.assertEqual((tuple(ent.shape), tuple(err.shape)), ((n, T), (n,)))
            other = [x.clone() for x in (mu, s, mu_A, image)]
            for x in other:                                                      # change every window but the first
                x[1:] = 3 * torch.randn(x[1:].shape, generator=g)
            ent2, err2 = terms(*other)
            self.assertTrue(torch.equal(ent[0], ent2[0]) and torch.equal(err[0], err2[0]), frame)
            self.assertFalse(torch.equal(ent[1], ent2[1]) or torch.equal(err[1], err2[1]), frame)

    def test_reader_noise_stream_is_forked_and_seeded(self):
        from as_jepa_lewm.anchor_reader import AnchorReader, reader_denoising_term
        torch.manual_seed(3); a = torch.rand(4)
        torch.manual_seed(3); reader = AnchorReader(4, 16, frame=False, width=32, depth=1, seed=0)
        mu = torch.zeros(2, 3, 4, 16); cond = reader.condition(torch.zeros(2, 4, 16))
        first = reader_denoising_term(reader, mu, cond, self.SIGMA0, 2, seed=5); b = torch.rand(4)
        self.assertTrue(torch.equal(a, b))                                       # neither building nor noise shifts the stream
        self.assertTrue(torch.equal(first, reader_denoising_term(reader, mu, cond, self.SIGMA0, 2, seed=5)))
        self.assertFalse(torch.equal(first, reader_denoising_term(reader, mu, cond, self.SIGMA0, 2, seed=6)))

    def test_reader_runs_in_float32_under_autocast(self):
        from as_jepa_lewm.anchor_reader import AnchorReader
        reader = AnchorReader(4, 16, frame=True, width=32, depth=1, seed=0)
        g = torch.Generator().manual_seed(14)
        s = torch.randn(2, 3, 4, 16, generator=g); mu_A = torch.randn(2, 4, 16, generator=g)
        image = torch.randn(2, 3, 224, 224, generator=g)
        with torch.no_grad():
            ref = reader(s, reader.condition(mu_A, image))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                cond = reader.condition(mu_A, image)
                out = reader(s, cond)
        self.assertEqual((cond.dtype, out.dtype), (torch.float32, torch.float32))
        self.assertEqual(tuple(cond.shape), (2, 64 + 192))
        self.assertTrue(torch.equal(out, ref))
        with self.assertRaises(ValueError):
            reader.condition(mu_A)                                               # this reader needs the anchor image

    # 7 ---------------------------------------------------------------- nce
    def test_window_nce_bound_zero_and_per_window(self):
        from as_jepa_lewm.anchor_reader import window_nce
        sigma0, n, T, K, d = self.SIGMA0, 5, 3, 4, 16
        g = torch.Generator().manual_seed(15)
        for spread in (0.01, 0.1, 1.):
            mu = spread * torch.randn(n, T, K, d, generator=g); mu_A = spread * torch.randn(n, K, d, generator=g)
            s = mu + sigma0 * torch.randn(n, T, K, d, generator=g)
            value = window_nce(mu, s, mu_A, sigma0)
            self.assertEqual((tuple(value.shape), value.dtype), ((n, T), torch.float64))
            self.assertTrue(bool((value <= math.log(T + 1)).all()), spread)
        self.assertAlmostEqual(float(value.mean()), math.log(T + 1), delta=1e-6)   # well separated means reach the bound
        # all means coincide: every candidate has the same logit
        same = torch.randn(n, 1, K, d, generator=g).expand(n, T, K, d)
        collapsed = window_nce(same, same + sigma0 * torch.randn(n, T, K, d, generator=g), same[:, 0], sigma0)
        self.assertLess(float(collapsed.abs().max()), 1e-9)
        # strictly per window: other windows do not enter (overlapping means, so the values are below the bound)
        mu = 0.05 * torch.randn(n, T, K, d, generator=g); mu_A = 0.05 * torch.randn(n, K, d, generator=g)
        s = mu + sigma0 * torch.randn(n, T, K, d, generator=g)
        value = window_nce(mu, s, mu_A, sigma0)
        other = [x.clone() for x in (mu, s, mu_A)]
        for x in other:
            x[1:] = torch.randn(x[1:].shape, generator=g)
        value2 = window_nce(*other, sigma0)
        self.assertTrue(torch.equal(value[0], value2[0]))
        self.assertFalse(torch.equal(value[1], value2[1]))


if __name__ == "__main__":
    unittest.main()
