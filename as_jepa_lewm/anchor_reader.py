"""Anchor-reader AS-JEPA: the log-det estimate of Info is replaced by a denoiser that reads the anchor.

Two players, per window, everything in nats (fixed-variance code, sigma0), T = W - 1 frames (1..T):

    encoder side   J = beta * Entry + beta * sum_t Surprise_t + lam * sum_t Forecast_t - sum_t h(s_t | anchor)
    reader side    R = E || xi_hat(mu_t + sigma0 * xi', anchor) - xi' ||^2        (mean over elements)

  Entry, Surprise_t, Forecast_t   exactly as in decoder_free.py (same tensors, same random draws)
  h(s_t | anchor)                 entropy of the noisy state s_t = mu_t + sigma0 * xi given the anchor. It differs
                                  from Info = I(x_t; s_t | x_A) only by the constant entropy of the noise, so
                                  descending -h is ascending Info. It has no value here, only a gradient.
  xi_hat                          the anchor reader: a small denoiser that sees a noisy state and the anchor and
                                  predicts the injected unit noise. Its optimum recovers xi exactly when the
                                  anchor already determines the state, which is the case of no information
                                  beyond the anchor.

Who minimises what. The reader minimises R and nothing else. The encoder, pool, predictor and action encoder
minimise J and never receive R. The gradient of -h(s_t | anchor) with respect to mu_t is the expected score of
p(s_t | anchor) at the sample (entropy-gradient identity), and Tweedie's formula gives that score from the
optimal denoiser: score = -xi_hat / sigma0. The encoder therefore descends

    ent_t = sg(score_t) . mu_t,        score_t = -xi_hat(s_t, anchor) / sigma0.

The score is detached because each player takes the gradient of its own loss: differentiating ent_t through the
reader would train the reader on the encoder's objective, and differentiating R through mu would train the
encoder to make the reader's task easy. This is the construction of EBReg (arXiv 2611.11379), made conditional
on the anchor. The JEPA prediction targets stay attached as before. One optimizer steps both players:

    loss = beta * Entry + beta * sum_t Surprise_t + lam * sum_t Forecast_t + sum_t ent_t + reader_loss_weight * R

Knobs (in addition to those of decoder_free.py):
  estimator           "denoiser": the two-player objective above (canonical);
                      "nce": zero-parameter control, strictly per window. Each posterior sample s_t is scored
                      against the T encoded means of its own window plus the anchor's mean with the Gaussian
                      logit -||s_t - c||^2 / (2 sigma0^2); the InfoNCE value (at most log(T + 1) nats per frame)
                      replaces Info, with no stop-gradient;
                      "logdet", "logdet_window": DecoderFreeLeWM unchanged (bitwise, no reader is built).
                      The new estimators need conditioning="anchor".
  reader_anchor       "frame": the reader is conditioned on the anchor's state mean (detached) and on features of
                      the anchor image from its own small CNN (canonical: what the anchor determines does not
                      depend on what the encoder kept of it);
                      "state": the anchor's state mean only
  reader_width, reader_depth   width and number of residual blocks of the reader's trunk
  reader_draws        fresh noise draws per frame in R (from their own random stream)
  reader_lr           learning rate of the reader's parameter group (training_optimizer_config)
  reader_loss_weight  factor on R in the total loss. The trainers clip one global gradient norm over all
                      parameters, so an unweighted R would let the reader's gradient shrink the encoder's
                      update. Adam's step is invariant to a constant factor on a group's gradient, so the
                      reader's own update does not depend on this factor. The logged reader_loss is unweighted.
  reader_seed         seed of the reader's initial weights and of its noise stream
  rho                 AR(1) correlation of the posterior noise across consecutive frames (default 0). Each frame
                      keeps marginal noise sigma0; the surprise becomes the exact KL of the AR(1) posterior's
                      conditional q(s_t | s_<t, x) = N(mu_t + rho (s_{t-1} - mu_{t-1}), sigma0^2 (1 - rho^2))
                      against the prior N(m_t, sigma0^2), as in ForecastScoredLeWM and RobustLeWM. The reader and
                      the entropy term are unchanged, because each frame's noisy state is still a draw from its
                      marginal. Needs prior_input="sample", and either lam = 0 or the one-step noiseless forecast
                      (where Forecast_t equals Surprise_t). With "logdet" or "logdet_window" the information term
                      is DecoderFreeLeWM's, computed on the same means: the closed-form reward with this noise
                      and no reader.
  reader_relative     False (default): the reader denoises s_t. True: it denoises s_t - mu_A, and the encoder
                      term becomes sg(score) . (mu_t - mu_A) with mu_A attached. h(s_t | anchor) is unchanged,
                      because mu_A is a function of the anchor; a common shift of every code now receives exactly
                      zero gradient from this term, so an imperfect reader cannot drag the code's origin.
                      Denoiser only (nce is translation invariant already).

Logged in addition to DecoderFreeLeWM's keys: reader_loss (R), reader_mse (error per dimension of xi_hat on the
posterior sample: 0 means the noise is fully recovered, so the anchor determines the state; values toward 1 mean
the reader cannot separate signal from noise), score_rms, and temporal_noise_rho and reader_relative when
they differ from their defaults. For the denoiser the logged info is the pooled log-det
diagnostic and the logged loss contains the surrogate sum_t ent_t, so neither is the trained quantity; for nce
info is the trained InfoNCE value. information_diagnostics stays a no-grad monitor for every estimator.

The reader always runs in float32 with autocast disabled. Its initial weights and its noise come from separate
random streams, so every other tensor and every main-stream draw equals DecoderFreeLeWM's at the same seed.
There is no renderer, no EMA and no augmentation; the only stop-gradient is the one on the score. Planning,
probes and LeWM's rollout use encoder means and prior means exactly as for L2 and never touch the reader.

AnchorReader, encoder_entropy_term, reader_denoising_term and window_nce are plain and model-free, so another
model family can reuse the mechanism (see experiments/anchor_reader_spt.py).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.decoder_free import DecoderFreeLeWM, ACTIVE_NATS
from as_jepa_lewm.forecast_scored import ar1
from vforge.models.as_jepa.codes import entry_nats, surprise_nats

READER_INIT_STREAM = 15485863   # separate random stream for the reader's initial weights
READER_STREAM = 104729          # separate random stream for the reader's training noise (as FORECAST_STREAM)
READER_FRAME_PX = 64            # side of the anchor image the reader's CNN sees
READER_FRAME_FEATURES = 192     # width of the CNN's anchor features
NEW_ESTIMATORS = ("denoiser", "nce")


def reader_stream_seed(reader_seed, step):
    """Seed of the reader's noise stream at a step (same convention as the forecast stream)."""
    return READER_STREAM + 1000003 * int(reader_seed) + int(step)


class AnchorReader(nn.Module):
    """Denoiser xi_hat(s, anchor) for s = mu + sigma0 * xi. Always float32; no dropout, no batch statistics,
    so every window is processed independently of the rest of the batch.

    Conditioning vector: the anchor's state mean, detached and flattened ([B, K * d]); with frame=True also
    features of the anchor image from the reader's own CNN ([B, 192]). Trunk: Linear, `depth` residual MLP
    blocks (LayerNorm, Linear, GELU, Linear), LayerNorm, Linear. The weights are drawn inside a forked CPU
    stream, so building a reader never shifts the caller's random stream."""

    def __init__(self, state_slots, state_width, frame=True, width=512, depth=3, seed=0):
        super().__init__()
        self.state_dim = state_slots * state_width
        self.cond_dim = self.state_dim + (READER_FRAME_FEATURES if frame else 0)
        with torch.random.fork_rng(devices=[]):
            # CPU generator only: torch.manual_seed would also reseed CUDA, which the fork does not restore here
            torch.default_generator.manual_seed(READER_INIT_STREAM + 1000003 * int(seed))
            self.frame = None
            if frame:
                chans = (3, 32, 64, 128, READER_FRAME_FEATURES)
                convs = []
                for c_in, c_out in zip(chans[:-1], chans[1:]):
                    convs += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.GELU()]
                self.frame = nn.Sequential(*convs, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                           nn.Linear(READER_FRAME_FEATURES, READER_FRAME_FEATURES))
            self.inp = nn.Linear(self.state_dim + self.cond_dim, width)
            self.blocks = nn.ModuleList(
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
                for _ in range(depth))
            self.norm = nn.LayerNorm(width)
            self.out = nn.Linear(width, self.state_dim)

    def condition(self, mu_A, anchor=None):
        """mu_A: [B, K, d] anchor state mean; anchor: [B, 3, H, W] anchor image at any resolution (needed with
        frame=True). Returns [B, C]. Nothing upstream of mu_A or the image receives a gradient through it; only
        the reader's CNN does."""
        with torch.autocast(device_type=mu_A.device.type, enabled=False):
            cond = mu_A.detach().float().flatten(1)
            if self.frame is None:
                return cond
            if anchor is None:
                raise ValueError("this reader is conditioned on the anchor image")
            image = anchor.detach().float()
            if image.shape[-2:] != (READER_FRAME_PX, READER_FRAME_PX):
                image = F.interpolate(image, size=(READER_FRAME_PX, READER_FRAME_PX), mode="bilinear",
                                      align_corners=False, antialias=True)
            return torch.cat([cond, self.frame(image)], 1)

    def forward(self, s, cond):
        """s: [B, ..., K, d] noisy states; cond: [B, C] from condition(). Returns xi_hat shaped like s."""
        with torch.autocast(device_type=s.device.type, enabled=False):
            x = s.float().flatten(-2)                                               # [B, ..., D]
            c = cond.float().reshape(len(cond), *([1] * (x.ndim - 2)), -1).expand(*x.shape[:-1], -1)
            h = self.inp(torch.cat([x, c], -1))
            for block in self.blocks:
                h = h + block(h)
            return self.out(self.norm(h)).reshape(s.shape)


def encoder_entropy_term(reader, mu, s, cond, sigma0):
    """Encoder-side term ent = sg(score) . mu, [B, T], with score = -xi_hat(s, anchor) / sigma0.

    mu, s: [B, T, K, d] state means and their noisy samples s = mu + sigma0 * xi; cond: [B, C]. The reader is
    evaluated without a graph, so ent sends a gradient to mu only: the gradient of -h(s | anchor) when the
    reader is optimal. The value of ent is a surrogate with no meaning of its own. Also returns the no-grad
    monitors reader_mse (mean squared error per dimension of xi_hat against the injected noise) and score_rms."""
    with torch.no_grad():
        xi_hat = reader(s.detach(), cond).float()
        score = -xi_hat / sigma0
        xi = (s.detach().float() - mu.detach().float()) / sigma0
        stats = dict(reader_mse=(xi_hat - xi).square().mean(), score_rms=score.square().mean().sqrt())
    return (score * mu.float()).sum((-2, -1)), stats


def reader_denoising_term(reader, mu, cond, sigma0, draws, seed):
    """The reader's own loss per window, [B]: the mean over `draws` fresh noise draws xi', frames and dimensions
    of (reader(mu + sigma0 * xi', anchor) - xi')^2. Its batch mean is the reader loss R.

    mu: [B, T, K, d], detached here, so R sends no gradient to whatever produced the states; cond: [B, C]
    (it carries the gradient into the reader's CNN). The noise comes from a forked stream seeded by `seed`,
    so the caller's random stream is never shifted."""
    clean = mu.detach().float()
    devices = [clean.device.index] if clean.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        xi = torch.randn((len(clean), int(draws), *clean.shape[1:]), device=clean.device, dtype=clean.dtype)
    return (reader(clean[:, None] + sigma0 * xi, cond).float() - xi).square().flatten(1).mean(1)


def window_nce(mu, s, mu_A, sigma0):
    """InfoNCE within one window, [B, T], nats per frame, float64, at most log(T + 1).

    mu, s: [B, T, K, d] (frames 1..T); mu_A: [B, K, d]. Query s_t; candidates: the T means of the same window
    plus the anchor's mean; logit -||s_t - c||^2 / (2 sigma0^2); positive: mu_t. No other window enters."""
    with torch.autocast(device_type=mu.device.type, enabled=False):
        T = mu.shape[1]
        q = s.flatten(2).double()                                                   # [B, T, D]
        c = torch.cat([mu, mu_A[:, None]], 1).flatten(2).double()                   # [B, T + 1, D]
        logits = -(q[:, :, None] - c[:, None]).square().sum(-1) / (2 * sigma0 ** 2)  # [B, T, T + 1]
        positive = logits[:, :, :T].diagonal(dim1=1, dim2=2)
        return positive - torch.logsumexp(logits, 2) + math.log(T + 1)


class AnchorReaderLeWM(DecoderFreeLeWM):
    def __init__(self, estimator="denoiser", reader_anchor="frame", reader_width=512, reader_depth=3,
                 reader_draws=4, reader_lr=5e-4, reader_seed=0, reader_loss_weight=0.01, rho=0.,
                 reader_relative=False, **kwargs):
        new = estimator in NEW_ESTIMATORS
        super().__init__(estimator="logdet" if new else estimator, **kwargs)      # the parent rejects unknown values
        if new and self.conditioning != "anchor":
            raise ValueError("the denoiser and nce estimators need the anchor reference")
        if not 0. <= rho < 1.:
            raise ValueError("rho must lie in [0, 1)")
        if rho and not (self.prior_input == "sample"
                        and (self.lam == 0 or (self.horizon == "one_step" and not self.forecast_noise))):
            raise ValueError("rho needs prior_input='sample', and lam = 0 or the one-step noiseless forecast")
        if reader_relative and estimator != "denoiser":
            raise ValueError("reader_relative applies to the denoiser only")
        self.rho, self.reader_relative = float(rho), bool(reader_relative)
        if reader_anchor not in {"frame", "state"}:
            raise ValueError("unknown reader_anchor")
        for name, value in (("reader_width", reader_width), ("reader_depth", reader_depth),
                            ("reader_draws", reader_draws)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("reader_lr", reader_lr), ("reader_loss_weight", reader_loss_weight)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.estimator, self.reader_anchor = estimator, reader_anchor
        self.reader_draws, self.reader_lr = reader_draws, float(reader_lr)
        self.reader_seed, self.reader_loss_weight = int(reader_seed), float(reader_loss_weight)
        # built last and from its own stream, so all other initial tensors match DecoderFreeLeWM at a seed
        self.reader = (AnchorReader(self.state_slots, self.state_width, frame=reader_anchor == "frame",
                                    width=reader_width, depth=reader_depth, seed=self.reader_seed)
                       if estimator == "denoiser" else None)

    def audit_expected_gradients(self):
        expected = super().audit_expected_gradients()
        if self.reader is not None:
            expected["reader"] = "active"
        return expected

    def training_optimizer_config(self, max_steps):
        """The trainer's default groups and schedule (as_jepa_lewm/train.py, configure_optimizers) for every
        non-reader parameter, plus one group for the reader. The schedule's factor multiplies every group, so
        the ratio of the two learning rates is constant. Without a reader this is the trainer's default path."""
        reader, other = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                (reader if name.startswith("reader.") else other).append(param)
        groups = [dict(params=other, lr=5e-5, weight_decay=1e-3, betas=(.9, .999))]   # no renderer group: none exists
        if reader:
            groups.append(dict(params=reader, lr=self.reader_lr, weight_decay=0., betas=(.9, .95)))
        optimizer = torch.optim.AdamW(groups, foreach=True)
        total = max_steps
        warmup = max(1, int(.01 * total))

        def schedule(step):
            if step < warmup:
                return max(1, step) / warmup
            return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    # ---------------------------------------------------------------- objective
    def window_terms(self, batch, step=0):
        """DecoderFreeLeWM.loss up to the information term, in the same order, so every tensor and every
        main-stream draw equals the parent's at the same seed and step when rho = 0. Also returns the anchor
        frame and the surprise target (mu_t at rho = 0; the AR(1) posterior's conditional mean otherwise)."""
        frames = self.frames(batch)
        info_in = self.encode(dict(pixels=frames, action=batch["action"]))
        flat, act_emb = info_in["emb"], info_in["act_emb"]
        b, w1 = flat.shape[:2]; w = w1 - 1
        mu_all = flat.reshape(b, w1, self.state_slots, self.state_width)
        mu, mu_A = mu_all[:, :w], mu_all[:, w]
        s = mu + self.sigma0 * ar1(torch.randn_like(mu), self.rho)                  # same draws; AR(1) for rho > 0
        inputs = s if self.prior_input == "sample" else mu
        predicted = self.predict(inputs.flatten(2)[:, :-1], act_emb[:, :-1]).reshape_as(mu[:, 1:])
        entry = entry_nats(mu[:, 0], self.sigma0)                                   # [B]
        if self.rho:
            target = mu[:, 1:] + self.rho * (s[:, :-1] - mu[:, :-1])
            variance_kl_per_dimension = .5 * (-self.rho ** 2 - math.log1p(-self.rho ** 2))
            surprise = surprise_nats(target, predicted, self.sigma0) + self.state_dim * variance_kl_per_dimension
        else:
            target = mu[:, 1:]
            surprise = surprise_nats(mu[:, 1:], predicted, self.sigma0)             # [B, T]
        forecast = (self.forecast_term(mu, s, inputs, predicted, act_emb, surprise, step)
                    if self.lam > 0 else torch.zeros_like(surprise))
        return dict(mu=mu, mu_A=mu_A, s=s, inputs=inputs, predicted=predicted, entry=entry, surprise=surprise,
                    forecast=forecast, anchor_frame=frames[:, w], target=target)

    def loss(self, batch, step=0, ramp_steps=0):
        if self.estimator not in NEW_ESTIMATORS and not self.rho:
            return super().loss(batch, step, ramp_steps)
        if ramp_steps:
            raise ValueError("no schedule on the decoder-free objective")
        terms = self.window_terms(batch, step)
        mu, mu_A, s, inputs, predicted = (terms[k] for k in ("mu", "mu_A", "s", "inputs", "predicted"))
        entry, surprise, forecast = terms["entry"], terms["surprise"], terms["forecast"]
        loss = (self.beta * entry.mean() + self.beta * surprise.sum(1).mean()
                + self.lam * forecast.sum(1).mean())
        extra, info = {}, None
        if self.estimator == "denoiser":
            cond = self.reader.condition(mu_A, terms["anchor_frame"])
            state, sample = mu[:, 1:], s[:, 1:]
            if self.reader_relative:            # mu_A attached: a common shift cancels exactly in the encoder term
                state, sample = state - mu_A[:, None], sample - mu_A[:, None]
            ent, extra = encoder_entropy_term(self.reader, state, sample, cond, self.sigma0)
            reader_loss = reader_denoising_term(self.reader, state, cond, self.sigma0, self.reader_draws,
                                                reader_stream_seed(self.reader_seed, step)).mean()
            loss = loss + ent.sum(1).mean() + self.reader_loss_weight * reader_loss
            extra["reader_loss"] = reader_loss.detach()
        elif self.estimator == "nce":
            nce = window_nce(mu[:, 1:], s[:, 1:], mu_A, self.sigma0)                # [B, T]
            loss = loss - nce.sum(1).mean().to(entry.dtype)
            info = nce.mean().detach().float()
        else:                                   # log-det with AR(1) noise: the parent's information term
            trained = self.information(mu[:, 1:], mu_A)
            loss = loss - mu[:, 1:].shape[1] * trained.to(entry.dtype)
            info = trained.detach().float()
        with torch.no_grad():                                                       # as DecoderFreeLeWM.loss
            diagnostics = self.information_diagnostics(mu[:, 1:].detach(), mu_A.detach())
            if info is None:                    # the denoiser's term has no value: log the pooled log-det instead
                info = diagnostics["info_pooled"].float()
            prev, target = inputs[:, :-1].float(), terms["target"].float()
            d_true, d_pred = target - prev, predicted.float() - prev
            calib_slope = (d_true * d_pred).sum() / d_pred.square().sum().clamp_min(1e-12)
            per_dim_rate = (.5 * mu[:, 0].float().square()
                            + (target - predicted.float()).square().sum(1) / (2 * self.sigma0 ** 2)).mean(0)
            metrics = dict(loss=loss, info=info, entry=entry.mean(), surprise=surprise.mean(),
                           forecast=forecast.mean(), rate=(entry + surprise.sum(1)).mean(),
                           feature_std=mu.float().std((0, 1)).mean(),
                           spread_over_sigma0=mu.float().std((0, 1)).mean() / self.sigma0,
                           active_dims=(per_dim_rate > ACTIVE_NATS).sum(), calib_slope=calib_slope)
            metrics.update(diagnostics)
            metrics.update(extra)
            if self.rho:
                metrics["temporal_noise_rho"] = loss.new_tensor(self.rho)
            if self.reader_relative:
                metrics["reader_relative"] = loss.new_tensor(1.)
        return loss, metrics
