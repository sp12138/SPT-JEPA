"""Decoder-free AS-JEPA: the renderer is replaced by the state's information beyond the anchor.

Objective, per window, everything in nats (fixed-variance code, sigma0):

    J = beta * Entry + beta * sum_t Surprise_t + lam * sum_t Forecast_t - T * Info

  Entry      = KL(N(mu_0, sigma0^2) || N(0, I))                 price of the first state
  Surprise_t = ||mu_t - m_t||^2 / (2 sigma0^2)                   the one-step code length (the JEPA loss)
  Forecast_t = cross-entropy of the target state s_t under a Gaussian decoder centred on a forecast,
               minus a constant; a lower bound on the forecast's information is Info - Forecast_t
  Info       = information of one state beyond the anchor, I(x_t; s_t | x_A), estimated in state space

With horizon="one_step" and forecast_noise=False, Forecast_t equals Surprise_t, so the total weight on
the prediction loss is beta + lam. The anchor is encoded by the same encoder and is used only in Info:
anything constant within an episode cancels in the state's change relative to the anchor's state.

Knobs (spec Section 1.2):
  lam             weight of the forecast-information term (0 gives the round-1 objective with Info exact)
  horizon         "one_step": teacher-forced prior mean from the encoded past;
                  "rollout": open-loop rollout of the prior from the encoded first state
  forecast_noise  False: decode from the prior mean (decoder N(m, sigma0^2));
                  True: decode from a sigma0-noisy sample (decoder N(sample, 2 sigma0^2))
  prior_input     "sample": the prior reads posterior samples s (canonical: what a receiver holds);
                  "mean": the prior reads mu (LeWM-like; the rate is then not a code length)
  estimator       "logdet": 0.5 logdet(I + Sigma / sigma0^2) with Sigma pooled over the batch (canonical);
                  "logdet_window": the same per window, from its T changes relative to its own anchor, then averaged.
                  Both are Gaussian estimates. The pooled one is an upper bound on Info (Gaussian maximum entropy
                  plus a Jensen gap across anchors), which an encoder can inflate by making the anchor-relative
                  distribution differ across episodes; the per-window one removes that gap. Every step logs both,
                  plus a pairwise (collision) value, as gaming diagnostics.
  conditioning    "anchor": Sigma from state changes relative to the anchor's state (canonical);
                  "batch": Sigma from the batch covariance of states (no anchor; ablation)

There is no renderer, no EMA, no stop-gradient and no augmentation. Planning, probes and
LeWM's rollout use encoder means and prior means exactly as for L2.
"""
import math
import torch

from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.codes import entry_nats, surprise_nats

FORECAST_STREAM = 7919   # separate random stream for forecast noise (same convention as forecast_scored.py)
ACTIVE_NATS = 0.1        # a direction counts as informative above this many nats per frame


class DecoderFreeLeWM(ASJepaLeWM):
    def __init__(self, lam=1., horizon="one_step", forecast_noise=False, prior_input="sample",
                 estimator="logdet", conditioning="anchor", forecast_seed=0, **kwargs):
        super().__init__(**kwargs)
        if self.mode != "L2":
            raise ValueError("the decoder-free objective applies to the AS-JEPA L2 state only")
        if lam < 0 or not math.isfinite(lam):
            raise ValueError("lam must be finite and non-negative")
        if (horizon not in {"one_step", "rollout"} or prior_input not in {"sample", "mean"}
                or estimator not in {"logdet", "logdet_window"} or conditioning not in {"anchor", "batch"}):
            raise ValueError("unknown horizon, prior_input, estimator or conditioning")
        if estimator == "logdet_window" and conditioning != "anchor":
            raise ValueError("the per-window estimator needs the anchor reference")
        self.lam, self.horizon, self.forecast_noise = float(lam), horizon, bool(forecast_noise)
        self.prior_input, self.estimator, self.conditioning = prior_input, estimator, conditioning
        self.forecast_seed = int(forecast_seed)
        self.renderer = None      # built by ASJepaLeWM last, so all other initial tensors match L2 at a seed

    def audit_expected_gradients(self):
        return {name: "active" for name in ("encoder", "pool", "predictor", "action_encoder")}

    # ---------------------------------------------------------------- inputs
    def frames(self, batch):
        """Window frames and the anchor at encoder resolution, with the run's domain transform,
        normalized: [B, W + 1, C, H, W]; the anchor is the last entry."""
        pixels = torch.cat([batch["pixels"].float(), batch["anchor224"].float()[:, None]], 1)
        if self.domain == "color":     # identical to ASJepaLeWM.prepare
            ep = batch["episode"].to(pixels.device).float()
            factors = torch.stack([.35 + .65 * torch.remainder(torch.sin(ep * 12.9898 + k * 78.233) * 43758.5453, 1)
                                   for k in range(3)], 1)
            bias = torch.stack([35 * torch.remainder(torch.sin(ep * 9.125 + k * 23.77) * 12781.13, 1)
                                for k in range(3)], 1)
            pixels = (pixels * factors[:, None, :, None, None] + bias[:, None, :, None, None]).clamp(0, 255)
        if self.domain == "dot":
            rows = torch.cat([batch["rows"], batch["anchor_row"][:, None]], 1)
            pixels = self.add_dot(pixels, rows)
        return self.normalize(pixels)

    # ---------------------------------------------------------------- information term
    def information(self, mu, mu_A):
        """Information of one state beyond the anchor, in nats per frame, as trained (float64).
        mu: [B, T, K, d] (frames 1..T); mu_A: [B, K, d]."""
        D = mu[0, 0].numel()
        s2 = self.sigma0 ** 2
        if self.conditioning == "batch":
            flat = mu.flatten(0, 1).flatten(1).double()
            flat = flat - flat.mean(0)
            cov = flat.T @ flat / len(flat)
            return 0.5 * torch.logdet(torch.eye(D, dtype=cov.dtype, device=cov.device) + cov / s2)
        diff = (mu - mu_A[:, None]).flatten(2).double()                           # [B, T, D] change relative to the anchor
        if self.estimator == "logdet_window":                                     # per window, Sylvester form (T x T)
            T = diff.shape[1]
            gram = diff @ diff.transpose(1, 2) / (2 * T * s2)
            eye = torch.eye(T, dtype=gram.dtype, device=gram.device)
            return 0.5 * torch.logdet(eye + gram).mean()
        flat = diff.flatten(0, 1)
        cov = flat.T @ flat / (2 * len(flat))                                     # E[dd^T] = 2 Sigma for independent draws
        return 0.5 * torch.logdet(torch.eye(D, dtype=cov.dtype, device=cov.device) + cov / s2)

    @torch.no_grad()
    def information_diagnostics(self, mu, mu_A):
        """Pooled and per-window log-det, the pairwise (collision) value and the participation ratio, all on
        anchor-relative changes. A pooled value far above the per-window one flags heterogeneity across
        episodes; a participation ratio far above the number of factors the probes find flags a curled code.
        The pairwise value is biased upward on minibatches; it is a trend diagnostic, not a bound."""
        D = mu[0, 0].numel()
        s2 = self.sigma0 ** 2
        diff = (mu - mu_A[:, None]).flatten(2).double()
        T = diff.shape[1]
        flat = diff.flatten(0, 1)
        cov = flat.T @ flat / (2 * len(flat))
        eye = torch.eye(D, dtype=cov.dtype, device=cov.device)
        pooled = 0.5 * torch.logdet(eye + cov / s2)
        gram = diff @ diff.transpose(1, 2) / (2 * T * s2)
        window = 0.5 * torch.logdet(torch.eye(T, dtype=gram.dtype, device=gram.device) + gram).mean()
        q = flat.square().sum(1) / (4 * s2)
        pairwise = -(torch.logsumexp(-q, 0) - math.log(len(q))) - 0.5 * D * math.log(math.e / 2)
        ev = torch.linalg.eigvalsh(cov / s2).clamp_min(0)
        participation = ev.sum().square() / ev.square().sum().clamp_min(1e-30)
        dirs = (0.5 * torch.log1p(ev) > ACTIVE_NATS).sum()
        return dict(info_pooled=pooled, info_window=window, info_pairwise=pairwise,
                    participation=participation, info_dirs=dirs)

    # ---------------------------------------------------------------- forecasts
    def forecast_term(self, mu, s, inputs, predicted, act_emb, surprise, step):
        """Forecast_t per frame, [B, T]: the decoder cross-entropy of s_t minus its constant."""
        if self.horizon == "one_step" and not self.forecast_noise:
            return surprise                                                         # decoder N(m_t, sigma0^2)
        b, w = mu.shape[:2]
        devices = [mu.device.index] if mu.is_cuda else []
        with torch.random.fork_rng(devices=devices):                                # never shifts the main stream
            torch.manual_seed(FORECAST_STREAM + 1000003 * self.forecast_seed + int(step))
            xi = torch.randn(mu[:, 1:].shape, device=mu.device, dtype=mu.dtype)
            if self.horizon == "one_step":
                centre = predicted + self.sigma0 * xi
            else:                                                                   # open loop from the first state
                history = [inputs[:, 0].flatten(1)]
                out = []
                for t in range(w - 1):
                    nxt = self.predict(torch.stack(history, 1), act_emb[:, :t + 1])[:, -1]
                    if self.forecast_noise:
                        nxt = nxt + self.sigma0 * xi[:, t].flatten(1)
                    history.append(nxt); out.append(nxt)
                centre = torch.stack(out, 1).reshape_as(mu[:, 1:])
        if self.forecast_noise:     # decode the noisy target from a noisy centre: decoder variance 2 sigma0^2
            return (s[:, 1:] - centre).float().square().sum((-2, -1)) / (4 * self.sigma0 ** 2)
        return surprise_nats(mu[:, 1:], centre, self.sigma0)                         # decoder N(centre, sigma0^2)

    # ---------------------------------------------------------------- objective
    def loss(self, batch, step=0, ramp_steps=0):
        if ramp_steps:
            raise ValueError("no schedule on the decoder-free objective")
        info_in = self.encode(dict(pixels=self.frames(batch), action=batch["action"]))
        flat, act_emb = info_in["emb"], info_in["act_emb"]
        b, w1 = flat.shape[:2]; w = w1 - 1
        mu_all = flat.reshape(b, w1, self.state_slots, self.state_width)
        mu, mu_A = mu_all[:, :w], mu_all[:, w]
        s = mu + self.sigma0 * torch.randn_like(mu)
        inputs = s if self.prior_input == "sample" else mu
        predicted = self.predict(inputs.flatten(2)[:, :-1], act_emb[:, :-1]).reshape_as(mu[:, 1:])
        entry = entry_nats(mu[:, 0], self.sigma0)                                   # [B]
        surprise = surprise_nats(mu[:, 1:], predicted, self.sigma0)                 # [B, T]
        forecast = (self.forecast_term(mu, s, inputs, predicted, act_emb, surprise, step)
                    if self.lam > 0 else torch.zeros_like(surprise))
        info = self.information(mu[:, 1:], mu_A)
        T = w - 1
        loss = (self.beta * entry.mean() + self.beta * surprise.sum(1).mean()
                + self.lam * forecast.sum(1).mean() - T * info.to(entry.dtype))
        with torch.no_grad():
            prev = inputs[:, :-1].float()
            d_true, d_pred = mu[:, 1:].float() - prev, predicted.float() - prev
            calib_slope = (d_true * d_pred).sum() / d_pred.square().sum().clamp_min(1e-12)
            per_dim_rate = (.5 * mu[:, 0].float().square()
                            + (mu[:, 1:].float() - predicted.float()).square().sum(1) / (2 * self.sigma0 ** 2)).mean(0)
            metrics = dict(loss=loss, info=info.detach().float(), entry=entry.mean(), surprise=surprise.mean(),
                           forecast=forecast.mean(), rate=(entry + surprise.sum(1)).mean(),
                           feature_std=mu.float().std((0, 1)).mean(),
                           spread_over_sigma0=mu.float().std((0, 1)).mean() / self.sigma0,
                           active_dims=(per_dim_rate > ACTIVE_NATS).sum(), calib_slope=calib_slope)
            metrics.update(self.information_diagnostics(mu[:, 1:].detach(), mu_A.detach()))
        return loss, metrics

    @torch.no_grad()
    def render_states(self, *args, **kwargs):
        raise RuntimeError("DecoderFreeLeWM has no renderer; train a decoder on the frozen state for visualization")
