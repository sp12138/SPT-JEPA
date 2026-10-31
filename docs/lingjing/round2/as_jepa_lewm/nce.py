"""Candidate reconstruction-free AS-JEPA (contrastive critic instead of the renderer), checked on CPU."""
import math
import torch
from torch.nn import functional as F

from as_jepa_lewm.forecast_scored import ForecastScoredLeWM, FORECAST_STREAM
from vforge.models.as_jepa.codes import entry_nats, surprise_nats


class NCELeWM(ForecastScoredLeWM):
    """Same encoder, pooling, action encoder and prior as L2; no renderer.

    The renderer's flow loss is replaced by an InfoNCE bound on the information that the
    condition (encoded sample at lambda = 0, one-step forecast sample at lambda = 1) carries
    about its frame. The critic is the encoder's own Gaussian log-likelihood, so it adds no
    parameters: score(t, j) = log N(condition_t; mu(x_j), sigma0^2 I) up to a constant.
    Candidates are all rendered frames of the batch (B * T of them), including the other
    frames of the same window. Everything is in nats, so beta here is a calibrated price.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renderer = None   # nothing renders; generation is not available for this model

    def audit_expected_gradients(self):
        # The trainer's gradient audit expects a renderer by default; this model has none.
        return {name: "active" for name in ("encoder", "pool", "predictor", "action_encoder")}

    def loss(self, batch, step=0, ramp_steps=0):
        pixels, _, _ = self.prepare(batch)
        info = self.encode(dict(pixels=pixels, action=batch["action"]))
        flat = info["emb"]; b, w = flat.shape[:2]
        mu = flat.reshape(b, w, self.state_slots, self.state_width)
        s = mu + self.sigma0 * torch.randn_like(mu)
        predicted = self.predict(s.flatten(2)[:, :-1], info["act_emb"][:, :-1]).reshape_as(mu[:, 1:])
        entry = entry_nats(mu[:, 0], self.sigma0); surprise = surprise_nats(mu[:, 1:], predicted, self.sigma0)
        condition = s[:, 1:]
        use = torch.zeros(b, w - 1, 1, 1, dtype=torch.bool, device=mu.device)
        if self.lam > 0:
            devices = [mu.device.index] if mu.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(FORECAST_STREAM + 1000003 * self.forecast_seed + int(step))
                use = torch.rand(b, w - 1, 1, 1, device=mu.device) < self.lam
                xi = torch.randn(mu[:, 1:].shape, device=mu.device, dtype=mu.dtype)
            condition = torch.where(use, predicted + self.sigma0 * xi, condition)
        q = condition.flatten(0, 1).flatten(1).float()               # [N, STATE_DIM], N = B * T
        c = mu[:, 1:].flatten(0, 1).flatten(1).float()                # candidates: encoder means of the same frames
        score = -torch.cdist(q, c).square() / (2 * self.sigma0 ** 2)  # [N, N]
        target = torch.arange(len(q), device=q.device)
        nce = F.cross_entropy(score, target)                           # nats per frame
        rate = (entry + surprise.sum(1)).mean()                        # nats per window
        loss = (w - 1) * nce + self.beta * rate                        # per window, both terms in nats
        with torch.no_grad():
            d_true = (mu[:, 1:] - s[:, :-1]).float(); d_pred = (predicted - s[:, :-1]).float()
            calib_slope = (d_true * d_pred).sum() / d_pred.square().sum().clamp_min(1e-12)
            per_dim = (.5 * mu[:, 0].float().square()
                       + (mu[:, 1:].float() - predicted.float()).square().sum(1) / (2 * self.sigma0 ** 2)).mean(0)
        return loss, dict(loss=loss, nce=nce, info_bound=math.log(len(q)) - nce, rate=rate,
                          entry=entry.mean(), surprise=surprise.mean(), feature_std=mu.float().std((0, 1)).mean(),
                          active_dims=(per_dim > .1).sum(), forecast_share=use.float().mean(), calib_slope=calib_slope)
