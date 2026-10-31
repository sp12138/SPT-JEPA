"""AS-JEPA L2 with the forecast-scored renderer (round 2, lambda).

For every rendered frame t, a coin with probability `lam` decides what the renderer reads:
the posterior sample s_t = mu(x_t) + sigma0 * noise (as in L2), or a sample of the prior's
forecast s_hat_t = m_t + sigma0 * noise, where m_t is the prior mean computed from earlier
states and actions. Everything else is ASJepaLeWM.loss unchanged: same encoder, prior,
rate (one-step Gaussian KL), renderer and flow loss. At lam = 0 the loss equals
ASJepaLeWM.loss exactly, with the same random draws.

horizon="one_step" (canonical): m_t is the teacher-forced one-step prior mean, i.e. the
  same tensor the surprise term uses.
horizon="rollout" (variant): m_t comes from an open-loop rollout of prior samples started
  at the encoded first state, with sigma0 noise at every step.
forecast_grad="joint" (default): no stop-gradient anywhere.
forecast_grad="encoder_only" (fallback, Section 12 of the spec): the forecast branch sends
  no gradient into the prior's parameters (predictor and action encoder); states still
  receive it. Used only when the calibration rule fires.

Coins, forecast noise and any extra predictor dropout come from their own random stream,
so the posterior noise, flow noise and renderer dropout are the same draws for every lam.
"""
import torch
from torch.func import functional_call

from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.constants import FRAME_ELEMS
from vforge.models.as_jepa.pixels import pack
from vforge.models.as_jepa.flow import flow_loss
from vforge.models.as_jepa.codes import entry_nats, surprise_nats

FORECAST_STREAM = 7919  # offset of the separate random stream for coins and forecast noise


class ForecastScoredLeWM(ASJepaLeWM):
    def __init__(self, lam=0., horizon="one_step", forecast_grad="joint", forecast_seed=0, **kwargs):
        super().__init__(**kwargs)
        if self.mode != "L2" or self.objective != "flow":
            raise ValueError("lambda applies to the AS-JEPA L2 flow objective only")
        if not 0. <= lam <= 1.:
            raise ValueError("lam must lie in [0, 1]")
        if horizon not in {"one_step", "rollout"} or forecast_grad not in {"joint", "encoder_only"}:
            raise ValueError("unknown horizon or forecast_grad")
        self.lam, self.horizon, self.forecast_grad = float(lam), horizon, forecast_grad
        self.forecast_seed = int(forecast_seed)

    def _prior_mean(self, states, act_emb):
        """Prior mean of the next states (identity carry included). With
        forecast_grad='encoder_only' the prior's own parameters get no gradient from this call."""
        if self.forecast_grad == "joint":
            return self.predict(states, act_emb)
        frozen = {k: v.detach() for k, v in self.predictor.named_parameters()}
        frozen.update(dict(self.predictor.named_buffers()))
        return states + functional_call(self.predictor, frozen, (states, act_emb.detach()))

    def _forecast(self, s, act_emb, xi):
        """A prior sample for every rendered frame: s_hat_t = m_t + sigma0 * xi_t, [B, T, K, d]."""
        b, w = s.shape[:2]
        if self.horizon == "one_step":                 # teacher-forced: m_t reads the encoded s_<t
            m = self._prior_mean(s.flatten(2)[:, :-1], act_emb[:, :-1]).reshape_as(s[:, 1:])
            return m + self.sigma0 * xi
        history = [s[:, 0].flatten(1)]                  # open loop from the encoded first state
        for t in range(w - 1):
            m = self._prior_mean(torch.stack(history, 1), act_emb[:, :t + 1])[:, -1]
            history.append(m + self.sigma0 * xi[:, t].flatten(1))
        return torch.stack(history[1:], 1).reshape_as(s[:, 1:])

    def loss(self, batch, step=0, ramp_steps=0):
        pixels, render, anchor = self.prepare(batch)
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
                if self.horizon == "one_step" and self.forecast_grad == "joint":
                    forecast = predicted + self.sigma0 * xi   # reuses the surprise term's forward pass
                else:
                    forecast = self._forecast(s, info["act_emb"], xi)
            condition = torch.where(use, forecast, condition)
        y = pack(render[:, 1:].flatten(0, 1)); a = pack(anchor).repeat_interleave(w - 1, dim=0)
        fm = flow_loss(self.renderer, y, condition.flatten(0, 1), a, self.training_shift)
        rate = (entry + surprise.sum(1)).mean() / ((w - 1) * FRAME_ELEMS)
        beta = self.beta * min(1., step / max(1, ramp_steps)) if ramp_steps else self.beta
        loss = fm + beta * rate
        with torch.no_grad():
            d_true = (mu[:, 1:] - s[:, :-1]).float(); d_pred = (predicted - s[:, :-1]).float()
            calib_slope = (d_true * d_pred).sum() / d_pred.square().sum().clamp_min(1e-12)
            per_dim = (.5 * mu[:, 0].float().square()
                       + (mu[:, 1:].float() - predicted.float()).square().sum(1) / (2 * self.sigma0 ** 2)).mean(0)
        return loss, dict(loss=loss, fm=fm, rate=rate, entry=entry.mean(), surprise=surprise.mean(),
                          rate_share=self.beta * rate / fm.detach().clamp_min(1e-8),
                          feature_std=mu.float().std((0, 1)).mean(), active_dims=(per_dim > .1).sum(),
                          forecast_share=use.float().mean(), calib_slope=calib_slope)


@torch.no_grad()
def calibration_slope(model, s, mu, act_emb):
    """Least-squares slope of the true state change on the prior's predicted change.

    s: posterior samples [B, W, K, d]; mu: encoder means [B, W, K, d]. With the identity carry,
    true change = mu_t - s_{t-1}, predicted change = m_t - s_{t-1}. A prior whose mean is the
    conditional expectation gives slope 1; below 1 its predicted changes are exaggerated
    (overshoot), above 1 they are shrunk. Evaluate in FP32 on held-out windows.
    """
    m = model.predict(s.flatten(2)[:, :-1], act_emb[:, :-1]).reshape_as(mu[:, 1:])
    prev = s[:, :-1]
    d_true, d_pred = (mu[:, 1:] - prev).double(), (m - prev).double()
    return float((d_true * d_pred).sum() / d_pred.square().sum().clamp_min(1e-30))
