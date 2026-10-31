"""Candidate learned-variance code (posterior and prior variances), checked on CPU. Draft for the round-2 spec."""
import math
import torch
from torch import nn
from torch.func import functional_call

from as_jepa_lewm.forecast_scored import ForecastScoredLeWM, FORECAST_STREAM
from vforge.models.as_jepa.constants import FRAME_ELEMS
from vforge.models.as_jepa.pixels import pack
from vforge.models.as_jepa.flow import flow_loss

LOGVAR_MIN, LOGVAR_MAX = 2 * math.log(1e-3), 0.0   # numerical guard: code spread between 0.001 and 1 prior unit


def entry_nats_lv(mu0, lv0):                         # KL(N(mu0, e^lv0) || N(0, I)) per window, [B]
    mu0, lv0 = mu0.float(), lv0.float()
    return 0.5 * (lv0.exp() + mu0.square() - 1 - lv0).sum((-2, -1))


def surprise_nats_lv(mu, lvq, m, lvp):               # KL(N(mu, e^lvq) || N(m, e^lvp)), [B, T]
    mu, lvq, m, lvp = mu.float(), lvq.float(), m.float(), lvp.float()
    return 0.5 * (lvp - lvq + (lvq.exp() + (mu - m).square()) / lvp.exp() - 1).sum((-2, -1))


class LearnedVarianceLeWM(ForecastScoredLeWM):
    """Same modules as L2; the pooling head and the predictor's output layer gain a log-variance half.

    Do not combine with the trainer's `calibrate_state_std`: that rescales `pool.head` as a mean-only head.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        width, dd = self.pool.head.in_features, self.state_width
        head = nn.Linear(width, 2 * dd)
        with torch.no_grad():
            head.weight[:dd].copy_(self.pool.head.weight); head.bias[:dd].copy_(self.pool.head.bias)
            head.weight[dd:].zero_(); head.bias[dd:].fill_(2 * math.log(self.sigma0))
        self.pool.head = head
        out = self.predictor.transformer.output_proj
        proj = nn.Linear(out.in_features, 2 * out.out_features)
        with torch.no_grad():
            proj.weight[:out.out_features].copy_(out.weight); proj.bias[:out.out_features].copy_(out.bias)
            proj.weight[out.out_features:].zero_(); proj.bias[out.out_features:].fill_(2 * math.log(self.sigma0))
        self.predictor.transformer.output_proj = proj

    def encode(self, info):
        pixels = info["pixels"]; b, w = pixels.shape[:2]
        tokens = self.encoder(pixels.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state
        out = self.pool(tokens).reshape(b, w, self.state_slots, 2 * self.state_width)
        mu, lv = out.split(self.state_width, -1)
        info["emb"] = mu.reshape(b, w, self.state_dim)            # planning and probes read the mean, as before
        info["logvar"] = lv.clamp(LOGVAR_MIN, LOGVAR_MAX)
        if "action" in info: info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict_dist(self, emb, act_emb, frozen=False):
        """Prior mean (identity carry) and log variance of the next states. With frozen=True the
        prior's own parameters receive no gradient from this call (states still do)."""
        if frozen:
            params = {k: v.detach() for k, v in self.predictor.named_parameters()}
            params.update(dict(self.predictor.named_buffers()))
            out = functional_call(self.predictor, params, (emb, act_emb.detach()))
        else:
            out = self.predictor(emb, act_emb)
        change, lv = out.split(self.state_dim, -1)
        return emb + change, lv.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def predict(self, emb, act_emb):                              # LeWM's rollout reads means only
        return self.predict_dist(emb, act_emb)[0]

    def loss(self, batch, step=0, ramp_steps=0):
        pixels, render, anchor = self.prepare(batch)
        info = self.encode(dict(pixels=pixels, action=batch["action"]))
        b, w = info["emb"].shape[:2]
        shape = (b, w, self.state_slots, self.state_width)
        mu, lvq = info["emb"].reshape(shape), info["logvar"]
        s = mu + (0.5 * lvq).exp() * torch.randn_like(mu)
        m, lvp = self.predict_dist(s.flatten(2)[:, :-1], info["act_emb"][:, :-1])
        m, lvp = m.reshape_as(mu[:, 1:]), lvp.reshape_as(mu[:, 1:])
        entry = entry_nats_lv(mu[:, 0], lvq[:, 0]); surprise = surprise_nats_lv(mu[:, 1:], lvq[:, 1:], m, lvp)
        condition = s[:, 1:]
        use = torch.zeros(b, w - 1, 1, 1, dtype=torch.bool, device=mu.device)
        if self.lam > 0:
            if self.horizon != "one_step":
                raise NotImplementedError("learned variance: one-step forecasts only in this round")
            devices = [mu.device.index] if mu.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(FORECAST_STREAM + 1000003 * self.forecast_seed + int(step))
                use = torch.rand(b, w - 1, 1, 1, device=mu.device) < self.lam
                xi = torch.randn(mu[:, 1:].shape, device=mu.device, dtype=mu.dtype)
                if self.forecast_grad == "joint":
                    m_fc, lvp_fc = m, lvp                  # reuses the rate's forward pass
                else:                                      # encoder_only: same forecast, prior parameters detached
                    m_fc, lvp_fc = self.predict_dist(s.flatten(2)[:, :-1], info["act_emb"][:, :-1], frozen=True)
                    m_fc, lvp_fc = m_fc.reshape_as(mu[:, 1:]), lvp_fc.reshape_as(mu[:, 1:])
            condition = torch.where(use, m_fc + (0.5 * lvp_fc).exp() * xi, condition)
        y = pack(render[:, 1:].flatten(0, 1)); a = pack(anchor).repeat_interleave(w - 1, dim=0)
        fm = flow_loss(self.renderer, y, condition.flatten(0, 1), a, self.training_shift)
        rate = (entry + surprise.sum(1)).mean() / ((w - 1) * FRAME_ELEMS)
        loss = fm + self.beta * rate
        with torch.no_grad():
            # calibration: mean squared miss of a posterior sample over the prior's variance (1 when calibrated)
            var_ratio = ((mu[:, 1:] - m).float().square() + lvq[:, 1:].float().exp()).mean() / lvp.float().exp().mean()
            d_true = (mu[:, 1:] - s[:, :-1]).float(); d_pred = (m - s[:, :-1]).float()
            calib_slope = (d_true * d_pred).sum() / d_pred.square().sum().clamp_min(1e-12)
            mu0, lv0 = mu[:, 0].float(), lvq[:, 0].float()
            per_dim = (0.5 * (lv0.exp() + mu0.square() - 1 - lv0)
                       + 0.5 * (lvp.float() - lvq[:, 1:].float() + (lvq[:, 1:].float().exp()
                                + (mu[:, 1:].float() - m.float()).square()) / lvp.float().exp() - 1).sum(1)).mean(0)
        return loss, dict(loss=loss, fm=fm, rate=rate, entry=entry.mean(), surprise=surprise.mean(),
                          rate_share=self.beta * rate / fm.detach().clamp_min(1e-8),
                          feature_std=mu.float().std((0, 1)).mean(), active_dims=(per_dim > .1).sum(),
                          forecast_share=use.float().mean(), calib_slope=calib_slope, prior_var_ratio=var_ratio,
                          posterior_sd=(0.5 * lvq.float()).exp().mean(), prior_sd=(0.5 * lvp.float()).exp().mean())
