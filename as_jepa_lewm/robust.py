"""Matched-size posterior and forecast ablations of the original full-data AS-JEPA."""
import math
import torch
from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.codes import entry_nats, surprise_nats
from vforge.models.as_jepa.constants import FRAME_ELEMS
from vforge.models.as_jepa.flow import flow_loss
from vforge.models.as_jepa.pixels import pack


class RobustLeWM(ASJepaLeWM):
    def __init__(self, rho=0., forecast_fraction=0., forecast_seed=20261029, **kwargs):
        super().__init__(**kwargs)
        if self.mode != "L2" or self.objective != "flow":
            raise ValueError("This controlled extension applies only to AS-JEPA L2 flow loss")
        if not 0 <= rho < 1 or not 0 <= forecast_fraction <= 1:
            raise ValueError("Require0<=rho<1 and0<=forecast_fraction<=1")
        self.rho = float(rho)
        self.forecast_fraction = float(forecast_fraction)
        self.forecast_seed = int(forecast_seed)
        # Evaluation tools recognize the exact posterior family; no parameter or buffer is added.
        self.method = "correlated_noise" if self.rho else "forecast_render"

    def context_samples(self, mu, samples, generator):
        noise = torch.randn((samples, *mu.shape[1:]), device=mu.device, generator=generator)
        if self.rho:
            for t in range(1, noise.shape[1]):
                noise[:, t] = self.rho * noise[:, t - 1] + math.sqrt(1 - self.rho ** 2) * noise[:, t]
        return mu.expand(samples, -1, -1) + self.sigma0 * noise

    def transition_noise(self, value, generator):
        return self.sigma0 * torch.randn(value.shape, device=value.device, generator=generator)

    def loss(self, batch, step=0, ramp_steps=0):
        if not self.rho and not self.forecast_fraction:
            return super().loss(batch, step, ramp_steps)
        pixels, render, anchor = self.prepare(batch)
        info = self.encode(dict(pixels=pixels, action=batch["action"]))
        flat = info["emb"]
        b, w = flat.shape[:2]
        mu = flat.reshape(b, w, self.state_slots, self.state_width)
        noise = torch.randn_like(mu)
        if self.rho:
            for t in range(1, w):
                noise[:, t] = self.rho * noise[:, t - 1] + math.sqrt(1 - self.rho ** 2) * noise[:, t]
        noisy = mu + self.sigma0 * noise
        predicted = self.predict(noisy.flatten(2)[:, :-1], info["act_emb"][:, :-1]).reshape_as(mu[:, 1:])
        entry = entry_nats(mu[:, 0], self.sigma0)
        target_mean = mu[:, 1:] + self.rho * (noisy[:, :-1] - mu[:, :-1])
        surprise = surprise_nats(target_mean, predicted, self.sigma0)
        variance_kl_per_dimension = .5 * (-self.rho ** 2 - math.log1p(-self.rho ** 2))
        surprise = surprise + self.state_dim * variance_kl_per_dimension
        condition = noisy[:, 1:]
        if self.forecast_fraction:
            devices = [flat.device.index] if flat.is_cuda else []
            # Extra predictor dropout and condition masks use a separate stream.
            # Common posterior/renderer noise keeps its original draw order across arms.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.forecast_seed + int(step))
                history = [noisy[:, 0].flatten(1)]
                future = []
                for t in range(w - 1):
                    past = torch.stack(history, 1)
                    nxt = self.predict(past, info["act_emb"][:, :t + 1])[:, -1]
                    history.append(nxt)
                    future.append(nxt)
                future = torch.stack(future, 1).reshape_as(condition)
                choose = torch.rand(b, w - 1, 1, 1, device=flat.device) < self.forecast_fraction
                condition = torch.where(choose, future, condition)
        y = pack(render[:, 1:].flatten(0, 1))
        anchors = pack(anchor).repeat_interleave(w - 1, dim=0)
        fm = flow_loss(self.renderer, y, condition.flatten(0, 1), anchors, self.shift)
        rate = (entry + surprise.sum(1)).mean() / ((w - 1) * FRAME_ELEMS)
        beta = self.beta * min(1., step / max(1, ramp_steps)) if ramp_steps else self.beta
        loss = fm + beta * rate
        variable_rate = (.5 * mu[:, 0].float().square() +
                         (target_mean.float() - predicted.float()).square().sum(1) / (2 * self.sigma0 ** 2)).mean(0)
        return loss, dict(loss=loss, fm=fm, rate=rate, entry=entry.mean(), surprise=surprise.mean(),
                          rate_share=beta * rate / fm.detach().clamp_min(1e-8),
                          feature_std=mu.float().std((0, 1)).mean(), active_dims=(variable_rate > .1).sum(),
                          temporal_noise_rho=loss.new_tensor(self.rho), forecast_fraction=loss.new_tensor(self.forecast_fraction))
