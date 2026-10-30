"""Matched-size posterior and forecast ablations of the original full-data AS-JEPA."""
import math
import torch
from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.codes import entry_nats, surprise_nats
from vforge.models.as_jepa.constants import FRAME_ELEMS
from vforge.models.as_jepa.flow import flow_loss
from vforge.models.as_jepa import flow as flow_ops
from vforge.models.as_jepa.pixels import pack


class RobustLeWM(ASJepaLeWM):
    def __init__(self, rho=0., forecast_fraction=0., forecast_seed=20261029, rate_prior="one-step",
                 posterior_noise_scale=1., flow_force_t_one=False, **kwargs):
        super().__init__(**kwargs)
        if self.mode != "L2" or self.objective != "flow":
            raise ValueError("This controlled extension applies only to AS-JEPA L2 flow loss")
        if not 0 <= rho < 1 or not 0 <= forecast_fraction <= 1:
            raise ValueError("Require0<=rho<1 and0<=forecast_fraction<=1")
        if rate_prior not in {"one-step","forecast"}:raise ValueError("Unknown rate prior")
        if not math.isfinite(posterior_noise_scale) or not 0 <= posterior_noise_scale <= 1:
            raise ValueError("posterior_noise_scale must be finite and in [0,1]")
        if not isinstance(flow_force_t_one, bool):
            raise ValueError("flow_force_t_one must be a bool")
        self.rate_prior = rate_prior
        self.rho = float(rho)
        self.forecast_fraction = float(forecast_fraction)
        self.forecast_seed = int(forecast_seed)
        # Training-only causal diagnostics. Sampling noise and target corruption
        # change independently of the rate's sigma0; this is not an exact ELBO.
        # Evaluation and checkpoint parameter structure remain unchanged.
        self.posterior_noise_scale = float(posterior_noise_scale)
        self.flow_force_t_one = flow_force_t_one
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
        if (not self.rho and not self.forecast_fraction and self.rate_prior == "one-step"
                and self.posterior_noise_scale == 1. and not self.flow_force_t_one):
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
        # Still draw the noise when disabled, preserving paired outer RNG state.
        if self.posterior_noise_scale != 1.:
            noise = noise * self.posterior_noise_scale
        noisy = mu + self.sigma0 * noise
        predicted = self.predict(noisy.flatten(2)[:, :-1], info["act_emb"][:, :-1]).reshape_as(mu[:, 1:])
        entry = entry_nats(mu[:, 0], self.sigma0)
        target_mean = mu[:, 1:] + self.rho * (noisy[:, :-1] - mu[:, :-1])
        surprise = surprise_nats(target_mean, predicted, self.sigma0)
        variance_kl_per_dimension = .5 * (-self.rho ** 2 - math.log1p(-self.rho ** 2))
        surprise = surprise + self.state_dim * variance_kl_per_dimension
        condition = noisy[:, 1:]
        if self.forecast_fraction or self.rate_prior == "forecast":
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
        if self.rate_prior == "forecast":
            # The rate and renderer constrain the same causal forecast trajectory.
            # The reference one-step forward above preserves common RNG draw order
            # for the matched adaptation control; it contributes no extra objective.
            predicted = future
            surprise = surprise_nats(target_mean, predicted, self.sigma0)
            surprise = surprise + self.state_dim * variance_kl_per_dimension
        y = pack(render[:, 1:].flatten(0, 1))
        anchors = pack(anchor).repeat_interleave(w - 1, dim=0)
        if self.flow_force_t_one:
            # Consume the original time draw before replacing its value so that
            # posterior, image-noise and later random draws remain paired.
            t = flow_ops.draw_timesteps(y.shape[0], y.device, self.training_shift)
            t = torch.ones_like(t)
            eps = torch.randn_like(y)
            with self.renderer.anchored(anchors):
                velocity = self.renderer(eps, t, condition=condition.flatten(0, 1))
            fm = (y - eps - velocity).square().mean()
        else:
            fm = flow_loss(self.renderer, y, condition.flatten(0, 1), anchors, self.training_shift)
        rate = (entry + surprise.sum(1)).mean() / ((w - 1) * FRAME_ELEMS)
        beta = self.beta * min(1., step / max(1, ramp_steps)) if ramp_steps else self.beta
        loss = fm + beta * rate
        variable_rate = (.5 * mu[:, 0].float().square() +
                         (target_mean.float() - predicted.float()).square().sum(1) / (2 * self.sigma0 ** 2)).mean(0)
        return loss, dict(loss=loss, fm=fm, rate=rate, entry=entry.mean(), surprise=surprise.mean(),
                          rate_share=beta * rate / fm.detach().clamp_min(1e-8),
                          feature_std=mu.float().std((0, 1)).mean(), active_dims=(variable_rate > .1).sum(),
                          temporal_noise_rho=loss.new_tensor(self.rho), forecast_fraction=loss.new_tensor(self.forecast_fraction))
