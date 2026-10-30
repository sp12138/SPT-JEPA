"""Matched-size posterior and forecast ablations of the original full-data AS-JEPA."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.codes import entry_nats, surprise_nats
from vforge.models.as_jepa.constants import FRAME_ELEMS
from vforge.models.as_jepa.flow import flow_loss
from vforge.models.as_jepa import flow as flow_ops
from vforge.models.as_jepa.pixels import pack


class RobustLeWM(ASJepaLeWM):
    def __init__(self, rho=0., forecast_fraction=0., forecast_seed=20261029, rate_prior="one-step",
                 posterior_noise_scale=1., flow_force_t_one=False, response_query_mode="none",
                 response_query_encoded_fraction=0., **kwargs):
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
        if response_query_mode not in {"none", "direct", "annihilator", "direct_relative",
                                       "annihilator_relative", "random_relative"}:
            raise ValueError("Unknown response_query_mode")
        if response_query_mode != "none" and kwargs.get("action_dim", 10) % 5:
            raise ValueError("Response queries require five raw actions per observation block")
        if not math.isfinite(response_query_encoded_fraction) or not 0 <= response_query_encoded_fraction <= 1:
            raise ValueError("response_query_encoded_fraction must be finite and in [0,1]")
        if response_query_mode == "none" and response_query_encoded_fraction != 0:
            raise ValueError("Encoded response queries require an active response_query_mode")
        self.response_query_mode = response_query_mode
        self.response_query_encoded_fraction = float(response_query_encoded_fraction)
        if response_query_mode != "none":
            # The shared 550 baseline tensors and caller RNG stay unchanged.
            # Direct and annihilator draw exactly the same auxiliary head.
            with torch.random.fork_rng(devices=[]):
                self.response_query_head = nn.Linear(self.state_dim, 3 * 8 * 8)
        # Evaluation tools recognize the exact posterior family; no parameter or buffer is added.
        self.method = "correlated_noise" if self.rho else "forecast_render"

    @staticmethod
    def response_projector(actions):
        """Per-clip orthogonal complement of [1, time, cumulative controls].

        Input is the existing standardized five-action blocks. The time column
        makes subtracting any per-step action mean a span-preserving change.
        No final padding action, physical label, or other clip is consulted.
        """
        if actions.ndim != 3 or actions.shape[-1] % 5:
            raise ValueError("Expected [batch, window, 5*action_dim] controls")
        b, w, width = actions.shape
        with torch.autocast(device_type=actions.device.type, enabled=False):
            blocks = actions.detach().to(torch.float64).reshape(b, w, 5, width // 5)
            cumulative = torch.cat([blocks.new_zeros(b, 1, width // 5),
                                    blocks[:, :-1].sum(2).cumsum(1)], 1)
            time = torch.arange(w, device=actions.device, dtype=torch.float64) / max(1, w-1)
            design = torch.cat([cumulative.new_ones(b, w, 1),
                                time[None, :, None].expand(b, -1, -1), cumulative], -1)
            norms = torch.linalg.vector_norm(design, dim=1, keepdim=True)
            design = design / torch.where(norms > 0, norms, torch.ones_like(norms))
            u, singular, _ = torch.linalg.svd(design, full_matrices=False)
            # Rank tolerance reflects FP32 action data, rather than pretending
            # standardized/rounded inputs have FP64 measurement precision.
            tolerance = max(design.shape[-2:]) * torch.finfo(torch.float32).eps * singular[:, :1]
            keep = singular > tolerance
            rank = w - keep.sum(1)
            projector = torch.eye(w, device=actions.device, dtype=torch.float64)[None] - (u * keep[:, None]) @ u.transpose(1, 2)
        return projector.float(), rank

    @staticmethod
    @lru_cache(maxsize=None)
    def _response_random_basis(window):
        """One fixed CPU-QR basis in span{constant, time}'s complement.

        This derived constant is not a learned tensor or checkpoint buffer.
        A private generator leaves every training/evaluation RNG untouched.
        """
        time = torch.arange(window, device="cpu", dtype=torch.float64)
        design = torch.stack([torch.ones_like(time), time], 1)
        complement = torch.linalg.qr(design, mode="complete").Q[:, min(2, window):]
        width = complement.shape[1]
        if width == 0:
            return complement
        generator = torch.Generator(device="cpu").manual_seed(20261031)
        gaussian = torch.randn(width, width, generator=generator, device="cpu", dtype=torch.float64)
        rotation = torch.linalg.qr(gaussian).Q
        return (complement @ rotation).contiguous()

    def response_query_operator(self, actions):
        """Exact current query operator; `none` exposes identity for audits."""
        if actions.ndim != 3:
            raise ValueError("Expected [batch, window, action_width] controls")
        b, w = actions.shape[:2]
        if self.response_query_mode in {"none", "direct", "direct_relative"}:
            return (torch.eye(w, device=actions.device, dtype=torch.float32)[None].expand(b, -1, -1),
                    torch.full((b,), w, device=actions.device, dtype=torch.long))
        projector, rank = self.response_projector(actions)
        if self.response_query_mode == "random_relative":
            basis = self._response_random_basis(w).to(actions.device)
            keep = torch.arange(basis.shape[1], device=actions.device)[None] < rank[:, None]
            with torch.autocast(device_type=actions.device.type, enabled=False):
                projector = ((basis[None] * keep[:, None]) @ basis.T).float()
        return projector, rank

    def response_query_loss(self, flat, act_emb, actions, render, step):
        """Grounded clean-latent prediction; original DiT/rate paths unchanged."""
        b, w = flat.shape[:2]
        devices = [flat.device.index] if flat.is_cuda else []
        # Retain the model's train/eval dropout mode, without consuming the
        # posterior/renderer stream. Clean means no posterior sampling noise.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.forecast_seed + 1000003 + int(step))
            history = [flat[:, 0]]
            for t in range(w - 1):
                history.append(self.predict(torch.stack(history, 1), act_emb[:, :t+1])[:, -1])
            forecast = torch.stack(history, 1)
            prediction = self.response_query_head(forecast).float()
        target = F.adaptive_avg_pool2d(render.detach().float().flatten(0, 1), (8, 8)).reshape(b, w, -1)
        error = prediction - target
        anchor = error[:, 0].square().mean(-1)
        if self.response_query_mode in {"annihilator", "annihilator_relative", "random_relative"}:
            projector, rank = self.response_query_operator(actions)
            with torch.autocast(device_type=flat.device.type, enabled=False):
                residual = torch.bmm(projector, error.float())
                query = residual.square().sum((1, 2)) / (rank.clamp_min(1) * error.shape[-1])
                query = torch.where(rank > 0, query, torch.zeros_like(query))
        else:
            query = error.square().mean((1, 2))
            rank = torch.full((b,), w, device=flat.device, dtype=torch.long)
        if self.response_query_encoded_fraction:
            # Ground the same target on the already-observed per-frame codes.
            # This branch never supplies future codes to the causal rollout,
            # which is still computed above in full in every experimental arm.
            encoded_error = self.response_query_head(flat).float() - target
            if self.response_query_mode in {"direct", "direct_relative"}:
                encoded_query = encoded_error.square().mean((1, 2))
            else:
                with torch.autocast(device_type=flat.device.type, enabled=False):
                    encoded_residual = torch.bmm(projector, encoded_error)
                    encoded_query = encoded_residual.square().sum((1, 2)) / (rank.clamp_min(1) * error.shape[-1])
                    encoded_query = torch.where(rank > 0, encoded_query, torch.zeros_like(encoded_query))
            forecast_query = query
            query = ((1 - self.response_query_encoded_fraction) * forecast_query
                     + self.response_query_encoded_fraction * encoded_query)
        relative_metrics = {}
        if self.response_query_mode.endswith("_relative"):
            with torch.autocast(device_type=flat.device.type, enabled=False):
                if self.response_query_mode == "direct_relative":
                    energy = target.square().mean((1, 2))
                else:
                    projected_target = torch.bmm(projector, target)
                    energy = projected_target.square().sum((1, 2)) / (rank.clamp_min(1) * target.shape[-1])
                    energy = torch.where(rank > 0, energy, torch.zeros_like(energy))
                # Render RGB uses [-1,1]. This is one original 8-bit intensity
                # step squared, not an estimated noise variance after pooling.
                resolution_floor = (1. / 127.5) ** 2
                raw_query = query
                query = query / (energy.detach() + resolution_floor)
                positive_rank = rank > 0
                relative_metrics = dict(response_query_raw_outcome=raw_query.mean(),
                    response_query_target_energy=energy.mean(),
                    response_query_floor_dominated_fraction=((energy <= resolution_floor) & positive_rank).float().sum()
                        / positive_rank.sum().clamp_min(1))
        if self.response_query_encoded_fraction:
            denominator = energy.detach() + resolution_floor if self.response_query_mode.endswith("_relative") else 1.
            relative_metrics.update(response_query_forecast_outcome=(forecast_query / denominator).mean(),
                response_query_encoded_outcome=(encoded_query / denominator).mean(),
                response_query_encoded_fraction=query.new_tensor(self.response_query_encoded_fraction))
        auxiliary = (anchor + query).mean()
        return auxiliary, dict(response_query_aux=auxiliary, response_query_anchor=anchor.mean(),
            response_query_outcome=query.mean(), response_query_residual_rank=rank.float().mean(),
            response_query_zero_rank_fraction=(rank == 0).float().mean(),
            response_native_state_std=flat.float().std((0, 1), unbiased=False).mean(),
            response_clean_forecast_std=forecast.float().std((0, 1), unbiased=False).mean(), **relative_metrics)

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
                and self.posterior_noise_scale == 1. and not self.flow_force_t_one
                and self.response_query_mode == "none"):
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
        metrics = dict(loss=loss, fm=fm, rate=rate, entry=entry.mean(), surprise=surprise.mean(),
                          rate_share=beta * rate / fm.detach().clamp_min(1e-8),
                          feature_std=mu.float().std((0, 1)).mean(), active_dims=(variable_rate > .1).sum(),
                          temporal_noise_rho=loss.new_tensor(self.rho), forecast_fraction=loss.new_tensor(self.forecast_fraction))
        if self.response_query_mode != "none":
            auxiliary, response_metrics = self.response_query_loss(flat, info["act_emb"], batch["action"], render, step)
            loss = loss + auxiliary
            metrics.update(response_metrics)
            metrics["loss"] = loss
        return loss, metrics
