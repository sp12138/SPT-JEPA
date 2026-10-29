"""Same-size sample-level objective hypotheses; all are exploratory controls."""
import math
import torch
from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.codes import entry_nats
from vforge.models.as_jepa.constants import FRAME_ELEMS
from vforge.models.as_jepa.flow import draw_timesteps
from vforge.models.as_jepa.pixels import pack

METHODS = {"original", "clean-prior", "innovation-corrected", "antithetic-rate", "marginal-prior",
           "overshoot", "window-free-bits", "motion-weighted", "shared-flow-noise", "differential-flow"}


class FrontierLeWM(ASJepaLeWM):
    def __init__(self, frontier_method="original", auxiliary_fraction=.5, free_nats=.5,
                 motion_strength=4., difference_weight=.25, private_seed=20261029, **kwargs):
        super().__init__(**kwargs)
        if self.mode != "L2" or self.objective != "flow" or frontier_method not in METHODS:
            raise ValueError("Require a named L2 flow ablation")
        self.frontier_method = frontier_method
        self.auxiliary_fraction = float(auxiliary_fraction)
        self.free_nats = float(free_nats)
        self.motion_strength = float(motion_strength)
        self.difference_weight = float(difference_weight)
        self.private_seed = int(private_seed)
        if not 0 <= self.auxiliary_fraction <= 1 or self.free_nats < 0:
            raise ValueError("Invalid fixed objective coefficient")

    def _second_prior(self, states, actions, step):
        devices = [states.device.index] if states.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.private_seed + int(step))
            return self.predict(states, actions)

    def loss(self, batch, step=0, ramp_steps=0):
        method = self.frontier_method
        if method == "original":
            return super().loss(batch, step, ramp_steps)
        pixels, frames, anchor = self.prepare(batch)
        info = self.encode(dict(pixels=pixels, action=batch["action"]))
        mu = info["emb"]
        b, w, dim = mu.shape
        perturb = self.sigma0 * torch.randn_like(mu)
        states = mu + perturb
        prior_inputs = mu if method == "clean-prior" else states
        prediction = self.predict(prior_inputs[:, :-1], info["act_emb"][:, :-1])
        if method == "innovation-corrected":
            # Remove known posterior-noise carried by the identity bypass.
            # This is a residual-prediction surrogate, not the original chain KL.
            prediction = prediction - perturb[:, :-1]
        if method in {"antithetic-rate", "marginal-prior"}:
            opposite = self._second_prior((mu - perturb)[:, :-1], info["act_emb"][:, :-1], step)
            if method == "marginal-prior":
                prediction = (prediction + opposite) * .5
            else:
                second_error = (mu[:, 1:].float() - opposite.float()).square() / (2 * self.sigma0 ** 2)
        per_dim = (mu[:, 1:].float() - prediction.float()).square() / (2 * self.sigma0 ** 2)
        if method == "antithetic-rate":
            per_dim = (per_dim + second_error) * .5
        if method == "overshoot":
            devices = [mu.device.index] if mu.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.private_seed + int(step))
                # Causal forecast from up to3observed states; targets stay attached.
                context = min(3, w - 1)
                history = list(mu[:, :context].unbind(1))
                errors = []
                for t in range(context, w):
                    nxt = self.predict(torch.stack(history, 1), info["act_emb"][:, :t])[:, -1]
                    errors.append((mu[:, t].float() - nxt.float()).square() / (2 * self.sigma0 ** 2))
                    history.append(nxt)
                multi = torch.stack(errors, 1).mean(1) * (w - 1)
            per_window = (1 - self.auxiliary_fraction) * per_dim.sum(1) + self.auxiliary_fraction * multi
        else:
            per_window = per_dim.sum(1)
        entry = entry_nats(mu[:, 0].reshape(b, self.state_slots, self.state_width), self.sigma0)
        raw_rate_per_window = entry + per_window.sum(-1)
        if method == "window-free-bits":
            variable = .5 * mu[:, 0].float().square() + per_window
            entry_constant = entry - .5 * mu[:, 0].float().square().sum(-1)
            priced = variable.clamp_min(self.free_nats).sum(-1) + entry_constant
        else:
            priced = raw_rate_per_window
        y = pack(frames[:, 1:].flatten(0, 1))
        anchors = pack(anchor).repeat_interleave(w - 1, dim=0)
        if method in {"shared-flow-noise", "differential-flow"}:
            times = draw_timesteps(b, y.device, self.shift).repeat_interleave(w - 1, 0)
            eps = torch.randn_like(y.reshape(b, w - 1, *y.shape[1:])[:, 0])
            eps = eps[:, None].expand(-1, w - 1, *eps.shape[1:]).reshape_as(y)
        else:
            times = draw_timesteps(len(y), y.device, self.shift)
            eps = torch.randn_like(y)
        tv = times.reshape(-1, 1, 1, 1, 1)
        with self.renderer.anchored(anchors):
            velocity = self.renderer((1 - tv) * y + tv * eps, times,
                                     condition=states[:, 1:].flatten(0, 1).reshape(-1, self.state_slots, self.state_width))
        error = (y - eps - velocity).float()
        squared = error.square()
        if method == "motion-weighted":
            motion = (frames[:, 1:].float() - frames[:, :-1].float()).abs().mean(2, keepdim=True)
            scale = motion.mean((1, 2, 3, 4), keepdim=True).clamp_min(1e-3)
            weights = 1 + self.motion_strength * (motion / scale).clamp_max(10)
            weights = weights / weights.mean((1, 2, 3, 4), keepdim=True)
            weights = pack(weights.expand(-1, -1, 3, -1, -1).flatten(0, 1))
            fm = (squared * weights).mean()
        else:
            fm = squared.mean()
        differential = fm * 0
        if method == "differential-flow":
            residual = error.reshape(b, w - 1, *error.shape[1:])
            differential = (residual[:, 1:] - residual[:, :-1]).square().mean()
            fm = fm + self.difference_weight * differential
        rate = priced.mean() / ((w - 1) * FRAME_ELEMS)
        raw_rate = raw_rate_per_window.mean() / ((w - 1) * FRAME_ELEMS)
        beta = self.beta * min(1., step / max(1, ramp_steps)) if ramp_steps else self.beta
        loss = fm + beta * rate
        variable = .5 * mu[:, 0].float().square() + per_window
        return loss, dict(loss=loss, fm=fm, rate=rate, raw_rate=raw_rate,
                          entry=entry.mean(), surprise=per_dim.sum(-1).mean(),
                          rate_share=beta * rate / fm.detach().clamp_min(1e-8),
                          feature_std=mu.float().std((0, 1)).mean(),
                          active_dims=(variable.mean(0) > .1).sum(), differential_flow=differential)
