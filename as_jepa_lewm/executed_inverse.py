"""Training-only inverse of the SAME observed transition's forward input.

No goal-conditioned policy, action proposals, or action decoder is deployed.
The inverse head sees observed adjacent states, and optionally the corresponding
forward prediction. Its target is exactly batch['action'][:, :-1], which is also
consumed by the forward predictor for those transitions.
"""
import math

import torch
from torch import nn

from .robust import RobustLeWM


class ExecutedInverseLeWM(RobustLeWM):
    def __init__(self, inverse_kind="mse", inverse_weight=10.,
                 inverse_predicted=False, inverse_width=256,
                 inverse_seed=20261107, inverse_flow_shift=math.log(4.), **kwargs):
        if inverse_kind not in {"mse", "flow"}:
            raise ValueError("Inverse must be an observed-transition MSE or flow loss")
        if kwargs.get("action_dim") != 20 or kwargs.get("window", 8) != 8:
            raise ValueError("This audited experiment uses eight states and five native 4D inputs per transition")
        if kwargs.get("response_query_mode", "none") != "none":
            raise ValueError("No other action/reference objective is allowed")
        super().__init__(**kwargs)
        self.inverse_kind = inverse_kind
        self.inverse_weight = float(inverse_weight)
        self.inverse_predicted = bool(inverse_predicted)
        self.inverse_seed = int(inverse_seed)
        self.inverse_flow_shift = float(inverse_flow_shift)
        self.inverse_action_dim = 20
        if not math.isfinite(self.inverse_weight) or self.inverse_weight <= 0:
            raise ValueError("Inverse weight must be finite and positive")
        extra = self.inverse_action_dim + 9 if inverse_kind == "flow" else 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.inverse_seed)
            self.inverse_head = nn.Sequential(
                nn.Linear(2 * self.state_dim + extra, inverse_width), nn.GELU(),
                nn.Linear(inverse_width, inverse_width), nn.GELU(),
                nn.Linear(inverse_width, self.inverse_action_dim))
        self._inverse_capture = False

    def encode(self, info):
        result = super().encode(info)
        if self._inverse_capture:
            self._inverse_observed = result["emb"]
            self._inverse_forward_inputs = result["act_emb"]
        return result

    def inverse_per_sample(self, origin, outcome, target, noise=None, time=None):
        """One separable loss per clip; never conditions on a desired goal."""
        condition = torch.cat([origin, outcome], -1)
        target = target.detach().float()
        if self.inverse_kind == "mse":
            estimate = self.inverse_head(condition)
            return (estimate.float() - target).square().mean((1, 2))
        if noise is None or time is None:
            raise ValueError("Explicit independently sampled training noise/time required")
        noisy_action = (1 - time) * target + time * noise
        frequency = torch.arange(1, 5, device=time.device, dtype=time.dtype)
        phase = math.pi * time * frequency
        time_features = torch.cat([time, phase.sin(), phase.cos()], -1)
        velocity = self.inverse_head(torch.cat([condition, noisy_action, time_features], -1))
        return (velocity.float() - (noise - target)).square().mean((1, 2))

    def loss(self, batch, step=0, ramp_steps=0):
        self._inverse_capture = True
        try:
            loss, metrics = super().loss(batch, step, ramp_steps)
        finally:
            self._inverse_capture = False
        observed = self._inverse_observed
        forward_inputs = self._inverse_forward_inputs
        del self._inverse_observed, self._inverse_forward_inputs
        target = batch["action"][:, :observed.size(1) - 1]
        if not torch.isfinite(target).all():
            raise ValueError("Every reconstructed action must be a valid executed forward input")
        origin, outcome = observed[:, :-1], observed[:, 1:]
        devices = [observed.device.index] if observed.is_cuda else []
        # Keep the original encoder/renderer/dropout random stream unchanged.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.inverse_seed + 1000003 * int(step))
            noise = torch.randn_like(target, dtype=torch.float32)
            time = torch.sigmoid(torch.randn_like(target[..., :1], dtype=torch.float32)
                                 + self.inverse_flow_shift)
            factual = self.inverse_per_sample(origin, outcome, target, noise, time)
            auxiliary = factual
            metrics["inverse_observed"] = factual.mean()
            if self.inverse_predicted:
                predicted = self.predict(origin, forward_inputs[:, :origin.size(1)])
                predicted_loss = self.inverse_per_sample(origin, predicted, target, noise, time)
                auxiliary = .5 * (factual + predicted_loss)
                metrics["inverse_predicted"] = predicted_loss.mean()
        loss = loss + self.inverse_weight * auxiliary.mean()
        metrics.update(loss=loss, inverse=auxiliary.mean(),
                       inverse_weight=loss.new_tensor(self.inverse_weight))
        return loss, metrics

    def audit_expected_gradients(self):
        return {name: "active" for name in
                ("encoder", "pool", "predictor", "action_encoder", "renderer", "inverse_head")}
