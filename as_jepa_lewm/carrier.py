"""Conditional fixed-carrier experiment; native RobustLeWM loss is inherited.

The frozen random ViT and addressed RP64 match the prior spatial readout audit.
Invertible residual networks supply the bound; this is not a novelty claim.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.robust import RobustLeWM


PROJECTION_SEED = 20261030
CHART_SEED = 20261031


class ResidualCarrierChart(nn.Module):
    """T(u)=u+c A2 tanh(A1 u+b1)+b2 with certified matrix bounds.

For bounded/identity modes each A has operator norm <=1, from a full SVD,
not a power-iteration estimate. Hence the residual is at most c-Lipschitz.
The identity control freezes the exact identity initialization of this same net.
"""
    def __init__(self, width=64, hidden=64, mode="bounded", bound=.5):
        super().__init__()
        if mode not in {"identity", "unconstrained", "bounded"}:
            raise ValueError("Unknown carrier chart mode")
        if not math.isfinite(bound) or not 0 < bound < 1:
            raise ValueError("carrier_bound must lie strictly between zero and one")
        if width < 1 or hidden < 1:
            raise ValueError("Chart dimensions must be positive")
        self.mode, self.bound = mode, float(bound)
        # nn.Linear constructors consume CPU RNG; restore it and do not seed or
        # touch CUDA generators. The explicit private stream initializes W1.
        with torch.random.fork_rng(devices=[]):
            self.input = nn.Linear(width, hidden)
            self.output = nn.Linear(hidden, width)
        generator = torch.Generator(device="cpu").manual_seed(CHART_SEED)
        with torch.no_grad():
            # Ordinary nn.Linear initialization, from an isolated fixed stream.
            # Do not shrink W1 solely to make bounded/unconstrained gradients
            # identical: that could leave the actual constraint untested.
            nn.init.kaiming_uniform_(self.input.weight, a=math.sqrt(5), generator=generator)
            nn.init.uniform_(self.input.bias, -1/math.sqrt(width), 1/math.sqrt(width), generator=generator)
            self.output.weight.zero_()
            self.output.bias.zero_()
        if mode == "identity":
            self.requires_grad_(False)

    @staticmethod
    def bounded_weight(weight):
        # Full double-precision SVD gives the largest singular value directly.
        # A small FP32 rounding margin makes the implemented normalization
        # conservative; the theorem itself is stated in exact arithmetic.
        value = weight.float()
        norm = torch.linalg.matrix_norm(value.double(), ord=2)
        upper = (norm * (1 + 8 * torch.finfo(torch.float32).eps)).to(value.dtype)
        return value / upper.clamp_min(1)

    def effective_weights(self):
        if self.mode == "unconstrained":
            return self.input.weight.float(), self.output.weight.float()
        return self.bounded_weight(self.input.weight), self.bounded_weight(self.output.weight)

    def forward(self, value):
        # Do not let BF16 autocast silently change the matrix norm contract.
        with torch.autocast(device_type=value.device.type, enabled=False):
            first, second = self.effective_weights()
            hidden = torch.tanh(F.linear(value.float(), first, self.input.bias.float()))
            return value.float() + F.linear(hidden, self.bound * second, self.output.bias.float())

    @torch.no_grad()
    def norm_certificate(self):
        first, second = self.effective_weights()
        raw_one = float(torch.linalg.matrix_norm(self.input.weight.double(), ord=2))
        raw_two = float(torch.linalg.matrix_norm(self.output.weight.double(), ord=2))
        one = float(torch.linalg.matrix_norm(first.double(), ord=2))
        two = float(torch.linalg.matrix_norm(second.double(), ord=2))
        product = self.bound * one * two
        guard = 1 + 8*torch.finfo(torch.float32).eps
        return dict(raw_input_operator_norm=raw_one, raw_output_operator_norm=raw_two,
                    input_operator_norm=one, output_operator_norm=two,
                    input_normalization_active=self.mode != "unconstrained" and raw_one*guard > 1,
                    output_normalization_active=self.mode != "unconstrained" and raw_two*guard > 1,
                    residual_lipschitz_upper=product,
                    guaranteed_lower=max(0., 1-product),
                    guaranteed_upper=1+product, mode=self.mode,
                    global_invertibility_certified=product < 1)

    @torch.no_grad()
    def inverse(self, value, iterations=80):
        if self.mode == "unconstrained":
            raise ValueError("Unconstrained chart has no guaranteed contraction inverse")
        if iterations < 1:
            raise ValueError("Inverse iteration count must be positive")
        result = value.float().clone()
        for _ in range(iterations):
            result = value.float() - (self(result) - result)
        return result


class CarrierLeWM(RobustLeWM):
    """Same native TF objective/planner with fixed spatial-RP64 state carrier.

carrier_mean/scale are immutable training-only calibration constants, also
persisted in state_dict. Omit them only when calling raw_carrier to calibrate.
"""
    def __init__(self, carrier_chart="bounded", carrier_mean=None, carrier_scale=None,
                 carrier_bound=.5, carrier_hidden=64, **kwargs):
        if kwargs.get("mode", "L2") != "L2":
            raise ValueError("Carrier gate is defined only for AS-JEPA L2")
        if kwargs.get("state_slots", 4) * kwargs.get("state_width", 16) != 64:
            raise ValueError("Audited spatial carrier has exactly 64 coordinates")
        if kwargs.get("domain", "standard") != "standard":
            raise ValueError("Carrier calibration requires standard benchmark observations")
        if kwargs.get("batchnorm_state", False) or kwargs.get("batchnorm_projector", False):
            raise ValueError("Carrier normalization is fixed, never batch coupled")
        if (carrier_mean is None) != (carrier_scale is None):
            raise ValueError("Provide both carrier_mean and carrier_scale, or neither for calibration")
        super().__init__(**kwargs)
        self.encoder.requires_grad_(False)
        self.pool.requires_grad_(False)
        self.encoder.eval()
        self.pool.eval()
        width = int(self.encoder.config.hidden_size)
        generator = np.random.default_rng(np.random.SeedSequence([PROJECTION_SEED, 16 * width]))
        projection = (generator.standard_normal((16 * width, 64)) / math.sqrt(64)).astype(np.float32)
        self.register_buffer("carrier_projection", torch.from_numpy(projection))
        self._carrier_has_statistics = carrier_mean is not None
        mean = torch.zeros(64) if carrier_mean is None else torch.as_tensor(carrier_mean, dtype=torch.float32)
        scale = torch.ones(64) if carrier_scale is None else torch.as_tensor(carrier_scale, dtype=torch.float32)
        if mean.shape != (64,) or scale.shape != (64,):
            raise ValueError("Carrier calibration must contain 64 mean/scale values")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("Carrier statistics must be finite with strictly positive scales")
        self.register_buffer("carrier_mean", mean.clone())
        self.register_buffer("carrier_scale", scale.clone())
        self.carrier_chart_module = ResidualCarrierChart(64, carrier_hidden, carrier_chart, carrier_bound)
        self.carrier_chart = carrier_chart

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.pool.eval()
        return self

    @torch.no_grad()
    def raw_carrier(self, pixels):
        """Normalized model pixels [B,T,3,H,W] -> uncalibrated addressed RP64."""
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError("Carrier expects normalized [batch,time,3,height,width] pixels")
        b, w = pixels.shape[:2]
        # FP32 also matches the original frozen-feature audit/calibration path.
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            tokens = self.encoder(pixels.float().flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state
            patches = tokens[:, 1:].float()
            side = math.isqrt(patches.shape[1])
            if side * side != patches.shape[1] or side < 4:
                raise ValueError("Carrier expects CLS plus a square patch grid of side >=4")
            grid = patches.transpose(1, 2).reshape(b*w, patches.shape[-1], side, side)
            spatial = F.adaptive_avg_pool2d(grid, (4, 4)).permute(0, 2, 3, 1).flatten(1)
            return (spatial @ self.carrier_projection.float()).reshape(b, w, 64)

    def encode(self, info):
        if not self._carrier_has_statistics:
            raise RuntimeError("Prepare training-only carrier statistics before encode/training")
        raw = self.raw_carrier(info["pixels"])
        normalized = (raw-self.carrier_mean) / self.carrier_scale
        info["emb"] = self.carrier_chart_module(normalized)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def audit_expected_gradients(self):
        return {"encoder": "frozen", "pool": "frozen",
                "carrier_chart_module": "frozen" if self.carrier_chart == "identity" else "active",
                "predictor": "active", "renderer": "active", "action_encoder": "optional"}

    def carrier_parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    frozen=sum(p.numel() for p in self.parameters() if not p.requires_grad),
                    chart=sum(p.numel() for p in self.carrier_chart_module.parameters()),
                    chart_trainable=sum(p.numel() for p in self.carrier_chart_module.parameters() if p.requires_grad))
