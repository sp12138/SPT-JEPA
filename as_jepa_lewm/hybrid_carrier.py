"""Fixed64 + fixed/adaptive QueryPool64 mechanism control, native TF loss.

Both arms share one frozen random ViT. Only the QueryPool output stop-gradient differs.
The second channel is not assumed independent, orthogonal, or complementary.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from as_jepa_lewm.carrier import PROJECTION_SEED
from as_jepa_lewm.robust import RobustLeWM


class HybridCarrierLeWM(RobustLeWM):
    def __init__(self, hybrid_query="fixed", hybrid_fixed_mean=None,
                 hybrid_fixed_scale=None, hybrid_query_mean=None,
                 hybrid_query_scale=None, state_slots=8, state_width=16, **kwargs):
        if hybrid_query not in {"fixed", "adaptive"}:
            raise ValueError("hybrid_query must be fixed or adaptive")
        if (state_slots, state_width) != (8, 16):
            raise ValueError("Hybrid state is exactly fixed64 + QueryPool64 = 8 x 16")
        if kwargs.get("mode", "L2") != "L2" or kwargs.get("domain", "standard") != "standard":
            raise ValueError("Hybrid gate requires standard benchmark L2")
        if kwargs.get("encoder_size", 224) != 224 or kwargs.get("pool_partition", "full") != "full":
            raise ValueError("Require the original224 encoder and full QueryPool")
        if kwargs.get("batchnorm_state", False) or kwargs.get("batchnorm_projector", False):
            raise ValueError("Hybrid calibration is fixed and sample independent")
        if kwargs.get("response_query_mode", "none") != "none":
            raise ValueError("No auxiliary query head in this gate")
        values = (hybrid_fixed_mean, hybrid_fixed_scale, hybrid_query_mean, hybrid_query_scale)
        if any(x is None for x in values) and not all(x is None for x in values):
            raise ValueError("Provide all four calibration constants, or none for calibration")
        # Recover the exact original64-D pool from the incoming CPU RNG state.
        # Its initialization follows the original64-D predictor/action modules,
        # whose RNG consumption differs from the new128-D modules. The temporary
        # reference is discarded; its construction cannot move the outer stream.
        with torch.random.fork_rng(devices=[]):
            reference = RobustLeWM(state_slots=4, state_width=16, **kwargs)
            original_pool = reference.pool
            del reference
        super().__init__(state_slots=state_slots, state_width=state_width, **kwargs)
        self.pool = original_pool
        self.hybrid_query = hybrid_query
        self.encoder.requires_grad_(False)
        # Identical differentiable forward in both arms. Merely changing a
        # parameter's requires_grad can change PyTorch's forward implementation.
        # The fixed arm detaches only the output; native AdamW therefore sees
        # grad=None and creates no state or updates for these parameters.
        self.pool.requires_grad_(True)
        self.encoder.eval()
        self.pool.eval()
        width = int(self.encoder.config.hidden_size)
        generator = np.random.default_rng(np.random.SeedSequence([PROJECTION_SEED, 16 * width]))
        projection = (generator.standard_normal((16 * width, 64)) / math.sqrt(64)).astype(np.float32)
        self.register_buffer("hybrid_projection", torch.from_numpy(projection))
        self._hybrid_has_statistics = hybrid_fixed_mean is not None
        for name, mean_value, scale_value in (
            ("fixed", hybrid_fixed_mean, hybrid_fixed_scale),
            ("query", hybrid_query_mean, hybrid_query_scale),
        ):
            mean = torch.zeros(64) if mean_value is None else torch.as_tensor(mean_value, dtype=torch.float32)
            scale = torch.tensor(1.) if scale_value is None else torch.as_tensor(scale_value, dtype=torch.float32)
            if mean.shape != (64,) or scale.ndim != 0:
                raise ValueError("Each channel requires64 coordinate means and ONE scalar RMS")
            if not bool(torch.isfinite(mean).all() and torch.isfinite(scale) and scale > 0):
                raise ValueError("Calibration must be finite with strictly positive scalar RMS")
            self.register_buffer("hybrid_" + name + "_mean", mean.clone())
            self.register_buffer("hybrid_" + name + "_scale", scale.clone())

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        # QueryPool has no BatchNorm, MHA dropout=0, and frame-local LayerNorm.
        # Its eval mode does not disable gradients in the adaptive arm.
        self.pool.eval()
        return self

    def raw_channels(self, pixels):
        if pixels.ndim != 5 or pixels.shape[2:] != (3, 224, 224):
            raise ValueError("Expected normalized[B,T,3,224,224] pixels")
        b, w = pixels.shape[:2]
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            with torch.no_grad():
                tokens = self.encoder(pixels.float().flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state
                patches = tokens[:, 1:].float()
                if patches.shape[1:] != (256, 192):
                    raise ValueError("Expected CLS plus16x16 patches of width192")
                grid = patches.transpose(1, 2).reshape(b*w, 192, 16, 16)
                spatial = F.adaptive_avg_pool2d(grid, (4, 4)).permute(0, 2, 3, 1).flatten(1)
                fixed = (spatial @ self.hybrid_projection.float()).reshape(b, w, 64)
            # Outside no_grad: trainable QueryPool sees the same detached tokens.
            query = self.pool(tokens.float()).reshape(b, w, 64)
            if self.hybrid_query == "fixed":
                query = query.detach()
        return fixed, query

    def encode(self, info):
        if not self._hybrid_has_statistics:
            raise RuntimeError("Prepare original128 training-frame calibration before encode/training")
        fixed, query = self.raw_channels(info["pixels"])
        fixed = (fixed-self.hybrid_fixed_mean) / self.hybrid_fixed_scale
        query = (query-self.hybrid_query_mean) / self.hybrid_query_scale
        info["emb"] = torch.cat((fixed, query), dim=-1)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def audit_expected_gradients(self):
        return dict(encoder="frozen", pool="active" if self.hybrid_query == "adaptive" else "frozen",
                    predictor="active", renderer="active", action_encoder="optional")

    def hybrid_parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    frozen_by_requires_grad=sum(p.numel() for p in self.parameters() if not p.requires_grad),
                    effective_trainable=sum(p.numel() for n, p in self.named_parameters()
                        if p.requires_grad and not (self.hybrid_query == "fixed" and n.startswith("pool."))),
                    query_pool=sum(p.numel() for p in self.pool.parameters()),
                    query_pool_effective_trainable=(sum(p.numel() for p in self.pool.parameters())
                        if self.hybrid_query == "adaptive" else 0),
                    fixed_projection_elements=self.hybrid_projection.numel())
