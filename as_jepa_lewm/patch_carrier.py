"""Native local1024 carrier with a frame-causal shared patch-token predictor.

Only the predictor is replaced; encode, raw carry, loss, renderer and goals
retain the original interfaces. Fixed mixing is internal to prediction.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM


def orthogonal_basis():
    """Exact previously declared signed/permuted Hadamard rule; private RNG."""
    matrix = np.ones((1, 1), dtype=np.float32)
    while len(matrix) < 1024:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    rng = np.random.default_rng(np.random.SeedSequence([20261031, 1024]))
    matrix = matrix[rng.permutation(1024)][:, rng.permutation(1024)]
    row = rng.choice(np.array([-1., 1.], dtype=np.float32), 1024)
    col = rng.choice(np.array([-1., 1.], dtype=np.float32), 1024)
    return torch.from_numpy(matrix * row[:, None] * col[None, :] / np.float32(32.))


def frame_mask(frames, device=None):
    index = torch.arange(frames, device=device).repeat_interleave(64)
    # SDPA boolean True means allowed. Plain is_causal=True would incorrectly
    # hide later spatial slots in the SAME frame.
    return index[None, :] <= index[:, None]


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.to_qkv = nn.Linear(64, 3 * 64, bias=False)
        self.to_out = nn.Sequential(nn.Linear(64, 64), nn.Dropout(.1))

    def forward(self, x, mask):
        b, t, s, d = x.shape
        x = self.norm(x).reshape(b, t*s, d)
        q, k, v = [a.reshape(b, t*s, 4, 16).transpose(1, 2)
                   for a in self.to_qkv(x).chunk(3, dim=-1)]
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                    dropout_p=.1 if self.training else 0., is_causal=False)
        return self.to_out(value.transpose(1, 2).reshape(b, t, s, d))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.mlp = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 256),
                nn.GELU(), nn.Dropout(.1), nn.Linear(256, 64), nn.Dropout(.1))
        self.norm1 = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(64, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(64, 6*64))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, condition, mask):
        # Once per FRAME, not64 redundant copies of the same linear operation.
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = [v.unsqueeze(2)
            for v in self.adaLN_modulation(condition).chunk(6, dim=-1)]
        x = x + gate_a*self.attn(self.norm1(x)*(1+scale_a)+shift_a, mask)
        return x + gate_m*self.mlp(self.norm2(x)*(1+scale_m)+shift_m)


class PatchTokenPredictor(nn.Module):
    """Same native predict input/output dimensions, returns raw residual only."""
    num_frames = 7
    input_dim = output_dim = 1024

    def __init__(self, basis="identity", seed=None):
        super().__init__()
        if basis not in {"identity", "orthogonal"}:
            raise ValueError(basis)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Explicit nonnegative predictor seed required")
        self.basis = basis
        with torch.random.fork_rng(devices=[]):
            # Seed only CPU; torch.manual_seed also reseeds CUDA generators,
            # which a devices=[] fork would not restore for a live GPU caller.
            torch.random.default_generator.manual_seed(seed)
            # Retain the original full temporal/spatial positional budget.
            self.pos_embedding = nn.Parameter(torch.randn(1, 7, 1024))
            self.input_proj = nn.Linear(16, 64)
            self.cond_proj = nn.Linear(1024, 64)
            self.layers = nn.ModuleList([Block() for _ in range(6)])
            self.norm = nn.LayerNorm(64)
            self.output_proj = nn.Linear(64, 16)
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        if basis == "orthogonal":
            self.register_buffer("coordinate_basis", orthogonal_basis())

    def forward(self, emb, act_emb):
        if (emb.ndim != 3 or emb.shape[-1] != 1024 or act_emb.shape != emb.shape
                or not 1 <= emb.shape[1] <= 7):
            raise ValueError("Require matching[B,T,1024],1<=T<=7; no padded future action")
        b, t, _ = emb.shape
        if self.basis == "orthogonal":
            # Match the old coordinate-control accuracy policy, not TF32.
            with torch.autocast(device_type=emb.device.type, enabled=False):
                coordinates = (emb.double() @ self.coordinate_basis.double()).to(emb.dtype)
        else:
            coordinates = emb
        x = self.input_proj((coordinates+self.pos_embedding[:, :t]).reshape(b, t, 64, 16))
        condition = self.cond_proj(act_emb)
        mask = frame_mask(t, emb.device)
        for block in self.layers:
            x = block(x, condition, mask)
        residual = self.output_proj(self.norm(x)).reshape(b, t, 1024)
        if self.basis == "orthogonal":
            with torch.autocast(device_type=emb.device.type, enabled=False):
                residual = (residual.double() @ self.coordinate_basis.double().T).to(residual.dtype)
        return residual


class PatchCarrierLeWM(SpatialCarrierLeWM):
    """Replace only native predictor, after constructing all original tensors.

    Private initialization preserves every nonpredictor tensor and outer RNG.
    The explicit seed is saved in model config, varies with training seed, and
    is identical for each paired identity/orthogonal experiment.
    """
    def __init__(self, patch_basis="identity", patch_predictor_seed=None, **kwargs):
        if (kwargs.get("spatial_mode", "local") != "local"
                or kwargs.get("spatial_basis", "identity") != "identity"
                or kwargs.get("spatial_chart", "none") != "none"
                or kwargs.get("spatial_readout_mode", "none") != "none"
                or kwargs.get("spatial_state_interface", "chart") != "chart"
                or kwargs.get("carry", True) is not True
                or kwargs.get("window", 8) != 8):
            raise ValueError("Patch gate requires unchanged local1024, window8, no chart/readout")
        if not isinstance(patch_predictor_seed, int) or isinstance(patch_predictor_seed, bool) or patch_predictor_seed < 0:
            raise ValueError("Explicit nonnegative predictor seed required in config")
        super().__init__(**kwargs)
        self.predictor = PatchTokenPredictor(basis=patch_basis, seed=patch_predictor_seed)
        self.patch_basis = patch_basis
        self.patch_predictor_seed = patch_predictor_seed

    def patch_parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            frozen=sum(p.numel() for p in self.parameters() if not p.requires_grad),
            predictor=sum(p.numel() for p in self.predictor.parameters()),
            action_encoder=sum(p.numel() for p in self.action_encoder.parameters()),
            basis_buffer_elements=(self.predictor.coordinate_basis.numel()
                                  if self.patch_basis == "orthogonal" else 0))

