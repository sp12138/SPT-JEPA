"""World-DiT primitives extracted from RepMiniWorld; no training lifecycle code."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

_MODES = ("adaln_lora", "fully_shared", "per_block")


def _build_prefix_block_causal_attn_mask(
    seq_len: int,
    tokens_per_frame: int,
    context_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Two-block causal mask for ``[clean context, noisy target]`` tokens.

    Both state blocks remain dense internally.  Target queries may attend to
    both blocks, while clean-context queries cannot read target keys.  This is
    deliberately different from a token-level lower-triangular mask, which
    would impose an artificial raster order on spatial tokens.
    """
    if seq_len <= 0 or tokens_per_frame <= 0 or context_frames <= 0:
        raise ValueError("prefix block-causal mask dimensions must be > 0")
    context_tokens = int(context_frames) * int(tokens_per_frame)
    if context_tokens >= seq_len:
        raise ValueError(
            "prefix context must leave at least one target token, got "
            f"context_tokens={context_tokens}, seq_len={seq_len}"
        )
    token_idx = torch.arange(seq_len, device=device)
    block_idx = (token_idx >= context_tokens).to(torch.long)
    mask = block_idx.unsqueeze(1) >= block_idx.unsqueeze(0)
    float_mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    float_mask.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return float_mask


def _build_full_history_teacher_forcing_attn_mask(
    seq_len: int,
    tokens_per_frame: int,
    state_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mask interleaved clean-content/noisy-query state pairs.

    The temporal blocks are ``[C0,Q1,C1,Q2,...,C(T-2),Q(T-1)]``.  A clean
    block ``Ci`` reads only clean blocks ``C0..Ci``.  A noisy query ``Qt``
    reads clean blocks ``C0..C(t-1)`` and itself, but no other query.  This
    exposes all next-state teacher-forcing targets in one forward without
    letting a later target consume a noisy rather than clean history state.
    """
    seq_len = int(seq_len)
    tokens_per_frame = int(tokens_per_frame)
    state_frames = int(state_frames)
    tokens_per_state = tokens_per_frame * state_frames
    if min(seq_len, tokens_per_frame, state_frames) <= 0:
        raise ValueError("teacher-forcing mask dimensions must be > 0")
    if seq_len % tokens_per_state:
        raise ValueError(
            "teacher-forcing token length must contain complete states, got "
            f"seq_len={seq_len}, tokens_per_state={tokens_per_state}"
        )
    num_blocks = seq_len // tokens_per_state
    if num_blocks < 2 or num_blocks % 2:
        raise ValueError(
            "teacher-forcing sequence must contain clean/query pairs, got "
            f"{num_blocks} state blocks"
        )

    block = torch.arange(num_blocks, device=device)
    is_clean = block.remainder(2).eq(0)
    state_index = torch.where(is_clean, block // 2, (block + 1) // 2)
    row_clean = is_clean[:, None]
    key_clean = is_clean[None, :]
    row_state = state_index[:, None]
    key_state = state_index[None, :]
    same_block = block[:, None].eq(block[None, :])
    allowed_blocks = torch.where(
        row_clean,
        key_clean & key_state.le(row_state),
        (key_clean & key_state.lt(row_state)) | same_block,
    )

    token_block = torch.arange(seq_len, device=device) // tokens_per_state
    allowed = allowed_blocks[token_block[:, None], token_block[None, :]]
    float_mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    float_mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    return float_mask


def modulate(x: Tensor, shift: Tensor | None, scale: Tensor) -> Tensor:
    """AdaLN modulation. ``shift`` / ``scale`` may be ``(B, D)`` or ``(B, N, D)``."""
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
        if shift is not None:
            shift = shift.unsqueeze(1)
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class PatchEmbed3D(nn.Module):
    """(B, C, T, H, W) -> (B, N, D) via a 3D conv patchifier."""

    def __init__(
        self,
        input_size: int | tuple[int, int, int],
        patch_size: int | tuple[int, int, int],
        in_chans: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            patch_size = (1, patch_size[0], patch_size[1])

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.num_patches = (
            (input_size[0] // patch_size[0])
            * (input_size[1] // patch_size[1])
            * (input_size[2] // patch_size[2])
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (B, D, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


def broadcat(tensors, dim: int = -1):
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors), strict=True))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims), "invalid broadcast dims"
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims), strict=True))
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes, strict=True)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast3D(nn.Module):
    """Axial 3D RoPE (time / height / width), borrowed from EVA/lightning_wm."""

    def __init__(
        self, dim: int, num_frames: int, frame_height: int, frame_width: int, theta: int = 10000
    ) -> None:
        super().__init__()
        dim_h = (dim // 3) // 2 * 2
        dim_w = (dim // 3) // 2 * 2
        dim_t = dim - dim_h - dim_w
        if dim_t % 2 != 0:
            dim_t -= 1

        freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

        self.register_buffer("base_freqs_t", freqs_t)
        self.register_buffer("base_freqs_h", freqs_h)
        self.register_buffer("base_freqs_w", freqs_w)
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.dim = dim
        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        self._num_frames = num_frames

        freqs_cos, freqs_sin = self._build_freqs(
            num_frames, freqs_t, freqs_h, freqs_w, frame_height, frame_width, dim
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    @staticmethod
    def _build_freqs(num_frames, freqs_t, freqs_h, freqs_w, fh, fw, dim, start_frame: int = 0):
        device = freqs_t.device
        t_time = torch.arange(
            start_frame, start_frame + num_frames, device=device, dtype=torch.float32
        )
        return VisionRotaryEmbeddingFast3D._build_freqs_at_positions(
            t_time, freqs_t, freqs_h, freqs_w, fh, fw, dim
        )

    @staticmethod
    def _build_freqs_at_positions(frame_positions, freqs_t, freqs_h, freqs_w, fh, fw, dim):
        device = freqs_t.device
        t_time = frame_positions.to(device=device, dtype=torch.float32).reshape(-1)
        num_frames = int(t_time.numel())
        t_height = torch.arange(fh, device=device, dtype=torch.float32)
        t_width = torch.arange(fw, device=device, dtype=torch.float32)
        ft = repeat(torch.einsum("n,d->nd", t_time, freqs_t), "... n -> ... (n r)", r=2)
        fht = repeat(torch.einsum("n,d->nd", t_height, freqs_h), "... n -> ... (n r)", r=2)
        fwt = repeat(torch.einsum("n,d->nd", t_width, freqs_w), "... n -> ... (n r)", r=2)
        freqs = broadcat(
            (ft.view(num_frames, 1, 1, -1), fht.view(1, fh, 1, -1), fwt.view(1, 1, fw, -1)), dim=-1
        )
        return freqs.cos().view(-1, dim), freqs.sin().view(-1, dim)

    def _get_freqs(self, num_frames: int, device: torch.device, start_frame: int = 0):
        if start_frame == 0 and num_frames == self._num_frames:
            return self.freqs_cos, self.freqs_sin
        return self._build_freqs(
            num_frames,
            self.base_freqs_t.to(device),
            self.base_freqs_h.to(device),
            self.base_freqs_w.to(device),
            self.frame_height,
            self.frame_width,
            self.dim,
            start_frame=start_frame,
        )

    def forward(
        self,
        t: Tensor,
        num_frames_override: int | None = None,
        start_frame: int = 0,
        frame_position_ids: Tensor | None = None,
        token_slice: tuple[int, int] | None = None,
        token_position_ids: Tensor | None = None,
    ) -> Tensor:
        num_frames = num_frames_override if num_frames_override is not None else self._num_frames
        if frame_position_ids is None:
            cos, sin = self._get_freqs(num_frames, t.device, start_frame=start_frame)
        else:
            if int(start_frame) != 0:
                raise ValueError("custom RoPE frame positions require start_frame=0")
            if frame_position_ids.dim() != 1 or frame_position_ids.numel() != num_frames:
                raise ValueError(
                    "frame_position_ids must have one entry per temporal frame, "
                    f"got {tuple(frame_position_ids.shape)} for T={num_frames}"
                )
            cos, sin = self._build_freqs_at_positions(
                frame_position_ids,
                self.base_freqs_t.to(t.device),
                self.base_freqs_h.to(t.device),
                self.base_freqs_w.to(t.device),
                self.frame_height,
                self.frame_width,
                self.dim,
            )
        if token_position_ids is not None:
            if token_slice is not None:
                raise ValueError("Choose a token slice or explicit token positions")
            if (
                token_position_ids.dtype != torch.long
                or token_position_ids.shape != (t.shape[0], t.shape[-2])
                or token_position_ids.device != t.device
            ):
                raise ValueError("RoPE token positions must be int64 [B,N] on the input device")
            if bool(((token_position_ids < 0) | (token_position_ids >= cos.shape[0])).any()):
                raise ValueError("RoPE token position outside the original grid")
            cos, sin = cos[token_position_ids].unsqueeze(1), sin[token_position_ids].unsqueeze(1)
        if token_slice is not None:
            start, length = token_slice
            cos, sin = cos.narrow(0, start, length), sin.narrow(0, start, length)
        return t * cos + rotate_half(t) * sin

    def rope_shift_time(self, delta: int, cached: Tensor) -> Tensor:
        """Re-rotate cached K/Q on the temporal axis by ``delta`` frames.

        Given a tensor originally RoPE-rotated at temporal positions
        ``[p, ..., p+N-1]``, returns it rotated as if positions were
        ``[p+delta, ..., p+delta+N-1]``. Use ``delta=-k`` after evicting ``k``
        leading frames from a streaming cache to renumber positions back to 0.
        Only the temporal slice (first ``dim_t`` dims) is rotated; spatial dims
        receive identity rotation.
        """
        if delta == 0 or cached.numel() == 0:
            return cached
        device = cached.device
        out_dtype = cached.dtype

        base_freqs_t = self.base_freqs_t.to(device=device, dtype=torch.float32)
        angle_t = float(delta) * base_freqs_t
        angle_t_rep = repeat(angle_t, "n -> (n r)", r=2)
        cos_t = angle_t_rep.cos()
        sin_t = angle_t_rep.sin()

        rest = self.dim - self.dim_t
        cos_rest = torch.ones(rest, device=device, dtype=torch.float32)
        sin_rest = torch.zeros(rest, device=device, dtype=torch.float32)

        cos_full = torch.cat([cos_t, cos_rest], dim=-1).to(out_dtype)
        sin_full = torch.cat([sin_t, sin_rest], dim=-1).to(out_dtype)

        return cached * cos_full + rotate_half(cached) * sin_full


class TimestepEmbedder(nn.Module):
    """Scalar timestep -> D-dim vector (sinusoidal + MLP)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            return self.mlp(self.timestep_embedding(t, self.freq_dim))
        if t.dim() == 2:
            b, s = t.shape
            emb = self.mlp(self.timestep_embedding(t.reshape(-1), self.freq_dim))
            return emb.view(b, s, -1)
        raise ValueError(f"Unsupported timestep shape: {t.shape}")


class BlockModulation(nn.Module):
    """Per-block modulation producer, respecting the model-wide ``adaln_mode``.

    * ``adaln_lora``   : ``shared_mod + lora(emb)`` where ``lora = SiLU -> D->r
      -> r->MD`` (zero-init, so a block starts exactly at ``shared_mod``).
    * ``fully_shared`` : ``shared_mod`` (no per-block params; FLUX.2 style).
    * ``per_block``    : ``full(emb)`` with ``full = SiLU -> D->MD`` (classic
      per-block AdaLN, zero-init).
    """

    def __init__(
        self, hidden_size: int, adaln_mode: str, n_mod_chunks: int = 6, lora_dim: int = 256
    ) -> None:
        super().__init__()
        assert adaln_mode in _MODES, f"adaln_mode must be one of {_MODES}"
        self.adaln_mode = adaln_mode
        out = n_mod_chunks * hidden_size
        if adaln_mode == "adaln_lora":
            self.lora = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, lora_dim, bias=False),
                nn.Linear(lora_dim, out, bias=False),
            )
        elif adaln_mode == "per_block":
            self.full = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, out, bias=True))
        # fully_shared: no parameters

    def forward(self, emb: Tensor, shared_mod: Tensor | None, pose_mod: Tensor | None) -> Tensor:
        if self.adaln_mode == "adaln_lora":
            mod = shared_mod + self.lora(emb)
        elif self.adaln_mode == "fully_shared":
            mod = shared_mod
        else:  # per_block
            mod = self.full(emb)
        if pose_mod is not None:
            mod = mod + pose_mod
        return mod


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        cond = self.adaLN_modulation(emb)
        if cond.dim() == 2:
            shift, scale = cond.chunk(2, dim=1)
        else:
            shift, scale = cond.chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))
