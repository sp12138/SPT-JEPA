"""Normal-map rotation and deferred spatial preprocessing.

Two stages that both work on normal-map pixels, kept in one module because they
are always configured together and the second one has to know what the first
one deferred.

The first stage rotates a normal clip in SO(3), per sample, inside the loader
worker. A normal video stores a unit vector field RGB-encoded as
``n = 2 * rgb - 1``. Rotating that field is a legal augmentation -- the result
is still a unit normal field, just with systematically turned colors -- and it
widens the color distribution the model sees without touching the RGB target.

Only the training-time augmentation path is ported here. Taichu's module also
carries an environment-variable-driven ``NormalRotator`` for inference-time
injection and a batched ``rotate_normal_condition``; neither has a caller on
the dataset side, so porting them would be dead code.

Every sampler here draws from the **global** torch RNG rather than owning a
Generator. Worker seeding is by ``(dp_rank, worker_id)``, so SP peers -- which
share a ``dp_rank`` -- draw the same matrix and stay in lockstep. A private
Generator would silently break that.

The second stage moves the expensive spatial work off the workers entirely.
Decoding a gameplay window inside a loader worker is cheap; resizing it is not.
At 149x3x720x1280 the fp32 intermediate a cover-then-crop needs is ~1.6 GB, and
a few workers in flight exhaust host memory long before the GPU is saturated.
So the workers hand back the raw ``uint8`` frames they decoded, and everything
after the decode happens once, on the main rank's device, for the whole batch.
The wire payload shrinks by 4x on top of that, since uint8 crosses the worker
boundary instead of bfloat16.

Metadata the main rank needs -- target geometry, whether a normal was found,
the rotation matrix that was drawn but not applied -- travels as private marker
keys. :func:`preprocess_deferred_batch` consumes every one of them, so none
reach the encoder.

The deferred operations are ordered to match taichu's
``taichu/data/deferred_spatial.py`` bit for bit. That is a hard constraint, not
a stylistic one: taichu's published latent caches were written by that code, and
vforge has to extend them rather than start over. Three details are load-bearing
and must not be tidied up -- fp32 is held until the final cast, the normal is
rotated before the mask is blended in, and the norm is taken over ``dim=2``
because the tensors are rank 5 there. ``tests/data/test_preprocess_parity.py``
imports taichu's module directly and compares the two implementations byte for
byte.

The two stages keep separate rotation implementations on purpose: the worker
path takes a rank-4 clip and one matrix, the deferred path takes a rank-5 batch
and one matrix per item. Folding them together would put a reshape in the way of
the bit-parity constraint above.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


def rotation_matrix_axis_angle(
    axis: Sequence[float],
    angle_deg: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rodrigues' formula: rotate ``angle_deg`` about ``axis``, as ``[3, 3]``."""
    vector = torch.tensor(list(axis), dtype=dtype)
    norm = torch.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError(f"rotation axis must be non-zero, got {list(axis)}")
    vector = vector / norm
    theta = math.radians(angle_deg)
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    cos, sin = math.cos(theta), math.sin(theta)
    versine = 1.0 - cos
    return torch.tensor(
        [
            [cos + x * x * versine, x * y * versine - z * sin, x * z * versine + y * sin],
            [y * x * versine + z * sin, cos + y * y * versine, y * z * versine - x * sin],
            [z * x * versine - y * sin, z * y * versine + x * sin, cos + z * z * versine],
        ],
        dtype=dtype,
    )


def random_rotation_matrix(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Uniform sample from SO(3) as ``[3, 3]``, via Shoemake's quaternion method.

    Determinant is always +1, so this never produces a reflection.
    """
    u1, u2, u3 = (float(value) for value in torch.rand(3))
    q1 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q2 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    w, x, y, z = q4, q1, q2, q3
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=dtype,
    )


def random_axis(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Uniform direction on the unit sphere as ``[3]``.

    Normalizing a 3D Gaussian is what makes this uniform; sampling angles
    directly would cluster near the poles.
    """
    vector = torch.randn(3)
    norm = torch.linalg.norm(vector)
    if norm < 1e-8:
        return torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    return (vector / norm).to(dtype)


def rotate_normal_frames(normal_tchw: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a ``[T, 3, H, W]`` normal clip in ``[0, 1]``, same shape out.

    Decode, rotate, and re-encode fold into a single affine map::

        n = 2 * rgb - 1;  n' = R @ n;  rgb' = (n' + 1) / 2
        =>  rgb' = R @ rgb + (1 - R @ [1, 1, 1]) / 2

    Two properties fall out of ``R`` being orthogonal, and both are why there is
    no per-pixel renormalize here: unit vectors stay unit, and neutral grey 0.5
    is a fixed point, so missing or occluded regions stay grey instead of being
    turned into colored noise.
    """
    if normal_tchw.dim() != 4 or normal_tchw.shape[1] != 3:
        raise ValueError(f"expected a [T, 3, H, W] normal clip, got {tuple(normal_tchw.shape)}")

    work_dtype = torch.float32
    rotation = rotation.to(dtype=work_dtype)
    bias = (1.0 - rotation @ torch.ones(3, dtype=work_dtype)) * 0.5

    frames, _, height, width = normal_tchw.shape
    flat = normal_tchw.to(work_dtype).reshape(frames, 3, height * width)
    rotated = torch.einsum("ij,tjk->tik", rotation, flat)
    rotated = rotated + bias.view(1, 3, 1)
    # Float error can push a component a hair outside [0, 1]; clamp is a
    # numerical guard, not a correction of the math above.
    rotated = rotated.reshape(frames, 3, height, width).clamp_(0.0, 1.0)
    return rotated.to(normal_tchw.dtype)


class NormalRotationAugmentor:
    """Draw one rotation per sample and apply it to the whole clip.

    The rotation is constant across time on purpose: a per-frame draw would
    make the normal condition flicker, which is a different (and unwanted)
    augmentation.

    Args:
        prob: Probability of augmenting a sample. The rest pass through
            untouched, which keeps clean normals in the mix.
        max_angle_deg: Cap for the limited mode -- axis uniform on the sphere,
            angle uniform in ``[0, max]``. Ignored when ``full_so3`` is set.
        full_so3: Sample the whole rotation group instead of capping the angle.
            A non-positive ``max_angle_deg`` also forces this, since capping at
            zero would make the augmentation a no-op.
    """

    def __init__(
        self,
        prob: float = 1.0,
        max_angle_deg: float = 0.0,
        full_so3: bool = True,
    ) -> None:
        self.prob = float(prob)
        self.max_angle_deg = float(max_angle_deg)
        self.full_so3 = bool(full_so3) or self.max_angle_deg <= 0.0

    def sample_matrix(self) -> torch.Tensor:
        """Draw one rotation matrix from the configured distribution."""
        if self.full_so3:
            return random_rotation_matrix()
        return rotation_matrix_axis_angle(
            random_axis(),
            float(torch.rand(())) * self.max_angle_deg,
        )

    def should_apply(self) -> bool:
        """Draw the per-sample coin. Consumes global RNG only when ``prob < 1``."""
        return self.prob >= 1.0 or float(torch.rand(())) < self.prob

    def __call__(self, normal_tchw: torch.Tensor) -> torch.Tensor:
        if not self.should_apply():
            return normal_tchw
        return rotate_normal_frames(normal_tchw, self.sample_matrix())

    def describe(self) -> str:
        if self.full_so3:
            return f"full-SO3 random, prob={self.prob}"
        return f"axis-uniform, angle<={self.max_angle_deg}deg, prob={self.prob}"


DEFERRED_SPATIAL_KEY = "__prepare_deferred_spatial__"
TARGET_HEIGHT_KEY = "__prepare_target_height__"
TARGET_WIDTH_KEY = "__prepare_target_width__"
NORMAL_MISSING_KEY = "__prepare_normal_missing__"
NORMAL_ROTATION_KEY = "__prepare_normal_rotation__"
RAW_NORMAL_MASK_KEY = "__prepare_raw_normal_mask__"

#: Every private key, for the collate to route and the tests to assert against.
MARKER_KEYS = (
    DEFERRED_SPATIAL_KEY,
    TARGET_HEIGHT_KEY,
    TARGET_WIDTH_KEY,
    NORMAL_MISSING_KEY,
    NORMAL_ROTATION_KEY,
    RAW_NORMAL_MASK_KEY,
)

#: Marker keys holding one scalar for the whole batch rather than a tensor.
SCALAR_MARKER_KEYS = (
    DEFERRED_SPATIAL_KEY,
    TARGET_HEIGHT_KEY,
    TARGET_WIDTH_KEY,
    NORMAL_MISSING_KEY,
)


def has_deferred_spatial(batch: dict[str, Any]) -> bool:
    """Whether ``batch`` still needs the spatial stage run over it."""
    value = batch.get(DEFERRED_SPATIAL_KEY)
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else False
    return bool(value)


def _scalar_int(value: Any, name: str) -> int:
    """Read a batch-wide int that may have been collated into a 1-element tensor."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one value, got {tuple(value.shape)}")
        return int(value.item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must contain one value, got {value!r}")
        return int(value[0])
    return int(value)


def _scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else False
    return bool(value)


def _resize_and_center_crop(
    raw_btchw: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Scale ``[B, T, C, H, W]`` uint8 to cover the target, then center crop it.

    The scale is ``max`` of the two ratios so the target is covered and the
    overflow is trimmed; a ``min`` would letterbox, and the camera intrinsics
    adjustment on the dataset side assumes exactly this geometry.
    """
    if raw_btchw.ndim != 5:
        raise ValueError(
            f"deferred spatial input must be [B,T,C,H,W], got {tuple(raw_btchw.shape)}"
        )
    if raw_btchw.dtype != torch.uint8:
        raise TypeError(f"deferred spatial input must be uint8, got {raw_btchw.dtype}")

    batch, frames, channels, height, width = raw_btchw.shape
    scale = max(target_h / height, target_w / width)
    new_h = max(int(round(height * scale)), target_h)
    new_w = max(int(round(width * scale)), target_w)

    # The host-to-device copy carries uint8 and the widening happens on the
    # device, which is a quarter of the traffic of converting first.
    work = raw_btchw.to(device=device, non_blocking=True)
    work = work.reshape(batch * frames, channels, height, width)
    work = work.to(torch.float32).mul_(1.0 / 255.0)
    work = F.interpolate(
        work,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    work = work[:, :, top : top + target_h, left : left + target_w].contiguous()
    return work.reshape(batch, frames, channels, target_h, target_w)


def _normalize_normals(normal: torch.Tensor) -> torch.Tensor:
    """Renormalize an RGB-encoded normal clip in place, staying in ``[0, 1]``.

    Interpolation blends neighboring normals, which shortens the vectors, so a
    resized normal map is no longer unit length. ``dim=2`` is the channel axis
    of the rank-5 batch this stage works on.
    """
    normal.mul_(2.0).sub_(1.0)
    norm = torch.linalg.norm(normal, dim=2, keepdim=True).clamp_(min=1e-6)
    normal.div_(norm)
    return normal.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)


def _rotate_normals(normal: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Apply one RGB-space rotation per batch item to a ``[B, T, 3, H, W]`` clip.

    Decode, rotate, and re-encode collapse into a single affine map, so there is
    no per-pixel renormalize here::

        n = 2 * rgb - 1;  n' = R @ n;  rgb' = (n' + 1) / 2
        =>  rgb' = R @ rgb + (1 - R @ [1, 1, 1]) / 2
    """
    rotations = rotations.to(device=normal.device, dtype=torch.float32)
    if rotations.ndim == 2:
        rotations = rotations.unsqueeze(0)
    if rotations.shape != (normal.shape[0], 3, 3):
        raise ValueError(
            f"normal rotation must be [B,3,3], got {tuple(rotations.shape)} "
            f"for batch {normal.shape[0]}"
        )
    bias = (
        1.0
        - torch.matmul(
            rotations,
            torch.ones(normal.shape[0], 3, 1, device=normal.device, dtype=torch.float32),
        ).squeeze(-1)
    ) * 0.5
    rotated = torch.einsum("bij,btjhw->btihw", rotations, normal)
    return rotated.add_(bias[:, None, :, None, None]).clamp_(0.0, 1.0)


def preprocess_deferred_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    output_dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """Materialize encoder-ready pixels from a deferred batch, in place.

    Returns ``True`` when the batch was deferred and has now been processed, and
    ``False`` when it was an ordinary batch that needed nothing -- so callers can
    invoke this unconditionally ahead of the encoder.
    """
    if not has_deferred_spatial(batch):
        return False
    if device.type != "cuda" and device.type != "cpu":
        raise ValueError(f"deferred spatial preprocessing needs a CUDA/CPU device, got {device}")

    try:
        target_h = _scalar_int(batch.pop(TARGET_HEIGHT_KEY), TARGET_HEIGHT_KEY)
        target_w = _scalar_int(batch.pop(TARGET_WIDTH_KEY), TARGET_WIDTH_KEY)
    except KeyError as exc:
        raise KeyError(f"deferred batch is missing required metadata {exc.args[0]!r}") from exc

    raw_video = batch["video_frames"]
    video = _resize_and_center_crop(raw_video, target_h=target_h, target_w=target_w, device=device)

    normal_missing = _scalar_bool(batch.pop(NORMAL_MISSING_KEY, False))
    raw_normal = batch.get("normal_condition")
    if normal_missing:
        # Built at full precision rather than decoded from uint8 127 or 128, so
        # the neutral normal is exactly 0.5 and every rotation fixes it.
        normal = torch.full_like(video, 0.5)
    else:
        if not isinstance(raw_normal, torch.Tensor) or raw_normal.numel() == 0:
            raise ValueError("deferred batch declares a normal stream but provides no pixels")
        normal = _resize_and_center_crop(
            raw_normal, target_h=target_h, target_w=target_w, device=device
        )
        normal = _normalize_normals(normal)

    rotations = batch.pop(NORMAL_ROTATION_KEY, None)
    if rotations is not None:
        normal = _rotate_normals(normal, rotations)

    raw_mask = batch.pop(RAW_NORMAL_MASK_KEY, None)
    normal_fg = None
    normal_bg = None
    if raw_mask is not None:
        mask = _resize_and_center_crop(
            raw_mask, target_h=target_h, target_w=target_w, device=device
        )
        if mask.shape[2] > 1:
            mask = mask.mean(dim=2, keepdim=True)
        mask.clamp_(0.0, 1.0)
        inv_mask = 1.0 - mask
        normal_fg = (normal * mask).add_(inv_mask * 0.5).clamp_(0.0, 1.0)
        normal_bg = (normal * inv_mask).add_(mask * 0.5).clamp_(0.0, 1.0)

    video = video.to(output_dtype)
    normal = normal.to(output_dtype)
    batch["video_frames"] = video
    batch["image_condition"] = video[:, 0:1]
    batch["normal_condition"] = normal
    if normal_fg is not None:
        batch["normal_fg_condition"] = normal_fg.to(output_dtype)
        batch["normal_bg_condition"] = normal_bg.to(output_dtype)
    batch.pop(DEFERRED_SPATIAL_KEY, None)
    return True


__all__ = [
    "DEFERRED_SPATIAL_KEY",
    "MARKER_KEYS",
    "NORMAL_MISSING_KEY",
    "NORMAL_ROTATION_KEY",
    "NormalRotationAugmentor",
    "RAW_NORMAL_MASK_KEY",
    "SCALAR_MARKER_KEYS",
    "TARGET_HEIGHT_KEY",
    "TARGET_WIDTH_KEY",
    "has_deferred_spatial",
    "preprocess_deferred_batch",
    "random_axis",
    "random_rotation_matrix",
    "rotate_normal_frames",
    "rotation_matrix_axis_angle",
]
