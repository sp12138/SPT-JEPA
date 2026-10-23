"""Gameplay camera-to-grid rendering migrated from Taichu.

The renderer consumes camera-to-world poses and pixel-space intrinsics and
produces an RGB grid video in ``[T, 3, H, W]`` and ``[0, 1]``.  It is model
input semantics rather than a dataset transport, so it lives with Wan Taichu.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _periodic_delta(value: torch.Tensor, anchor: torch.Tensor, spacing: float) -> torch.Tensor:
    return torch.remainder(value - anchor + spacing * 0.5, spacing) - spacing * 0.5


def _make_first_camera_reference(c2ws: torch.Tensor) -> torch.Tensor:
    reference_w2c = torch.linalg.inv(c2ws[0])
    relative = reference_w2c.unsqueeze(0) @ c2ws
    relative[0] = torch.eye(4, device=c2ws.device, dtype=c2ws.dtype)
    return relative


def _smooth_line_alpha(distance: torch.Tensor, half_width: torch.Tensor) -> torch.Tensor:
    feather = torch.clamp(half_width * 0.75, min=1.0e-4)
    return torch.clamp((half_width + feather - distance) / feather, 0.0, 1.0)


def _render_frame_impl(
    c2w: torch.Tensor,
    k: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    spacing: float,
    grid_anchor_x: torch.Tensor,
    grid_anchor_z: torch.Tensor,
    pillar_spacing: float,
    pillar_anchor_x: torch.Tensor,
    pillar_anchor_z: torch.Tensor,
    y_anchor: torch.Tensor,
    room_height: float,
    line_width_m: float,
    near_m: float,
    far_m: float,
    step_offsets: torch.Tensor,
    room_index_override: torch.Tensor | None,
) -> torch.Tensor:
    origin = c2w[:3, 3]
    rotation = c2w[:3, :3]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    if room_index_override is None:
        room_index = torch.floor((origin[1] - y_anchor) / room_height)
    else:
        room_index = room_index_override
    floor_y = y_anchor + room_index * room_height
    ceiling_y = floor_y + room_height
    center_y = (floor_y + ceiling_y) * 0.5
    half_height = room_height * 0.5

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs_cam = torch.stack([(xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)], dim=-1)
    dirs_world = dirs_cam @ rotation.T
    dirs_world = dirs_world / torch.linalg.vector_norm(
        dirs_world,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)

    inv_min_focal = 1.0 / torch.minimum(fx, fy)
    thickness = 2.0
    line_half = max(line_width_m * 0.5 * thickness, 1.0e-4)
    footprint_scale = 0.85 * thickness

    best_t = torch.full_like(dirs_world[..., 0], float("inf"))
    color = torch.zeros_like(dirs_world)
    alpha = torch.zeros_like(dirs_world[..., 0])

    for plane_y in (floor_y, ceiling_y):
        denom = dirs_world[..., 1]
        safe_denom = torch.where(
            torch.abs(denom) > 1.0e-7,
            denom,
            torch.full_like(denom, 1.0e-7),
        )
        t = (plane_y - origin[1]) / safe_denom
        valid = (torch.abs(denom) > 1.0e-7) & (t > near_m) & (t < far_m)
        hit = origin + dirs_world * t[..., None]
        d_x = torch.abs(_periodic_delta(hit[..., 0], grid_anchor_x, spacing))
        d_z = torch.abs(_periodic_delta(hit[..., 2], grid_anchor_z, spacing))
        pixel_footprint = t.clamp_min(0) * inv_min_focal
        half_width = torch.maximum(
            torch.full_like(t, line_half),
            pixel_footprint * footprint_scale,
        )
        red_a = _smooth_line_alpha(d_z, half_width)
        blue_a = _smooth_line_alpha(d_x, half_width)
        plane_a = torch.maximum(red_a, blue_a)
        plane_rgb = torch.stack([red_a, torch.zeros_like(red_a), blue_a], dim=-1)
        plane_rgb = plane_rgb / plane_a[..., None].clamp_min(1.0)
        take = valid & (plane_a > 0) & (t < best_t)
        best_t = torch.where(take, t, best_t)
        color = torch.where(take[..., None], plane_rgb, color)
        alpha = torch.where(take, plane_a, alpha)

    ox, oy, oz = origin[0], origin[1], origin[2]
    dx_w, dy_w, dz_w = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
    horiz_sq = dx_w.square() + dz_w.square()
    horiz_safe = horiz_sq.clamp_min(1.0e-12)

    ox_rel = ox - pillar_anchor_x
    oz_rel = oz - pillar_anchor_z

    marches = step_offsets[:, 0].view(-1, 1, 1)
    d_ix_offsets = step_offsets[:, 1].view(-1, 1, 1)
    d_iz_offsets = step_offsets[:, 2].view(-1, 1, 1)

    dx_wu = dx_w.unsqueeze(0)
    dz_wu = dz_w.unsqueeze(0)
    dy_wu = dy_w.unsqueeze(0)
    horiz_safe_u = horiz_safe.unsqueeze(0)
    horiz_sq_u = horiz_sq.unsqueeze(0)

    base_ix = torch.round((ox_rel + dx_wu * marches) / pillar_spacing)
    base_iz = torch.round((oz_rel + dz_wu * marches) / pillar_spacing)
    ix = base_ix + d_ix_offsets
    iz = base_iz + d_iz_offsets

    pillar_x = pillar_anchor_x + ix * pillar_spacing
    pillar_z = pillar_anchor_z + iz * pillar_spacing
    px_rel = pillar_x - ox
    pz_rel = pillar_z - oz

    t_star = (px_rel * dx_wu + pz_rel * dz_wu) / horiz_safe_u
    dist_x = px_rel - dx_wu * t_star
    dist_z = pz_rel - dz_wu * t_star
    dist_sq = dist_x.square() + dist_z.square()

    pixel_footprint = t_star.clamp_min(0) * inv_min_focal
    half_width = torch.maximum(
        torch.full_like(t_star, line_half),
        pixel_footprint * footprint_scale,
    )
    r_eff_sq = half_width.square()

    offset_sq = torch.clamp(r_eff_sq - dist_sq, min=0.0)
    t_enter = t_star - torch.sqrt(offset_sq / horiz_safe_u)

    y_at_enter = oy + dy_wu * t_enter
    outside_room = torch.abs(y_at_enter - center_y) > half_height + 1.0e-6

    invalid = (
        (dist_sq > r_eff_sq)
        | (t_star <= near_m)
        | (t_star >= far_m)
        | outside_room
        | (horiz_sq_u < 1.0e-10)
    )

    dist = torch.sqrt(dist_sq + 1.0e-12)
    edge_alpha = _smooth_line_alpha(dist, half_width)
    edge_alpha = torch.where(invalid, torch.zeros_like(edge_alpha), edge_alpha)

    best_alpha_pillar, best_idx = edge_alpha.max(dim=0)
    best_t_pillar = t_enter.gather(0, best_idx.unsqueeze(0)).squeeze(0)
    pillar_hit = best_alpha_pillar > 0

    pillar_rgb = torch.stack(
        [torch.zeros_like(ix), torch.ones_like(ix), torch.zeros_like(ix)],
        dim=-1,
    )
    idx_rgb = best_idx.unsqueeze(0).unsqueeze(-1).expand(1, *best_idx.shape, 3)
    best_rgb = pillar_rgb.gather(0, idx_rgb).squeeze(0)

    take_pillar = pillar_hit & (best_t_pillar < best_t)
    color = torch.where(take_pillar[..., None], best_rgb, color)
    alpha = torch.where(take_pillar, best_alpha_pillar, alpha)

    return color * alpha[..., None]


_RENDER_FRAME_COMPILED: dict[tuple[torch.dtype, str], object] = {}


def _blackwell_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 10
    except Exception:
        return False


def _get_compiled_render(dtype: torch.dtype, device_type: str = "cuda"):
    key = (dtype, device_type)
    if key not in _RENDER_FRAME_COMPILED:
        # PyTorch 2.9's Triton/ptxas path does not handle B200/B300 sm_10x here.
        if device_type == "cuda" and _blackwell_or_newer():
            _RENDER_FRAME_COMPILED[key] = _render_frame_impl
        else:
            mode = "reduce-overhead" if device_type == "cuda" else "default"
            _RENDER_FRAME_COMPILED[key] = torch.compile(
                _render_frame_impl,
                dynamic=False,
                mode=mode,
                fullgraph=False,
            )
    return _RENDER_FRAME_COMPILED[key]


@torch.inference_mode()
def render_infinite_grid_video(
    c2ws: np.ndarray | torch.Tensor,
    intrinsics: np.ndarray | torch.Tensor,
    *,
    height: int,
    width: int,
    device: torch.device | str,
    grid_spacing_m: float = 2.0,
    grid_height_m: float = 6.0,
    line_width_m: float = 0.035,
    pillar_radius_m: float = 0.045,
    pillar_stride_cells: int = 2,
    near_m: float = 0.05,
    far_m: float = 20.0,
    raymarch_steps: int = 72,
    tile_rows: int = 64,
    room_transition_m: float = 1.0,
    pose_reference: str = "first",
    use_compile: bool = True,
    ssaa: int = 2,
) -> torch.Tensor:
    """Render Taichu's infinite room grid on the requested device."""
    del raymarch_steps, tile_rows, pillar_radius_m

    dtype = torch.float32
    ssaa = max(int(ssaa), 1)
    out_height, out_width = int(height), int(width)
    render_height, render_width = out_height * ssaa, out_width * ssaa
    device = torch.device(device)
    if device.type != "cuda":
        use_compile = False

    c2ws_t = torch.as_tensor(c2ws, device=device, dtype=dtype)
    intrinsics_t = torch.as_tensor(intrinsics, device=device, dtype=dtype)
    if c2ws_t.ndim != 3 or c2ws_t.shape[1:] != (4, 4):
        raise ValueError(f"c2ws must be [T,4,4], got {tuple(c2ws_t.shape)}")
    if intrinsics_t.shape != (len(c2ws_t), 3, 3):
        raise ValueError(f"intrinsics must be [T,3,3], got {tuple(intrinsics_t.shape)}")

    if ssaa > 1:
        intrinsics_t = intrinsics_t.clone()
        intrinsics_t[:, 0, 0] *= ssaa
        intrinsics_t[:, 1, 1] *= ssaa
        intrinsics_t[:, 0, 2] *= ssaa
        intrinsics_t[:, 1, 2] *= ssaa

    if pose_reference == "first":
        c2ws_t = _make_first_camera_reference(c2ws_t)
    elif pose_reference != "world":
        raise ValueError(f"unsupported pose_reference={pose_reference!r}")

    spacing = max(float(grid_spacing_m), 1.0e-3)
    room_height = max(float(grid_height_m), 1.0e-3)
    pillar_spacing = spacing * max(int(pillar_stride_cells), 1)
    pillar_anchor_x = torch.full((), pillar_spacing * 0.5, device=device, dtype=dtype)
    pillar_anchor_z = torch.full((), pillar_spacing * 0.5, device=device, dtype=dtype)
    grid_anchor_x = torch.zeros((), device=device, dtype=dtype)
    grid_anchor_z = torch.zeros((), device=device, dtype=dtype)
    c2ws_t = c2ws_t.clone()
    c2ws_t[:, 2, 3] += 2.0
    y_anchor = c2ws_t[0, 1, 3] - room_height * 0.5

    xs = torch.arange(render_width, device=device, dtype=dtype) + 0.5
    ys = torch.arange(render_height, device=device, dtype=dtype) + 0.5

    max_steps = max(int(math.ceil(far_m / pillar_spacing)) + 1, 4)
    offsets = [
        [(step + 0.5) * pillar_spacing, float(d_ix), float(d_iz)]
        for step in range(max_steps)
        for d_ix in (-1, 0, 1)
        for d_iz in (-1, 0, 1)
    ]
    step_offsets = torch.tensor(offsets, device=device, dtype=dtype)
    render_fn = _get_compiled_render(dtype, device.type) if use_compile else _render_frame_impl

    def render_room(
        c2w: torch.Tensor,
        k: torch.Tensor,
        room_index_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return render_fn(
            c2w,
            k,
            xs,
            ys,
            spacing,
            grid_anchor_x,
            grid_anchor_z,
            pillar_spacing,
            pillar_anchor_x,
            pillar_anchor_z,
            y_anchor,
            room_height,
            float(line_width_m),
            float(near_m),
            float(far_m),
            step_offsets,
            room_index_override,
        )

    transition_width = max(float(room_transition_m), 0.0)
    frames = torch.empty(
        (len(c2ws_t), 3, out_height, out_width),
        device=device,
        dtype=dtype,
    )
    room_coord_all = (c2ws_t[:, 1, 3] - y_anchor) / room_height
    boundary_idx_all = torch.round(room_coord_all)
    distance_all = (room_coord_all - boundary_idx_all) * room_height
    in_transition = (
        (torch.abs(distance_all) < transition_width * 0.5).cpu().tolist()
        if transition_width > 0
        else [False] * len(c2ws_t)
    )

    def finalize(hw_last: torch.Tensor) -> torch.Tensor:
        chw = hw_last.clamp(0.0, 1.0).permute(2, 0, 1).contiguous()
        if ssaa > 1:
            chw = F.avg_pool2d(chw.unsqueeze(0), kernel_size=ssaa, stride=ssaa).squeeze(0)
        return chw

    need_clone = use_compile and device.type == "cuda"
    for index in range(len(c2ws_t)):
        c2w = c2ws_t[index]
        k = intrinsics_t[index]
        if in_transition[index]:
            distance = distance_all[index]
            boundary_index = boundary_idx_all[index]
            blend = torch.clamp(distance / transition_width + 0.5, 0.0, 1.0)
            blend = blend * blend * (3.0 - 2.0 * blend)
            lower_raw = render_room(c2w, k, boundary_index - 1)
            lower = lower_raw.clone() if need_clone else lower_raw
            upper_raw = render_room(c2w, k, boundary_index)
            upper = upper_raw.clone() if need_clone else upper_raw
            frame = torch.lerp(lower, upper, blend)
        else:
            raw = render_room(c2w, k, None)
            frame = raw.clone() if need_clone else raw
        frames[index] = finalize(frame)

    return frames


def _relative_c2w(c2ws: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(c2ws, torch.Tensor):
        return torch.linalg.inv(c2ws[0]).unsqueeze(0) @ c2ws
    c2ws_array = np.asarray(c2ws, dtype=np.float32)
    return (np.linalg.inv(c2ws_array[0])[None] @ c2ws_array).astype(np.float32)


def _sample_random_rotation_matrix(max_deg: float) -> np.ndarray:
    max_rad = float(max_deg) * math.pi / 180.0
    rx, ry, rz = np.random.default_rng().uniform(-max_rad, max_rad, size=3)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rotation_x @ rotation_y @ rotation_z).astype(np.float32)


def _apply_global_rotation(
    c2ws: np.ndarray | torch.Tensor,
    rotation: np.ndarray,
) -> np.ndarray | torch.Tensor:
    rotation_4d = np.eye(4, dtype=np.float32)
    rotation_4d[:3, :3] = rotation
    if isinstance(c2ws, torch.Tensor):
        rotation_tensor = torch.as_tensor(
            rotation_4d,
            device=c2ws.device,
            dtype=c2ws.dtype,
        )
        return rotation_tensor.unsqueeze(0) @ c2ws
    return (rotation_4d[None] @ np.asarray(c2ws, dtype=np.float32)).astype(np.float32)


def camera_to_grid(
    c2ws: np.ndarray | torch.Tensor,
    intrinsics: np.ndarray | torch.Tensor,
    height: int,
    width: int,
    device: torch.device | str = "cuda:0",
    random_rot_deg: float = 0.0,
) -> torch.Tensor:
    """Render camera motion with the same geometry and defaults as Taichu."""
    relative = _relative_c2w(c2ws)
    pose_reference = "first"
    if random_rot_deg > 0.0:
        relative = _apply_global_rotation(
            relative,
            _sample_random_rotation_matrix(random_rot_deg),
        )
        pose_reference = "world"
    return render_infinite_grid_video(
        relative,
        intrinsics,
        height=height,
        width=width,
        device=device,
        grid_spacing_m=2.0,
        grid_height_m=6.0,
        line_width_m=0.07,
        pillar_stride_cells=2,
        near_m=0.05,
        far_m=25.0,
        room_transition_m=1.0,
        pose_reference=pose_reference,
    )


__all__ = ["camera_to_grid", "render_infinite_grid_video"]
