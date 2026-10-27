import torch
from vforge.models.as_jepa.constants import CFG_SCALE, GRID, SAMPLER_STEPS, TOKEN_CH
from vforge.models.rep_world.model import FlowConfig, shift_timestep

FLOW = FlowConfig()                    # SPT's defaults: logit-normal mean and std, base shift, automatic shift

def timestep_shift(world_config, flow=FLOW):
    """SPT's automatic shift, exactly as RepWorld.__init__ computes it (vforge/models/rep_world/model.py)."""
    if flow.shift > 0:
        return flow.shift
    return flow.base_shift * ((world_config.grid // world_config.patch_size) ** 2 / 600) ** 0.5   # 600: SPT's reference token count

def draw_timesteps(n, device, shift, flow=FLOW, uniform=None):     # t = 1 is pure noise
    u = uniform if uniform is not None else torch.rand(n, 1, device=device).clamp_(1e-5, 1 - 1e-5)
    t = torch.sigmoid(flow.mean + flow.std * torch.special.ndtri(u.double())).float()
    return shift_timestep(t, shift)

def flow_loss(renderer, y, state, anchor, shift):   # y, anchor: [N, TOKEN_CH, 1, GRID, GRID]; state: [N, K, d]
    t = draw_timesteps(y.shape[0], y.device, shift)
    eps = torch.randn_like(y)
    tv = t.view(-1, 1, 1, 1, 1)
    with renderer.anchored(anchor):
        v = renderer((1 - tv) * y + tv * eps, t, condition=state)
    return (y - eps - v).square().mean()              # mean over all rendered values; SPT's sign: target velocity y - eps

@torch.no_grad()
def sample(renderer, state, anchor, shift, steps=SAMPLER_STEPS, cfg_scale=CFG_SCALE, generator=None):
    """SPT's Euler sampler (RepWorldInference._denoise), with optional guidance on the state.
    Call with the renderer in eval mode: in train mode WorldDiT drops the state for a fraction of samples."""
    n = state.shape[0]
    z = torch.randn((n, TOKEN_CH, 1, GRID, GRID), device=state.device, generator=generator)
    sched = shift_timestep(torch.linspace(1, 0, steps + 1, device=state.device), shift)
    drop = torch.ones(n, dtype=torch.bool, device=state.device)
    with renderer.anchored(anchor):
        for t_now, t_next in zip(sched[:-1], sched[1:]):
            t = t_now.expand(n, 1)
            v = renderer(z, t, condition=state)
            if cfg_scale > 1 and 0.2 < float(t_now) <= 1.0:        # SPT's guidance interval (0.2, 1.0]
                vu = renderer(z, t, condition=state, condition_drop=drop)
                v = vu + cfg_scale * (v - vu)
            dt = (t_next - t_now).reshape(1, 1, 1, 1, 1)           # a shaped fp32 tensor keeps the update in fp32 under bf16 autocast, as SPT's _denoise does
            z = z - dt * v
    return z
