"""Frozen decoder-response geometry; no fitting, optimizer, or planner changes."""
import hashlib
import math
import torch

SEED = 20261031
PROBES = 8


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        a = value.detach().cpu().contiguous()
        h.update(str((key, tuple(a.shape), a.dtype)).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cpu_generator(index, stream):
    return torch.Generator(device="cpu").manual_seed(SEED + 10007 * index + 1000003 * stream)


def draw(shape, index, stream, device, normal=False):
    fn = torch.randn if normal else torch.rand
    return fn(shape, generator=cpu_generator(index, stream), dtype=torch.float32).to(device)


@torch.no_grad()
def training_point(model, batch, index):
    """One fixed uniform horizon per train clip, training condition law in eval mode.

    This reproduces the posterior/forecast mixture distribution, not training
    dropout or the historical outer RNG stream. Only the condition derivative
    is taken later; x_t, anchor, encoder and forecast are fixed constants.
    """
    from vforge.models.as_jepa.flow import draw_timesteps, FLOW
    from vforge.models.as_jepa.pixels import pack
    assert not model.training and model.state_dim == 64
    assert model.objective == "flow" and not model.flow_force_t_one
    pixels, render, anchor = model.prepare(batch)
    info = model.encode({"pixels": pixels, "action": batch["action"]})
    mu = info["emb"]
    assert mu.shape[0] == 1 and mu.shape[1] == 8
    device = mu.device
    noise = draw(mu.shape, index, 1, device, normal=True)
    for t in range(1, mu.shape[1]):
        noise[:, t] = model.rho * noise[:, t-1] + math.sqrt(1-model.rho**2) * noise[:, t]
    noisy = mu + model.sigma0 * model.posterior_noise_scale * noise
    horizon = 1 + int(torch.randint(7, (1,), generator=cpu_generator(index, 2)).item())
    forecast = bool(draw((1,), index, 3, device).item() < model.forecast_fraction)
    if forecast:
        history = [noisy[:, 0]]
        for t in range(horizon):
            history.append(model.predict(torch.stack(history, 1), info["act_emb"][:, :t+1])[:, -1])
        condition = history[-1]
    else:
        condition = noisy[:, horizon]
    y = pack(render[:, horizon])
    uniform = draw((1, 1), index, 4, device).clamp(1e-5, 1-1e-5)
    time = draw_timesteps(1, device, model.training_shift, uniform=uniform)
    eps = draw(y.shape, index, 5, device, normal=True)
    x = (1-time.view(1, 1, 1, 1, 1))*y + time.view(1, 1, 1, 1, 1)*eps
    meta = dict(clip_index=index, horizon=horizon, forecast=forecast, time=float(time.item()),
                shift=model.training_shift, flow_mean=FLOW.mean, flow_std=FLOW.std,
                rho=model.rho, sigma0=model.sigma0, posterior_noise_scale=model.posterior_noise_scale,
                forecast_fraction=model.forecast_fraction, condition_shape=list(condition.shape),
                anchor_row=int(batch["rows"][0, 0]), target_row=int(batch["rows"][0, horizon]),
                output_elements=y.numel(), condition_is_leaf_for_jacobian=True)
    return condition.detach(), x.detach(), time, pack(anchor).detach(), meta


def response(model, x, time, anchor):
    def function(z):
        with model.renderer.anchored(anchor):
            return model.renderer(x, time, condition=z.reshape(1, model.state_slots, model.state_width))
    return function


def vjp_gram(function, z, probes, index=0):
    """E[Ghat|J] = J.T J / P; no identity floor or retained parameter gradient."""
    z = z.detach().requires_grad_(True)
    value = function(z)
    generator = cpu_generator(index, 6)
    vectors = []
    for i in range(probes):
        signs = torch.randint(2, value.shape, generator=generator, device="cpu")
        signs = (signs.to(device=value.device, dtype=value.dtype)*2-1)/math.sqrt(value.numel())
        vector, = torch.autograd.grad(value, z, grad_outputs=signs, retain_graph=i+1 < probes)
        vectors.append(vector.detach().flatten().double())
    vectors = torch.stack(vectors)
    return vectors.T @ vectors / probes, vectors


def directional_check(function, z, index=0):
    direction = draw(z.shape, index, 7, z.device, normal=True).to(z.dtype)
    direction = direction/direction.norm()
    _, tangent = torch.func.jvp(function, (z.detach(),), (direction,))
    # Two predetermined FP32 central differences are a numerical check, not a search.
    h = torch.finfo(z.dtype).eps**(1/3) * max(1., float(z.norm()))
    errors = []
    with torch.no_grad():
        for step in [h, h/2]:
            finite = (function(z+step*direction)-function(z-step*direction))/(2*step)
            rel = (finite-tangent).norm()/tangent.norm().clamp_min(torch.finfo(z.dtype).tiny)
            errors.append(dict(step=step, relative_error=float(rel)))
    return dict(jvp_norm=float(tangent.norm()), mean_squared_response=float(tangent.square().mean()),
                finite_differences=errors), direction.flatten().double()


def spectrum(gram):
    eigen = torch.linalg.eigvalsh((gram+gram.T)/2)
    tol = torch.finfo(gram.dtype).eps*len(eigen)*eigen.abs().max()
    trace = eigen.sum()
    return dict(eigenvalues=eigen.cpu().tolist(), numerical_rank=int((eigen > tol).sum()),
                tolerance=float(tol), psd=bool(eigen[0] >= -tol), trace=float(trace),
                participation_rank=float(trace.square()/eigen.square().sum()) if float(trace) > 0 else 0.)
