"""A finite diffusion negative ELBO in nats, including both endpoint terms.

The diffusion identity is VDM Eq13: https://arxiv.org/abs/2107.00630.
This is a likelihood bound for the declared reverse Gaussian chain, not exact NLL.
"""
import math
import torch


def vp_coefficients(log_snr):
    return torch.sigmoid(log_snr).sqrt(), torch.sigmoid(-log_snr).sqrt()


def reverse_moments(z_noisy, estimate, log_snr_noisy, log_snr_clean):
    alpha_t, sigma_t = vp_coefficients(log_snr_noisy)
    alpha_s, sigma_s = vp_coefficients(log_snr_clean)
    # Stable 1-(alpha_t/alpha_s)^2 even near the clean endpoint.
    delta = -torch.expm1(torch.nn.functional.logsigmoid(log_snr_noisy)
                        - torch.nn.functional.logsigmoid(log_snr_clean))
    ratio = alpha_t / alpha_s
    variance = sigma_s.square() * delta / sigma_t.square()
    mean = alpha_s * delta / sigma_t.square() * estimate + ratio * sigma_s.square() / sigma_t.square() * z_noisy
    return mean, variance


def transition_kl(target, estimate, log_snr_noisy, log_snr_clean):
    return .5 * (log_snr_clean.exp() - log_snr_noisy.exp()) * (target - estimate).square()


def discrete_gaussian_log_prob(target, mean, scale, bin_width=2 / 255, limits=(-1., 1.)):
    target, mean = target.float(), mean.float()
    scale = torch.as_tensor(scale, dtype=mean.dtype, device=mean.device)
    lower = (target - mean - bin_width / 2) / scale
    upper = (target - mean + bin_width / 2) / scale
    # Mirror intervals above zero before subtracting CDFs, avoiding 1-1 cancellation.
    mirrored = (lower + upper) > 0
    low = torch.where(mirrored, -upper, lower)
    high = torch.where(mirrored, -lower, upper)
    a, b = torch.special.log_ndtr(high), torch.special.log_ndtr(low)
    interior = a + torch.log(-torch.expm1((b - a).clamp_max(-torch.finfo(mean.dtype).eps)))
    result = torch.where(target <= limits[0] + 1e-6, torch.special.log_ndtr(upper), interior)
    return torch.where(target >= limits[1] - 1e-6, torch.special.log_ndtr(-lower), result)


def diffusion_vlb(renderer, target, condition, anchor, steps=64, minimum=-10., maximum=10.):
    n = len(target)
    index = torch.randint(steps, (n, 1), device=target.device)
    ell = minimum + index.float() * ((maximum - minimum) / steps)
    cleaner = ell + (maximum - minimum) / steps
    time = torch.sigmoid(-.5 * ell)
    shaped_time = time.reshape(-1, 1, 1, 1, 1)
    noise = torch.randn_like(target)
    bridge = (1 - shaped_time) * target + shaped_time * noise
    with renderer.anchored(anchor):
        velocity = renderer(bridge, time, condition=condition)
    estimate = bridge.float() + shaped_time * velocity.float()
    middle = (steps * transition_kl(target.float(), estimate,
                                    ell.reshape(-1, 1, 1, 1, 1),
                                    cleaner.reshape(-1, 1, 1, 1, 1))).mean()
    endpoint_time = torch.sigmoid(torch.tensor(-.5 * maximum, device=target.device))
    endpoint_noise = torch.randn_like(target)
    endpoint = (1 - endpoint_time) * target + endpoint_time * endpoint_noise
    with renderer.anchored(anchor):
        endpoint_velocity = renderer(endpoint, endpoint_time.expand(n, 1), condition=condition)
    endpoint_estimate = endpoint.float() + endpoint_time * endpoint_velocity.float()
    reconstruction = -discrete_gaussian_log_prob(
        target, endpoint_estimate, math.exp(-maximum / 2)).mean()
    alpha, sigma = vp_coefficients(torch.tensor(minimum, device=target.device))
    prior = .5 * (alpha.square() * target.float().square() + sigma.square() - 1 - sigma.square().log()).mean()
    total = middle + reconstruction + prior
    metrics = dict(density_middle_nats=middle.detach(),
                   density_reconstruction_nats=reconstruction.detach(),
                   density_prior_nats=prior.detach(),
                   density_steps=torch.tensor(float(steps), device=target.device),
                   raw_flow_mse=(velocity.float() - (target - noise).float()).square().mean().detach())
    return total, metrics


@torch.no_grad()
def sample_diffusion(renderer, condition, anchor, steps=64, minimum=-10., maximum=10., generator=None):
    batch = len(condition)
    z = torch.randn(batch, 48, 1, 16, 16, device=condition.device, generator=generator)
    levels = torch.linspace(minimum, maximum, steps + 1, device=z.device, dtype=torch.float64)
    with renderer.anchored(anchor):
        for noisy, cleaner in zip(levels[:-1], levels[1:]):
            time = torch.sigmoid(-.5 * noisy).float()
            norm = ((1 - time).square() + time.square()).sqrt()
            bridge = z * norm
            velocity = renderer(bridge, time.expand(batch, 1), condition=condition)
            estimate = bridge.float() + time * velocity.float()
            mean, variance = reverse_moments(z, estimate, noisy, cleaner)
            z = mean.float() + variance.sqrt().float() * torch.randn(z.shape, device=z.device, generator=generator)
        time = torch.sigmoid(-.5 * levels[-1]).float()
        bridge = z * ((1 - time).square() + time.square()).sqrt()
        estimate = bridge.float() + time * renderer(bridge, time.expand(batch, 1), condition=condition).float()
    # Sampling the endpoint Gaussian and rounding implements its256-bin likelihood.
    draw = estimate + math.exp(-maximum / 2) * torch.randn(z.shape, device=z.device, generator=generator)
    return ((draw.clamp(-1, 1) + 1) * 127.5).round() / 127.5 - 1
