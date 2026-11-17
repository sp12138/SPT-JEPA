"""Conditional next-RGB learning, with a detached factual-input readout later.

The current RGB is observed side information. The future-code posterior and
action-conditioned prior share a conditional Gaussian covariance. The decoder
sees current RGB and a noisy posterior code. Optional translation coordinates
use the known current input units, with free remaining world dynamics. There is
no initial-state rate, action output target, or rollout mixture.
"""
import types

import torch


def paired_pixel_mse(predicted, target, physical_batch, steps):
    """Packed RGB rows are physical sample, camera, then transition."""
    per_view = (predicted-target).square().flatten(1).mean(-1)
    return per_view.reshape(physical_batch, 2, steps).mean(1)


def input_noise_std(commands, sigma):
    """Resolve a consequence in units of its known, executed input.

    A zero input retains the base Gaussian, including consistency at rest.
    This changes both posterior noise and prior covariance, not pixel weights.
    """
    length_squared = commands.detach().square().sum(-1)
    return sigma*torch.where(length_squared > 0, length_squared,
                             torch.ones_like(length_squared)).sqrt()


def canonical_translation_mean(current, free_mean, input_effect, basis):
    """Transport translation coordinates; leave the other state free to evolve."""
    return free_mean+((current+input_effect-free_mean)@basis)@basis.T


def canonical_predict(self, means, actions):
    with torch.autocast(means.device.type, enabled=False):
        free = self._one_step_free_predict(means, actions)
        basis = self.action_encoder.weight[:, 7:10]
        return canonical_translation_mean(means.float(), free.float(), actions.float(), basis)


def translation_block_noise(standard_noise, sigma, base_sigma, basis):
    """One conditional Gaussian: input-relative pose, fixed-scale world noise."""
    with torch.autocast(standard_noise.device.type, enabled=False):
        flat = standard_noise.flatten(-2)
        correction = (((flat@basis)*(sigma-base_sigma)[..., None])@basis.T)
        return base_sigma*standard_noise+correction.reshape_as(standard_noise)


def conditional_world_terms(pixel_mse, latent_error, commands, sigma,
                            pixel_elements, risk="absolute", basis=None, base_sigma=None):
    """Gaussian pixel NLL plus latent KL, divided by both cameras' pixels.

    The action-scale variant is a weighted risk, not a different likelihood.
    Its weights use only this transition's known input, with no batch statistic.
    """
    if risk not in ("absolute", "action-scale"):
        raise ValueError("Unknown conditional world risk")
    if basis is None:
        rate = latent_error.square().sum((-2, -1))/(2*sigma**2)
    else:
        with torch.autocast(latent_error.device.type, enabled=False):
            flat = latent_error.flatten(-2)
            position = flat@basis
            world = flat-position@basis.T
            rate = position.square().sum(-1)/(2*sigma**2)
            rate = rate+world.square().sum(-1)/(2*base_sigma**2)
    weight = torch.ones_like(rate)
    if risk == "action-scale":
        length = commands.detach().square().sum(-1)
        weight = 1/torch.where(length > 0, length, torch.ones_like(length))
    return weight*(pixel_mse*.5+rate/pixel_elements), rate, weight


def loss(self, batch, step=0, ramp_steps=0):
    from vforge.models.as_jepa.pixels import pack
    if ramp_steps:
        raise ValueError("The conditional world objective has no ramp")
    bridge = self.renderer.anchor_embed
    assert not self._stereo_capture and bridge.context_tokens is None
    self._stereo_capture = True
    try:
        pixels, render, _ = self.prepare(batch)
        info = self.encode(dict(pixels=pixels, action=batch['action']))
        flat = info['emb']; b, window, width = flat.shape; h = window-1
        current = flat[:, :-1].reshape(b*h, 1, width)
        actions = info['act_emb'][:, :-1].reshape(b*h, 1, width)
        predicted = self.predict(current, actions)[:, 0].reshape(
            b, h, self.state_slots, self.state_width)
        future = flat[:, 1:].reshape_as(predicted)
        raw = batch['action'].reshape(b, 2, window, -1)
        assert torch.equal(raw[:, 0], raw[:, 1])
        mode = self.one_step_world_noise_scale
        sigma = (input_noise_std(raw[:, 0, :-1], self.sigma0)
                 if mode != 'fixed' else self.sigma0)
        basis = self.action_encoder.weight[:, 7:10] if mode == 'translation-block' else None
        standard = torch.randn_like(future)
        if basis is None:
            std = sigma[..., None, None] if torch.is_tensor(sigma) else sigma
            noise = std*standard
        else:
            noise = translation_block_noise(standard, sigma, self.sigma0, basis)
        noisy = future+noise
        condition = noisy[:, None].expand(-1, 2, -1, -1, -1).reshape(
            b*2*h, self.state_slots, self.state_width)
        patch = self.raw_render_size//16
        target = pack(render[:, 1:].flatten(0, 1), patch=patch)
        anchor = pack(render[:, :-1].flatten(0, 1), patch=patch)
        assert len(bridge.context_tokens) == len(target) == b*2*h
        with self.renderer.anchored(anchor):
            rgb = self.renderer(torch.zeros_like(target),
                torch.ones(len(target), 1, device=target.device), condition=condition)
        mse = paired_pixel_mse(rgb.float(), target.float(), b, h)
        elements = render.shape[2]*render.shape[3]*render.shape[4]*2
        terms, rate, weight = conditional_world_terms(mse,
            future.float()-predicted.float(), raw[:, 0, :-1], sigma,
            elements, self.one_step_world_risk, basis=basis, base_sigma=self.sigma0)
        total = terms.mean(); distortion = (weight*mse*.5).mean()
        coding = (weight*rate).mean()/elements
        return total, dict(loss=total, fm=distortion, rate=coding,
            entry=total.new_tensor(0.), surprise=rate.mean(),
            rate_share=coding/distortion.detach().clamp_min(1e-8),
            feature_std=flat.float().std((0, 1)).mean(),
            factual_prediction_mse=(future.float()-predicted.float()).square().mean(),
            conditional_rate_nats=rate.mean(), conditional_entry_rate=total.new_tensor(0.),
            temporal_noise_rho=total.new_tensor(0.), forecast_fraction=total.new_tensor(0.),
            analytic_coding_noise=total.new_tensor(1.), canonical_KL_weight=total.new_tensor(1.),
            action_scale_world_risk=total.new_tensor(float(self.one_step_world_risk == 'action-scale')))
    finally:
        self._stereo_capture = False
        bridge.context_tokens = None
        self.cached = None


def install(model, cfg):
    assert cfg.get('raw_physical_joint') and cfg.get('raw_shared_visual_anchor')
    assert cfg.get('pixel_objective') == 'future-mean' and model.joint_views
    assert not any(cfg.get(k, False) for k in ('conditional_rate_only', 'current_entropy',
        'anchor_reader', 'state_mean_observation', 'relative_state_coding',
        'pure_action_transport', 'isometric_actions', 'ema_future_targets'))
    model.one_step_pixel_world = True
    model.canonical_translation_prior = cfg.get('canonical_translation_prior', False)
    if cfg.get('orthogonal_world_actions', False):
        from experiments.robocasa_stable_scaling.train import IsometricActionEncoder
        with torch.random.fork_rng():
            torch.manual_seed(92320261114+cfg['seed'])
            model.action_encoder = IsometricActionEncoder(14, model.state_dim)
    if model.canonical_translation_prior:
        assert cfg.get('orthogonal_world_actions', False)
        model._one_step_free_predict = model.predict
        model.predict = types.MethodType(canonical_predict, model)
    model.one_step_world_risk = cfg['one_step_world_risk']
    assert model.one_step_world_risk in ('absolute', 'action-scale')
    model.one_step_world_noise_scale = cfg.get('one_step_world_noise_scale', 'fixed')
    assert model.one_step_world_noise_scale in ('fixed', 'executed-input', 'translation-block')
    if model.one_step_world_noise_scale != 'fixed':
        assert model.one_step_world_risk == 'absolute'
    if model.one_step_world_noise_scale == 'translation-block':
        assert cfg.get('orthogonal_world_actions', False)
    model.loss = types.MethodType(loss, model)
