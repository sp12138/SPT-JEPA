"""Unit-nat factual prediction and entropy conditioned on the observed current RGB.

For Y = E(next RGB) + sigma*noise and X = current RGB,
  CE(Y | X, actual a_t) - H(Y | X)
    = E_a KL(p(Y | X,a) || p_model(Y | X,a)) - I(Y; a | X).
This identity concerns the logged joint distribution, not causal identification.
The isometric actual-action input fixes the otherwise arbitrary code units.
"""
import types

import torch

from as_jepa_lewm.anchor_reader import (encoder_entropy_term, reader_denoising_term,
                                       reader_stream_seed)


def factual_prediction_rate(error, action_effect, sigma, normalize_action_scale=False):
    """Forward residual rate, optionally in units of the factual input length.

    Positive command lengths have no fitted scale or epsilon. A zero command
    retains the original absolute consistency loss. This is weighted factual
    prediction, not a mutual-information bound with action-dependent noise.
    """
    rate = error.float().square().sum((-2, -1)) / (2*sigma**2)
    if normalize_action_scale:
        length_squared = action_effect.detach().float().square().sum((-2, -1))
        rate = rate / torch.where(length_squared > 0, length_squared,
                                  torch.ones_like(length_squared))
    return rate


def profile_prediction_rate(rate, dimensions):
    """Eliminate extra isotropic predictive variance from each sample's KL.

    For q=N(y,bI), p=N(mean,vI), v>=b, the minimizing v is b+||y-mean||²/D.
    This is a profile criterion, not a normalized predictive density or MI bound.
    """
    return dimensions*.5*torch.log1p(2*rate/dimensions)


def current_entropy_loss(self, batch, step=0, ramp_steps=0):
    if ramp_steps:
        raise ValueError("the conditional information objective has no ramp")
    pixels, render, _ = self.prepare(batch)
    encoded = self.encode(dict(pixels=pixels, action=batch['action']))
    flat = encoded['emb']
    b, w = flat.shape[:2]
    current = flat[:, :-1].reshape(b*(w-1), self.state_slots, self.state_width)
    future = flat[:, 1:].reshape_as(current)
    actions = encoded['act_emb'][:, :-1].reshape(b*(w-1), 1, -1)
    query = self.predict(current.flatten(1)[:, None], actions)[:, 0].reshape_as(current)
    rate = (future.float()-query.float()).square().sum((-2, -1)) / (2*self.sigma0**2)
    relative = getattr(self, 'current_entropy_relative', False)
    target = future-current if relative else future
    noisy = target + self.sigma0*torch.randn_like(target)
    # Every reader condition is this transition's observed current image,
    # not the window's initial image or an inferred/predicted state.
    anchor = render[:, :-1].reshape(b*(w-1), *render.shape[2:])
    condition = self.reader.condition(current, anchor)
    entropy, stats = encoder_entropy_term(self.reader, target[:, None], noisy[:, None],
                                         condition, self.sigma0)
    reader_loss = reader_denoising_term(self.reader, target[:, None], condition,
        self.sigma0, self.reader_draws, reader_stream_seed(self.reader_seed, step)).mean()
    loss = (rate+entropy[:, 0]).mean() + self.reader_loss_weight*reader_loss
    return loss, dict(loss=loss, conditional_rate_nats=rate.mean(),
        reader_loss=reader_loss.detach(), feature_std=flat.float().std((0, 1)).mean(),
        factual_prediction_mse=(future.float()-query.float()).square().mean(),
        conditional_entry_rate=loss.new_tensor(0.), temporal_noise_rho=loss.new_tensor(0.),
        analytic_coding_noise=loss.new_tensor(1.),
        entropy_displacement_coordinates=loss.new_tensor(float(relative)), **stats)


def conditional_rate_loss(self, batch, step=0, ramp_steps=0):
    """Matched forward-only ablation; fixed Gaussian noise is integrated out."""
    if ramp_steps:
        raise ValueError("the conditional prediction objective has no ramp")
    pixels, _, _ = self.prepare(batch)
    encoded = self.encode(dict(pixels=pixels, action=batch['action']))
    flat = encoded['emb']
    b, w = flat.shape[:2]
    current = flat[:, :-1].reshape(b*(w-1), 1, -1)
    future = flat[:, 1:].reshape_as(current)
    actions = encoded['act_emb'][:, :-1].reshape_as(current)
    query = self.predict(current, actions)
    error = future.float()-query.float()
    normalize = getattr(self, 'action_scale_normalized_rate', False)
    rate = factual_prediction_rate(error, actions, self.sigma0, normalize)
    if getattr(self, 'profile_prediction_variance', False):
        rate = profile_prediction_rate(rate, error.shape[-2]*error.shape[-1])
    loss = rate.mean()
    return loss, dict(loss=loss, conditional_rate_nats=loss,
        feature_std=flat.float().std((0, 1)).mean(),
        factual_prediction_mse=error.square().mean(),
        action_scale_normalized_rate=loss.new_tensor(float(normalize)),
        profiled_prediction_variance=loss.new_tensor(float(
            getattr(self, 'profile_prediction_variance', False))),
        conditional_entry_rate=loss.new_tensor(0.), temporal_noise_rho=loss.new_tensor(0.),
        analytic_coding_noise=loss.new_tensor(0.))


def install(model, cfg):
    rate_only = cfg.get('conditional_rate_only', False)
    assert bool(rate_only) != bool(cfg.get('current_entropy', False))
    if rate_only:
        assert not cfg.get('anchor_reader', False)
    else:
        assert cfg.get('anchor_reader') and cfg.get('anchor_reader_beta_nats') == 1.
    model.action_scale_normalized_rate = cfg.get('action_scale_normalized_rate', False)
    assert not model.action_scale_normalized_rate or rate_only
    model.profile_prediction_variance = cfg.get('profile_prediction_variance', False)
    assert not model.profile_prediction_variance or rate_only
    model.current_entropy_relative = cfg.get('current_entropy_relative', False)
    assert not model.current_entropy_relative or not rate_only
    assert cfg.get('pure_action_transport') and cfg.get('constant_action_response')
    assert cfg.get('isometric_actions') and not cfg.get('relative_state_coding')
    assert not cfg.get('consequence_likelihood_only') and not cfg.get('ema_future_targets')
    # Current RGB is known side information. There is no initial-state rate;
    # future coding noise is independent in each conditional one-step channel.
    model.renderer.requires_grad_(False)
    model.loss = types.MethodType(conditional_rate_loss if rate_only else current_entropy_loss, model)
