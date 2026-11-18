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
from experiments.robocasa_stable_scaling.rigid_transport import rigid_prediction_error


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


def profile_prediction_rate(rate, dimensions, covariance="isotropic"):
    """Eliminate predictive covariance from each sample's Gaussian KL.

    For q=N(y,bI), p=N(mean,vI), v>=b, the minimizing v is b+||y-mean||²/D.
    With arbitrary covariance >= bI, its optimum is bI+error*error.T instead;
    the matrix determinant lemma gives log1p(2*rate)/2 without a D factor.
    This is a profile criterion, not a normalized predictive density or MI bound.
    """
    if covariance not in ("isotropic", "full"):
        raise ValueError("Unknown predictive covariance family")
    if covariance == "full":
        dimensions = 1
    return dimensions*.5*torch.log1p(2*rate/dimensions)


def block_profile_prediction_rate(error, action_effect, sigma, basis,
                                  normalize_action_scale=False):
    """Profile Gaussian covariance separately in two orthogonal subspaces.

    Columns of basis must be orthonormal. With P=basis*basis.T, the optimum
    block-diagonal covariance >= bI is bI + (Pe)(Pe).T + (Re)(Re).T, R=I-P.
    Its KL is the sum of two rank-one log determinants. This prevents errors
    outside P from reducing the gradient of errors inside P. As with the full
    profile, this is an energy criterion, not a normalized predictive density.
    """
    dtype = torch.float64 if error.dtype == torch.float64 else torch.float32
    with torch.autocast(error.device.type, enabled=False):
        flat = error.to(dtype).flatten(-2)
        basis = basis.to(dtype)
        assert basis.ndim == 2 and basis.shape[0] == flat.shape[-1]
        parallel = flat@basis
        remaining = flat-parallel@basis.T
        base = flat.new_full(flat.shape[:-1], sigma**2)
        if normalize_action_scale:
            length = action_effect.detach().to(dtype).square().sum((-2, -1))
            base = base*torch.where(length > 0, length, torch.ones_like(length))
        return .5*torch.log1p(parallel.square().sum(-1)/base) + \
            .5*torch.log1p(remaining.square().sum(-1)/base)


def direction_profile_prediction_rate(error, action_effect, sigma, normalize_action_scale=False):
    """Profile the command direction and its orthogonal complement separately.

    The actually issued input defines the direction on each transition.
    This allows an amplitude error to downweight its own component without
    downweighting transverse prediction errors in the same observation.
    """
    dtype = torch.float64 if error.dtype == torch.float64 else torch.float32
    with torch.autocast(error.device.type, enabled=False):
        flat = error.to(dtype).flatten(-2)
        command = action_effect.to(dtype).flatten(-2)
        length_squared = command.square().sum(-1, keepdim=True)
        safe_squared = torch.where(length_squared > 0, length_squared, 1.)
        direction = command * safe_squared.rsqrt()
        parallel = (flat * direction).sum(-1, keepdim=True)
        transverse = flat - parallel * direction
        base = sigma**2
        if normalize_action_scale:
            base = base * safe_squared.squeeze(-1).detach()
        return .5 * (torch.log1p(parallel.squeeze(-1).square() / base) +
                     torch.log1p(transverse.square().sum(-1) / base))


def prefix_prediction_rate(error, action_effect, sigma):
    """Profile causal innovations of a trajectory's integrated residuals.

    Inputs have shape (physical samples, transitions, code dimensions). Each
    increment stays in the units of its own issued command. If the normalized
    residuals are differences of independent observations with marginal
    increment variance one, this transformation whitens them. It is the
    residual of each cumulative observation against its preceding mean,
    including the zero origin, with its predictive variance removed.
    """
    dtype = torch.float64 if error.dtype == torch.float64 else torch.float32
    with torch.autocast(error.device.type, enabled=False):
        error = error.to(dtype)
        command_squared = action_effect.detach().to(dtype).square().sum(-1, keepdim=True)
        scale = torch.where(command_squared > 0, command_squared, 1.).sqrt() * sigma
        path = (error / scale).cumsum(dim=1)
        index = torch.arange(1, path.shape[1] + 1, dtype=dtype, device=path.device)
        index = index[None, :, None]
        preceding_mean = (path.cumsum(dim=1) - path) / index
        innovation = (path - preceding_mean) * (2 * index / (index + 1)).sqrt()
        return .5 * torch.log1p(innovation.square().sum(-1))


def causal_shortfall_residual(error, translation_effect, action_effect):
    """Estimate an absolute translation shortfall from observed history.

    Fit one nonnegative scalar along the issued translation directions in
    each physical sample, weighting earlier facts by the same inverse squared
    command lengths as the forward rate. Only earlier observations fit the
    shortfall for the current prediction. Clip it at the current translation
    length so that the estimate cannot reverse the issued direction.
    """
    dtype = torch.float64 if error.dtype == torch.float64 else torch.float32
    with torch.autocast(error.device.type, enabled=False):
        error = error.to(dtype)
        translation_effect = translation_effect.to(dtype)
        length = translation_effect.norm(dim=-1, keepdim=True)
        moving = length > 0
        direction = translation_effect / torch.where(moving, length, 1.)
        command_squared = action_effect.detach().to(dtype).square().sum(-1, keepdim=True)
        relative_precision = torch.where(command_squared > 0, command_squared, 1.).reciprocal()
        relative_precision = relative_precision * moving
        shortfall = -(error * direction).sum(-1, keepdim=True)
        stats = torch.cat((shortfall * relative_precision, relative_precision), dim=-1)
        history = torch.cat((torch.zeros_like(stats[:, :1]), stats.cumsum(dim=1)[:, :-1]), dim=1)
        numerator, denominator = history.split(1, dim=-1)
        estimate = (numerator / torch.where(denominator > 0, denominator, 1.)).clamp_min(0.)
        correction = torch.minimum(estimate, length)
        return error + correction * direction, correction


def camera_motion_rate(codes, action_effect, sigma, joint_codes=None):
    """Profile the mean squared forward residual across synchronized cameras.

    Each view predicts the effect of the same issued command. Its code can
    have a different constant origin. Mean squared residual is the error of
    the mean displacement plus the variance of the views' displacements.
    Supplying a joint observation recenters the camera estimates at the
    original stereo prediction while retaining exactly this variance.
    """
    dtype = torch.float64 if codes.dtype == torch.float64 else torch.float32
    codes = codes.to(dtype)
    command = action_effect.to(dtype)
    if joint_codes is None:
        error = codes[:, 1:] - (codes[:, :-1] + command.unsqueeze(2))
        square = error.square().sum(dim=-1).mean(dim=2)
    else:
        joint = joint_codes.to(dtype)
        error = joint[:, 1:] - (joint[:, :-1] + command)
        displacement = codes[:, 1:] - codes[:, :-1]
        deviation = displacement - displacement.mean(dim=2, keepdim=True)
        square = error.square().sum(dim=-1) + deviation.square().sum(dim=-1).mean(dim=2)
    length_squared = command.detach().square().sum(dim=-1)
    variance = sigma**2 * torch.where(length_squared > 0, length_squared, 1.)
    return .5 * torch.log1p(square / variance), square


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
    if getattr(self, 'camera_motion_consensus', False):
        rate, square = camera_motion_rate(
            encoded['camera_codes'], encoded['act_emb'][:, :-1], self.sigma0,
            joint_codes=flat if self.pool.motion_center == 'joint' else None)
        loss = rate.mean()
        return loss, dict(loss=loss, conditional_rate_nats=loss,
            feature_std=flat.float().std((0, 1)).mean(),
            factual_prediction_mse=square.mean()/flat.shape[-1],
            camera_motion_consensus=loss.new_tensor(1.))
    b, w = flat.shape[:2]
    current = flat[:, :-1].reshape(b*(w-1), 1, -1)
    future = flat[:, 1:].reshape_as(current)
    if getattr(self, 'rigid_command_transport', False):
        # Both camera rows describe one physical observation. Joint encode
        # already verifies that they carry the same executed command.
        command = batch['action'].reshape(b, 2, w, -1)[:, 0, :-1]
        command = command.reshape(b*(w-1), -1)
        error_square, rotation_error = rigid_prediction_error(
            current[:, 0], future[:, 0], command, self.action_encoder.weight)
        command_square = command.detach().float().square().sum(-1)
        base = self.sigma0**2 * torch.where(command_square > 0, command_square, 1.)
        loss = profile_prediction_rate(error_square/(2*base), flat.shape[-1], 'full').mean()
        return loss, dict(loss=loss, conditional_rate_nats=loss,
            feature_std=flat.float().std((0, 1)).mean(),
            factual_prediction_mse=error_square.mean()/flat.shape[-1],
            factual_rotation_error=rotation_error.mean(),
            profiled_full_predictive_covariance=loss.new_tensor(1.),
            rigid_command_transport=loss.new_tensor(1.))
    actions = encoded['act_emb'][:, :-1].reshape_as(current)
    query = self.predict(current, actions)
    error = future.float()-query.float()
    shortfall = getattr(self, 'factual_execution_shortfall', False)
    if shortfall:
        command = batch['action'].reshape(b, 2, w, -1)[:, 0, :-1, 7:10]
        with torch.autocast(error.device.type, enabled=False):
            translation_effect = command.float() @ self.action_encoder.weight[:, 7:10].T
        corrected, correction = causal_shortfall_residual(
            error.reshape(b, w-1, -1), translation_effect,
            actions.reshape(b, w-1, -1))
        error = corrected.reshape_as(error)
    normalize = getattr(self, 'action_scale_normalized_rate', False)
    covariance = getattr(self, 'profile_prediction_covariance', 'isotropic')
    prefix = getattr(self, 'factual_prefix_innovations', False)
    if prefix:
        rate = prefix_prediction_rate(error.reshape(b, w-1, -1),
            actions.reshape(b, w-1, -1), self.sigma0)
    elif covariance == 'translation-blocks':
        rate = block_profile_prediction_rate(error, actions, self.sigma0,
            self.action_encoder.weight[:, 7:10], normalize)
    elif covariance == 'command-direction':
        rate = direction_profile_prediction_rate(error, actions, self.sigma0, normalize)
    else:
        rate = factual_prediction_rate(error, actions, self.sigma0, normalize)
        if getattr(self, 'profile_prediction_variance', False):
            rate = profile_prediction_rate(rate, error.shape[-2]*error.shape[-1], covariance)
    loss = rate.mean()
    metrics = dict(loss=loss, conditional_rate_nats=loss,
        feature_std=flat.float().std((0, 1)).mean(),
        factual_prediction_mse=error.square().mean(),
        action_scale_normalized_rate=loss.new_tensor(float(normalize)),
        profiled_prediction_variance=loss.new_tensor(float(
            getattr(self, 'profile_prediction_variance', False))),
        profiled_full_predictive_covariance=loss.new_tensor(float(
            getattr(self, 'profile_prediction_covariance', 'isotropic') == 'full')),
        profiled_translation_blocks=loss.new_tensor(float(covariance == 'translation-blocks')),
        conditional_entry_rate=loss.new_tensor(0.), temporal_noise_rho=loss.new_tensor(0.),
        analytic_coding_noise=loss.new_tensor(0.))
    if covariance == 'command-direction':
        metrics['profiled_command_direction'] = loss.new_tensor(1.)
    if prefix:
        metrics['factual_prefix_innovations'] = loss.new_tensor(1.)
    if shortfall:
        metrics['factual_execution_shortfall'] = correction.detach().mean()
    return loss, metrics


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
    model.profile_prediction_covariance = cfg.get('profile_prediction_covariance', 'isotropic')
    assert model.profile_prediction_covariance in (
        'isotropic', 'full', 'translation-blocks', 'command-direction')
    assert model.profile_prediction_covariance == 'isotropic' or model.profile_prediction_variance
    if model.profile_prediction_covariance == 'translation-blocks':
        assert rate_only and cfg.get('raw_physical_joint')
    model.rigid_command_transport = cfg.get('rigid_command_transport', False)
    if model.rigid_command_transport:
        assert rate_only and cfg.get('raw_physical_joint')
        assert model.profile_prediction_covariance == 'full'
        assert model.action_scale_normalized_rate and cfg.get('learn_angular_commands')
    model.factual_prefix_innovations = cfg.get('factual_prefix_innovations', False)
    if model.factual_prefix_innovations:
        assert rate_only and cfg.get('raw_physical_joint')
        assert model.profile_prediction_covariance == 'full'
        assert model.action_scale_normalized_rate and not model.rigid_command_transport
    model.factual_execution_shortfall = cfg.get('factual_execution_shortfall', False)
    if model.factual_execution_shortfall:
        assert rate_only and cfg.get('raw_physical_joint')
        assert model.profile_prediction_covariance == 'full' and model.action_scale_normalized_rate
        assert not model.rigid_command_transport and not model.factual_prefix_innovations
    model.camera_motion_consensus = cfg.get('camera_motion_consensus', False)
    if model.camera_motion_consensus:
        assert rate_only and cfg.get('raw_physical_joint') and model.action_scale_normalized_rate
        assert model.profile_prediction_covariance == 'full'
        assert not (model.rigid_command_transport or model.factual_prefix_innovations
                    or model.factual_execution_shortfall)
    model.current_entropy_relative = cfg.get('current_entropy_relative', False)
    assert not model.current_entropy_relative or not rate_only
    assert cfg.get('pure_action_transport') and cfg.get('constant_action_response')
    assert cfg.get('isometric_actions') and not cfg.get('relative_state_coding')
    assert not cfg.get('consequence_likelihood_only') and not cfg.get('ema_future_targets')
    # Current RGB is known side information. There is no initial-state rate;
    # future coding noise is independent in each conditional one-step channel.
    model.renderer.requires_grad_(False)
    model.loss = types.MethodType(conditional_rate_loss if rate_only else current_entropy_loss, model)
