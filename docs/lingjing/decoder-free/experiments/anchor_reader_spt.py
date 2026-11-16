"""Anchor-reader objective for the AnalyticResidualSPT family (experiments/robocasa_refine/analytic_model.py).

attach_anchor_reader(model, beta_nats=..., **reader_kwargs) adds an AnchorReader to an AnalyticResidualSPT and
replaces its loss by anchor_reader_spt_loss, in the way experiments/robocasa_stable_scaling/train.py patches
methods onto the model. Copy this file to experiments/anchor_reader_spt.py in the collaborator's tree; it needs
as_jepa_lewm/anchor_reader.py.

Objective, per rendered frame (the family's convention), w - 1 frames per window:

    loss = (beta_nats * (Entry + sum_t Surprise_t) + sum_t ent_t) / (w - 1) + reader_loss_weight * R

  Entry, Surprise_t   AnalyticResidualSPT's analytic rate terms, unchanged: the same code up to and including
                      the rate, in the same random order, so entry, surprise, the clean rollout `means` and the
                      noisy codes `noisy` equal the original loss bitwise at a seed and step
  ent_t               sg(score_t) . mu_t with score_t = -xi_hat(noisy_t, anchor) / sigma0: the encoder side of
                      the anchor reader (see as_jepa_lewm/anchor_reader.py). It replaces the renderer's
                      distortion; self.renderer and self.distortion are never called
  R                   the reader's own denoising loss on mu_t + sigma0 * xi' with fresh noise

The anchor is the initial frame: the reader is conditioned on the detached initial code mu_0 and, with
reader_anchor="frame", on its own CNN features of the anchor image that self.prepare(batch) returns. The
posterior noise is AR(1) across frames (rho), but each frame's marginal noise is still N(0, sigma0^2), which is
all the per-frame score identity uses; the reader reads noisy[:, 1:] and is trained on independent draws.
The score is detached because each player takes the gradient of its own loss: the reader never receives the
encoder's terms, and the encoder, pool, predictor and action encoder never receive R.

beta_nats is a price in true nats per nat of rate. It is not the family's `beta`, which multiplies the rate per
rendered value (nats divided by FRAME_ELEMS) against a pixel distortion; `beta` is unused here. The logged
`rate` keeps the family's definition so runs stay comparable. reader_loss_weight keeps the reader's gradient
out of the trainer's shared clip norm (Adam makes the reader's own update independent of that factor); the
logged reader_loss is unweighted. Gaussian coding only: the Laplace coding family has a different score
identity and is rejected. No schedule: ramp_steps must be 0.

Metric keys kept from AnalyticResidualSPT.loss: loss, rate, entry, surprise, feature_std, active_dims,
temporal_noise_rho, clean_rollout_mse, analytic_coding_noise. Dropped with the renderer: fm, rate_share,
forecast_fraction. Added: reader_loss, reader_mse, score_rms.
"""
import math
import types

import torch

from as_jepa_lewm.anchor_reader import (AnchorReader, encoder_entropy_term, reader_denoising_term,
                                        reader_stream_seed)
from experiments.robocasa_refine.analytic_model import innovation_error, root_marginal_error
from vforge.models.as_jepa.codes import surprise_nats
from vforge.models.as_jepa.constants import FRAME_ELEMS


def attach_anchor_reader(model, *, beta_nats, reader_anchor="frame", reader_width=512, reader_depth=3,
                         reader_draws=4, reader_lr=5e-4, reader_seed=0, reader_loss_weight=0.01):
    """Add `model.reader` (parameters named reader.*) and switch `model.loss` to anchor_reader_spt_loss.
    The reader's weights come from their own random stream, so the caller's stream is not shifted.
    `reader_lr` is only stored on the model: the trainer has to give reader.* its own parameter group."""
    if getattr(model, "coding_family", "gaussian") != "gaussian":
        raise ValueError("the anchor reader's score identity needs the Gaussian coding family")
    if reader_anchor not in {"frame", "state"}:
        raise ValueError("unknown reader_anchor")
    for name, value in (("reader_width", reader_width), ("reader_depth", reader_depth),
                        ("reader_draws", reader_draws)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("beta_nats", beta_nats), ("reader_lr", reader_lr),
                        ("reader_loss_weight", reader_loss_weight)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    model.beta_nats, model.reader_anchor = float(beta_nats), reader_anchor
    model.reader_draws, model.reader_lr = reader_draws, float(reader_lr)
    model.reader_seed, model.reader_loss_weight = int(reader_seed), float(reader_loss_weight)
    reader = AnchorReader(model.state_slots, model.state_width, frame=reader_anchor == "frame",
                          width=reader_width, depth=reader_depth, seed=model.reader_seed)
    model.reader = reader.to(next(model.parameters()).device)
    model.loss = types.MethodType(anchor_reader_spt_loss, model)
    return model


def anchor_reader_spt_loss(self, batch, step=0, ramp_steps=0):
    if ramp_steps:
        raise ValueError("no schedule on the anchor-reader objective")
    if getattr(self, "coding_family", "gaussian") != "gaussian":
        raise ValueError("the anchor reader's score identity needs the Gaussian coding family")
    # ---- AnalyticResidualSPT.loss up to and including the rate terms, in the same random order
    pixels, render, anchor = self.prepare(batch)
    info = self.encode(dict(pixels=pixels, action=batch["action"]))
    flat = info["emb"]; b, w = flat.shape[:2]
    mu = self.posterior_means(flat.reshape(b, w, self.state_slots, self.state_width), batch, step)
    if len(anchor) != b:
        raise ValueError("one anchor image per code sequence is required (joint-view encoders are not supported)")
    noise = torch.randn_like(mu)
    for t in range(1, w):
        noise[:, t] = self.rho * noise[:, t - 1] + math.sqrt(1 - self.rho ** 2) * noise[:, t]
    noisy = mu + self.sigma0 * noise
    # the original's unused reference forward: it draws predictor dropout from the outer stream
    reference = self.predict(noisy.flatten(2)[:, :-1], info["act_emb"][:, :-1]); del reference
    devices = [flat.device.index] if flat.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(self.forecast_seed + int(step))
        means = self.mean_trajectory(flat[:, 0], info["act_emb"][:, :-1]).reshape_as(mu)
        # the original draws the decoder's forecast coins here; they only feed the renderer
    geometry = getattr(self, "forecast_rate_geometry", "markov")
    stationary = getattr(self, "stationary_coding_channel", False)
    if geometry == "root-marginal":
        assert not stationary
        error, marginal_constant = root_marginal_error(mu, means, self.rho)
    else:
        assert geometry == "markov"
        error = innovation_error(mu, means, self.rho)
    entry = self.initial_rate(mu[:, 0])
    coding_sigma = self.sigma0
    if stationary:
        coding_sigma *= math.sqrt(1 - self.rho ** 2)
    surprise = surprise_nats(error, torch.zeros_like(error), coding_sigma)
    if not stationary:
        surprise = surprise + self.state_dim * .5 * (-self.rho ** 2 - math.log1p(-self.rho ** 2))
        if geometry == "root-marginal":
            surprise = surprise + self.state_dim * marginal_constant[None, :]
    # ---- the anchor reader in place of the renderer's distortion
    cond = self.reader.condition(mu[:, 0], anchor)
    ent, stats = encoder_entropy_term(self.reader, mu[:, 1:], noisy[:, 1:], cond, self.sigma0)
    reader_loss = reader_denoising_term(self.reader, mu[:, 1:], cond, self.sigma0, self.reader_draws,
                                        reader_stream_seed(self.reader_seed, step)).mean()
    nats = (entry + surprise.sum(1)).mean()
    loss = (self.beta_nats * nats + ent.sum(1).mean()) / (w - 1) + self.reader_loss_weight * reader_loss
    variable = (.5 * mu[:, 0].float().square() + error.square().sum(1) / (2 * coding_sigma ** 2)).mean(0)
    return loss, dict(loss=loss, rate=nats / ((w - 1) * FRAME_ELEMS), entry=entry.mean(), surprise=surprise.mean(),
                      feature_std=mu.float().std((0, 1)).mean(), active_dims=(variable > .1).sum(),
                      temporal_noise_rho=loss.new_tensor(self.rho),
                      clean_rollout_mse=(mu[:, 1:].float() - means[:, 1:].float()).square().mean(),
                      analytic_coding_noise=loss.new_tensor(1.), reader_loss=reader_loss.detach(), **stats)
