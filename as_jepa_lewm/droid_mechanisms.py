"""Small, sample-level DROID mechanism controls; no pretrained components."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .robust import RobustLeWM
from .forecast_scored import ForecastScoredLeWM
from .learned_variance import LearnedVarianceLeWM


class ActionInformation:
    """Regress factual executed actions, never a desired/optimal policy action.

    The same head exists in all controls. It has no batch statistics, and its
    initialization does not consume the model's main random stream.
    """
    def __init__(self, idm_weight=0., **kwargs):
        super().__init__(**kwargs)
        self.idm_weight = float(idm_weight)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261105)
            self.idm = nn.Sequential(nn.Linear(2 * self.state_dim, 256), nn.GELU(),
                                     nn.Linear(256, self.action_dim_for_idm))
        self.idm.requires_grad_(self.idm_weight > 0)
        self._capture_idm = False

    def encode(self, info):
        result = super().encode(info)
        if self._capture_idm:
            self._idm_states = result['emb']
        return result

    def loss(self, batch, step=0, ramp_steps=0):
        self._capture_idm = self.idm_weight > 0
        try:
            loss, metrics = super().loss(batch, step, ramp_steps)
        finally:
            self._capture_idm = False
        if self.idm_weight:
            states = self._idm_states
            del self._idm_states
            estimate = self.idm(torch.cat([states[:, :-1], states[:, 1:]], -1))
            idm = F.mse_loss(estimate.float(), batch['action'][:, :-1].float())
            loss = loss + self.idm_weight * idm
            metrics.update(loss=loss, idm=idm)
        return loss, metrics


class ActionForecast(ActionInformation, RobustLeWM):
    action_dim_for_idm = 7


class ActionScored(ActionInformation, ForecastScoredLeWM):
    action_dim_for_idm = 7


class VarianceControl(ActionInformation, LearnedVarianceLeWM):
    action_dim_for_idm = 7

    def __init__(self, learn_p=False, learn_q=False, **kwargs):
        super().__init__(**kwargs)
        self.learn_p, self.learn_q = bool(learn_p), bool(learn_q)

    def encode(self, info):
        result = super().encode(info)
        if not self.learn_q:
            result['logvar'] = torch.full_like(result['logvar'], 2 * math.log(self.sigma0))
        return result

    def predict_dist(self, emb, act_emb, frozen=False):
        mean, logvar = super().predict_dist(emb, act_emb, frozen)
        if not self.learn_p:
            logvar = torch.full_like(logvar, 2 * math.log(self.sigma0))
        return mean, logvar


ARMS = ['tf', 'tf-idm', 'f8', 'f8-idm', 'variance-fixed', 'variance-prior', 'variance-joint']


def build(arm, seed):
    config = dict(mode='L2', beta=.25, sigma0=.1, window=8, anchor_gap=0,
                  renderer_width=384, renderer_depth=12, renderer_heads=6,
                  action_dim=7, encoder_size=256,
                  forecast_seed=20261029 + int(seed) * 100003)
    if arm in ['tf', 'tf-idm']:
        return ActionForecast(**config, rho=.8, forecast_fraction=.5,
                              idm_weight=1. if arm.endswith('idm') else 0.)
    if arm in ['f8', 'f8-idm']:
        return ActionScored(**config, rho=.8, lam=1., horizon='one_step',
                            forecast_noise='sample', idm_weight=1. if arm.endswith('idm') else 0.)
    if arm.startswith('variance-'):
        return VarianceControl(**config, rho=0., lam=1., horizon='one_step',
                               forecast_noise='sample', learn_p=arm != 'variance-fixed',
                               learn_q=arm == 'variance-joint')
    raise ValueError(arm)
