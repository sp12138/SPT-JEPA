"""A conditional action distribution pushed through the actuator's clip map.

The single label is the known, executed current action. A saturated value is
an event (latent command beyond a bound), rather than an interior density
observation. Point likelihood is the matched negative control. Mixture models
and censoring are established statistics, not claimed as new inventions.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .temporal_bridge import TemporalBridge


class CensoredActionBridge(TemporalBridge):
    def __init__(self, components=4, action_lower=None, action_upper=None,
                 observation_model='censored', **kwargs):
        assert kwargs.get('objective', 'mse') == 'mse'
        assert kwargs.get('chunk', 1) == 1 and kwargs.get('action_dim', 4) == 4
        assert kwargs.get('temporal_aux', False) is False
        assert kwargs.get('condition_horizon', 'none') == 'none'
        assert observation_model in ['point', 'censored'] and components >= 1
        super().__init__(**kwargs)
        self.objective = 'mixture'
        self.components = components
        self.observation_model = observation_model
        assert len(action_lower) == len(action_upper) == self.action_dim
        self.register_buffer('action_lower', torch.tensor(action_lower, dtype=torch.float32))
        self.register_buffer('action_upper', torch.tensor(action_upper, dtype=torch.float32))
        assert bool((self.action_upper > self.action_lower).all())
        # Replace only the final layer AFTER constructing every baseline module:
        # all common CNN/decoder/fusion initial tensors remain exactly paired.
        self.action_readout[-1] = nn.Linear(self.width, components*(2*self.action_dim+1))

    def action_field(self, x, time, horizon, memory):
        # Neither a noised target action nor a future command enters this head.
        return super().action_field(torch.zeros_like(x), torch.zeros_like(time),
                                    torch.zeros_like(horizon), memory)

    def distribution(self, memory):
        value = memory.new_zeros((len(memory), self.chunk, self.action_dim))
        raw = self.action_field(value, value.new_zeros(len(value)),
                                value.new_zeros(len(value)), memory).float()
        raw = raw.reshape(len(value), self.chunk, self.components, 2*self.action_dim+1)
        logits = raw[..., 0]
        location = raw[..., 1:1+self.action_dim]
        # Machine precision is the only scale floor; no task-specific bounds.
        scale = F.softplus(raw[..., 1+self.action_dim:])+torch.finfo(torch.float32).eps
        return logits, location, scale

    def log_probability(self, target, logits, location, scale):
        target = target.float().unsqueeze(-2)
        z = (target-location)/scale
        logp = F.logsigmoid(z)+F.logsigmoid(-z)-scale.log()
        if self.observation_model == 'censored':
            log_left = F.logsigmoid((self.action_lower-location)/scale)
            log_right = F.logsigmoid((location-self.action_upper)/scale)
            logp = torch.where(target <= self.action_lower, log_left, logp)
            logp = torch.where(target >= self.action_upper, log_right, logp)
        return torch.logsumexp(F.log_softmax(logits, -1)+logp.sum(-1), -1)

    def loss(self, batch, step=0, ramp_steps=0):
        logits, location, scale = self.distribution(self.memory(batch))
        per_step = -self.log_probability(batch['action'], logits, location, scale)/self.action_dim
        mask = batch['action_mask'].float()
        loss = ((per_step*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
        return loss, dict(loss=loss, action_nll=loss)

    @torch.inference_mode()
    def propose(self, batch, generator=None, steps=12):
        logits, location, scale = self.distribution(self.memory(batch))
        component = torch.multinomial(logits.softmax(-1).flatten(0, 1), 1,
                                      generator=generator).reshape(len(logits), self.chunk, 1, 1)
        index = component.expand(-1, -1, 1, self.action_dim)
        location = location.gather(2, index).squeeze(2)
        scale = scale.gather(2, index).squeeze(2)
        eps = torch.finfo(torch.float32).eps
        uniform = torch.rand(location.shape, device=location.device, generator=generator).clamp(eps, 1-eps)
        latent = location+scale*(uniform.log()-torch.log1p(-uniform))
        return latent.maximum(self.action_lower).minimum(self.action_upper)
