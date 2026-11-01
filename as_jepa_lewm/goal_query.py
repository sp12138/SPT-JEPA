"""Action-conditioned future-query classification with independent references.

The output is a goal-scoring function, not an autoregressed physical state.
The observation carrier is fixed; all targets are observed training images.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.fixed_forecast import _NoRenderer
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM


class GoalQueryLeWM(SpatialCarrierLeWM):
    renderer_type = _NoRenderer

    def __init__(self, query_seed=1, **kwargs):
        super().__init__(**kwargs)
        if (self.window != 8 or self.spatial_mode != 'local' or self.state_dim != 1024
                or self.spatial_basis != 'identity' or self.spatial_chart != 'none'
                or self.spatial_readout_mode != 'none' or self.spatial_state_interface != 'chart'):
            raise ValueError('Use the existing fixed local1024 observer and window8')
        self.renderer = None
        hidden = self.predictor.hidden_dim
        self.query_dim = hidden
        # Retain the complete native body. Its192D output parameterizes a goal
        # function; a removed192→1024 state readout has no unused parameters.
        self.predictor.transformer.output_proj = nn.Identity()
        self.predictor.output_dim = hidden
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(int(query_seed)).get_state())
            self.goal_projector = nn.Sequential(nn.Linear(1024, hidden), nn.LayerNorm(hidden))
        self.method = 'action_conditioned_goal_query'

    def query_predictions(self, context, action_embeddings):
        """All future queries in one causal pass; no future images or feedback.

        Each future query starts with the known current observation, plus the
        native time embedding. Earlier action prefixes remain accessible.
        """
        h = context.shape[1]
        n = action_embeddings.shape[1]
        if h != 3 or context.shape[-1] != 1024 or not h <= n <= 7:
            raise ValueError('Expected context3 and3..7 causal action tokens')
        inputs = torch.cat([context, context[:, -1:].expand(-1, n - h, -1)], 1)
        return self.predictor(inputs, action_embeddings)[:, h-1:]

    def query_logits(self, queries, goals):
        return (queries.float() * self.goal_projector(goals).float()).sum(-1) / math.sqrt(self.query_dim)

    def loss(self, batch, step=0, ramp_steps=0):
        if 'reference_pixels' not in batch:
            raise ValueError('Independent per-item reference pixels are required')
        info = self.encode(dict(pixels=self.normalize(batch['pixels']), action=batch['action']))
        observed, actions = info['emb'], info['act_emb']
        reference = self.encode(dict(pixels=self.normalize(batch['reference_pixels'])))['emb']
        if observed.shape[1] != 8 or reference.shape != observed[:, 3:].shape:
            raise ValueError('Matched actual/reference future offsets3..7 required')
        devices = [observed.device.index] if observed.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.forecast_seed + int(step))
            queries = self.query_predictions(observed[:, :3], actions[:, :7])
        positive = self.query_logits(queries, observed[:, 3:])
        negative = self.query_logits(queries, reference)
        per_clip = (F.softplus(-positive) + F.softplus(negative)).mean(1)
        loss = per_clip.mean()
        return loss, dict(loss=loss, positive_logit=positive.mean(), negative_logit=negative.mean(),
                          classification_accuracy=((positive > 0).float().mean() + (negative < 0).float().mean()) / 2,
                          feature_std=observed.float().std((0, 1)).mean(),
                          query_std=queries.float().std((0, 1)).mean())

    def predict(self, *args, **kwargs):
        raise RuntimeError('GoalQueryLeWM predicts goal-function parameters;use query_predictions/rollout')

    def rollout(self, info, action_sequence, history_size=None):
        b, samples, horizon = action_sequence.shape[:3]
        h = info['pixels'].shape[2]
        if not 1 <= h <= 3 or not 1 <= horizon <= 5:
            raise ValueError('Keep native at-most3 real context observations andH<=5')
        past = info.get('action_history')
        if past is None:
            past = action_sequence.new_zeros(b, samples, 0, action_sequence.shape[-1])
        if past.shape[2] != h - 1:
            raise ValueError('Actual executed context actions are required')
        if 'emb' not in info:
            context = self.encode(dict(pixels=info['pixels'][:, 0]))['emb']
            info['emb'] = context[:, None].expand(b, samples, h, -1)
        context = info['emb'].reshape(b*samples, h, 1024)
        all_actions = torch.cat([past, action_sequence], 2).reshape(b*samples, h-1+horizon, -1)
        if h < 3:
            # A cold start has no additional observations. Reuse its first
            # known frame and mark unavailable past normalized actions as0.
            padding = 3-h
            context = torch.cat([context[:, :1].expand(-1, padding, -1), context], 1)
            all_actions = torch.cat([all_actions.new_zeros(b*samples, padding, all_actions.shape[-1]), all_actions], 1)
        queries = self.query_predictions(context, self.action_encoder(all_actions))
        goals = info['goal_emb'][:, -1]
        query_goals = self.goal_projector(goals).float()[:, None]
        final = queries[:, -1].reshape(b, samples, self.query_dim).float()
        info['predicted_query'] = queries.reshape(b, samples, horizon, self.query_dim)
        info['goal_query_cost'] = -(final * query_goals).sum(-1) / math.sqrt(self.query_dim)
        return info

    def audit_expected_gradients(self):
        result = dict(encoder='frozen', pool='frozen', predictor='active', action_encoder='active', goal_projector='active')
        result.update({f'predictor.transformer.layers.{i}': 'active' for i in range(6)})
        return result

    def render_states(self, *args, **kwargs):
        raise RuntimeError('GoalQueryLeWM has no image renderer')


class GoalQueryObjective(nn.Module):
    """Negative learned log-density-ratio score; nativeCEM budget unchanged."""
    def __init__(self, pred_key='predicted_emb', goal_key='goal_emb', reduction='sum'):
        super().__init__()
        if (pred_key, goal_key, reduction) != ('predicted_emb', 'goal_emb', 'sum'):
            raise ValueError('Do not mix the query score with other goal/cost reductions')

    def forward(self, info):
        value=info['goal_query_cost']
        import os
        if os.environ.get('ASJEPA_QUERY_AUDIT') and not getattr(self,'_recorded',False):
            import hashlib,json
            from pathlib import Path
            path=Path(os.environ['ASJEPA_QUERY_AUDIT'])
            with path.open('x') as f:
                json.dump(dict(objective=type(self).__name__,cost_shape=list(value.shape),finite=bool(torch.isfinite(value).all()),
                               source=str(Path(__file__).resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
            self._recorded=True
        return value
