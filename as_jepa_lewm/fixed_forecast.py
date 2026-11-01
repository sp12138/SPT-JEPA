"""Direct factual forecasting in a fixed observed state, without a renderer.

The original native planner supplies history3 and predicts5 steps. Train that
same conditional map on the existing eight-frame clips. No optimal-action,
physical-state, success, batch-statistic or generated-target supervision.
"""
import torch
from torch import nn

from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from as_jepa_lewm.spatial_native import SpatialNativePredictor


class _NoRenderer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class FixedForecastLeWM(SpatialCarrierLeWM):
    renderer_type = _NoRenderer

    def __init__(self, predictor_kind='spatial', spatial_predictor_seed=1, **kwargs):
        if predictor_kind not in {'flat', 'spatial'}:
            raise ValueError('Use flat or full native spatial predictor')
        super().__init__(**kwargs)
        if self.window != 8 or not self.carry or self.spatial_mode != 'local':
            raise ValueError('Reviewed control requires history3+H5,carry,local1024')
        if any([self.spatial_basis != 'identity', self.spatial_chart != 'none',
                self.spatial_readout_mode != 'none', self.spatial_state_interface != 'chart']):
            raise ValueError('Do not mix new objectives with other coordinate interventions')
        self.predictor_kind = predictor_kind
        if predictor_kind == 'spatial':
            self.predictor = SpatialNativePredictor(self.predictor, spatial_predictor_seed)
        self.renderer = None
        self.method = 'fixed_clean_forecast'

    def loss(self, batch, step=0, ramp_steps=0):
        info = self.encode(dict(pixels=self.normalize(batch['pixels']), action=batch['action']))
        observed, actions = info['emb'], info['act_emb']
        if observed.shape[1] != 8:
            raise ValueError('Expected exactly three context and five target observations')
        devices = [observed.device.index] if observed.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.forecast_seed + int(step))
            history = list(observed[:, :3].unbind(1)); future = []
            for t in range(2, 7):
                prediction = self.predict(torch.stack(history, 1), actions[:, :t+1])[:, -1]
                history.append(prediction);future.append(prediction)
            future = torch.stack(future, 1)
        per_clip = (future.float() - observed[:, 3:].float()).square().mean((1, 2))
        loss = per_clip.mean()
        return loss, dict(loss=loss, pred=loss, feature_std=observed.float().std((0, 1)).mean(),
                          forecast_history=loss.new_tensor(3), forecast_horizon=loss.new_tensor(5))

    def context_samples(self, mu, samples, generator):
        return mu.expand(samples, -1, -1)

    def transition_noise(self, value, generator):
        return torch.zeros_like(value)

    def audit_expected_gradients(self):
        result = dict(encoder='frozen', pool='frozen', predictor='active', action_encoder='active')
        result.update({f'predictor.transformer.layers.{i}': 'active' for i in range(6)})
        return result
