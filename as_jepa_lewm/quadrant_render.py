"""Global predictive states supervised by one addressed local visual query.

All encoder/predictor observations remain the original full images. Only the
training renderer's target and anchor become the same randomly chosen quadrant.
The known query address bypasses the state bottleneck; it is not a state label.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.robust import RobustLeWM
from vforge.models.as_jepa.renderer import AnchoredWorldDiT


class QueryAnchoredWorldDiT(AnchoredWorldDiT):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Additional query parameters leave every legacy tensor/outer CPU/CUDA
        # stream unchanged, including when a CUDA context already exists.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261101)
            self.query_projection = nn.Linear(2, config.width)
        self._query_address = None

    @contextmanager
    def queried(self, address):
        if self._query_address is not None:
            raise RuntimeError('Nested renderer queries are not supported')
        if address.ndim != 2 or address.shape[1] != 2:
            raise ValueError('Expected one known [row,column] address per target')
        self._query_address = address
        try:
            yield self
        finally:
            self._query_address = None

    def _condition(self, tokens, batch, dtype, device, drop):
        context = super()._condition(tokens, batch, dtype, device, drop)
        if self._query_address is None:
            return context
        if self._query_address.shape[0] != batch:
            raise ValueError('Query and rendered target batches differ')
        # The output address is known even when classifier-free dropout hides
        # the state. It therefore enters after that original dropout operation.
        query = self.query_projection(self._query_address.to(
            device=device, dtype=self.query_projection.weight.dtype)).to(dtype)
        return torch.cat([context, query[:, None]], dim=1)


class QuadrantRenderLeWM(RobustLeWM):
    renderer_type = QueryAnchoredWorldDiT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.domain != 'standard' or self.mode != 'L2' or self.objective != 'flow':
            raise ValueError('Reviewed quadrant queries require standard L2 flow')
        if self.response_query_mode != 'none' or self.sample_forecast:
            raise ValueError('This control isolates local rendering from other new mechanisms')

    @staticmethod
    def quadrants(images, choices):
        """[B,...,C,224,224] -> matching112x112 crops; no interpolation here."""
        b = images.shape[0]
        if images.shape[-2:] != (224, 224) or choices.shape != (b,):
            raise ValueError('Expected224px images and one quadrant index per clip')
        if choices.dtype != torch.long or not bool(((choices >= 0) & (choices < 4)).all()):
            raise ValueError('Quadrant indices must be integers0..3')
        # Stacking views preserves exact source pixels, including at boundaries.
        tiles = torch.stack([images[..., y:y+112, x:x+112]
                             for y in [0, 112] for x in [0, 112]], dim=1)
        return tiles[torch.arange(b, device=images.device), choices.to(images.device)]

    def render_query_batch(self, batch, step):
        b = batch['pixels'].shape[0]
        generator = torch.Generator(device='cpu').manual_seed(
            self.forecast_seed + 0x51554552 + int(step))
        choices = torch.randint(4, (b,), generator=generator, device='cpu')
        target = self.quadrants(batch['pixels'], choices)
        anchor = self.quadrants(batch['anchor224'], choices)
        modified = dict(batch)
        modified['render'] = F.interpolate(target.float().flatten(0, 1),
            size=(64, 64), mode='area').round().unflatten(0, target.shape[:2])
        modified['anchor'] = F.interpolate(anchor.float(), size=(64, 64), mode='area').round()
        address = torch.stack([choices // 2, choices % 2], dim=-1).float() - .5
        return modified, address.to(batch['pixels'].device)

    def loss(self, batch, step=0, ramp_steps=0):
        queried, address = self.render_query_batch(batch, step)
        targets = batch['pixels'].shape[1] - 1
        with self.renderer.queried(address.repeat_interleave(targets, dim=0)):
            return super().loss(queried, step=step, ramp_steps=ramp_steps)

    def audit_expected_gradients(self):
        return {'encoder': 'active', 'predictor': 'active', 'renderer': 'active',
                'renderer.query_projection': 'active'}

    def render_states(self, states, anchor, steps=50, seed=0, *, quadrant):
        """Generate a64px query; anchor is that normalized64px quadrant.

        Planning uses only encode/predict and does not require an address.
        """
        if quadrant not in range(4):
            raise ValueError('Supply a quadrant index0..3 with its matching crop anchor')
        address = states.new_tensor([quadrant // 2 - .5, quadrant % 2 - .5])
        address = address[None].expand(len(states), -1)
        with self.renderer.queried(address):
            return super().render_states(states, anchor, steps=steps, seed=seed)
