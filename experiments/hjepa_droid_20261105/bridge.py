"""DROID interfaces only; the existing candidate's neural model/loss are unchanged."""
import os
from pathlib import Path
import sys
import types


def install_paths(root):
    root = Path(root)
    sys.path[:0] = [str(root / 'public/h_jepa'), str(root / 'public'),
                    str(root / 'candidate'), str(root / 'candidate/runtime')]
    # The newer public package dropped the historical wm namespace used by our model.
    # Add only that namespace; all public data/planning classes remain the pinned release.
    import stable_worldmodel
    if 'stable_worldmodel.wm' not in sys.modules:
        package = types.ModuleType('stable_worldmodel.wm')
        package.__path__ = [str(root / 'compat_wm')]
        sys.modules['stable_worldmodel.wm'] = package
        stable_worldmodel.wm = package
    # Import the unchanged model primitives without the unrelated Python-3.12
    # training/checkpoint-manager facade. The public training stack is Python 3.11.
    for name in ['vforge', 'vforge.models', 'vforge.distributed']:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / 'candidate/runtime' / name.replace('.', '/'))]
            sys.modules[name] = package


def candidate_config(seed):
    return dict(mode='L2', beta=.25, sigma0=.1, window=8, anchor_gap=0,
                renderer_width=384, renderer_depth=12, renderer_heads=6,
                action_dim=7, encoder_size=256, rho=.8, forecast_fraction=.5,
                forecast_seed=20261029 + int(seed) * 100003)


def build_candidate(arm, seed):
    if arm == 'temporal-forecast':
        from as_jepa_lewm.robust import RobustLeWM
        constructor = RobustLeWM
    elif arm == 'innovation':
        from as_jepa_lewm.innovation import InnovationLeWM
        constructor = InnovationLeWM
    else:
        raise ValueError(arm)
    return constructor(**candidate_config(seed))


def candidate_batch(batch):
    import torch
    from torch.nn import functional as F
    pixels = batch['pixels_level1']
    actions = batch['action_level1']
    assert pixels.dtype == torch.uint8 and pixels.shape[1:] == (8, 3, 256, 256)
    assert actions.shape[1:] == (8, 7)
    small = F.interpolate(pixels.flatten(0, 1).float(), size=(64, 64), mode='area').round()
    small = small.unflatten(0, pixels.shape[:2])
    # The first image is the sole training anchor; no older or future extra frame is read.
    return dict(pixels=pixels, action=actions, render=small, anchor=small[:, 0],
                anchor224=pixels[:, 0])


def cost_adapter(core):
    from models.jepa import JEPA
    import torch

    class TemporalCost(JEPA):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.core = core
            self.action_embed = core.action_encoder
            self.level = 1

        def encode(self, info, key='pixels', **kwargs):
            value = dict(pixels=info[key])
            if 'action' in info:
                value['action'] = info['action']
            encoded = self.core.encode(value)
            info['embed_0'] = encoded['emb']
            if 'act_emb' in encoded:
                info['action_0'] = encoded['act_emb']
            return info

        def predict(self, emb, act_emb):
            return self.core.predict(emb, act_emb)

    # Rollout, one-state planning history, terminal goal cost and action gradients
    # are the public JEPA methods, rather than a second implementation of the solver.
    return TemporalCost()
