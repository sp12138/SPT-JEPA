"""Causal messages between synchronized views of one physical world.

The paired self-view control has identical parameters and initialization. The
only contrast is whether the second input comes from the same sensor or the
other sensor. Independent physical examples never exchange information.
"""
import types

import torch
from torch import nn


class ViewMessage(nn.Module):
    def __init__(self,width,mode,hidden=192):
        super().__init__()
        assert mode in ("self","cross")
        self.mode=mode
        self.net=nn.Sequential(nn.LayerNorm(3*width),nn.Linear(3*width,hidden),
                               nn.GELU(),nn.Linear(hidden,width))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,state,action):
        assert state.ndim==3 and state.shape==action.shape and state.shape[0]%2==0
        if self.mode=="cross":
            other=state.reshape(-1,2,*state.shape[1:]).flip(1).reshape_as(state)
        else:other=state
        return self.net(torch.cat([state,other,action],-1))


def _predict(self,state,action):
    return self._sensor_predict(state,action)+self.view_bridge(state,action)


def attach(model,mode,seed):
    """Zero residual preserves the parent's initial predictor exactly."""
    assert mode in ("self","cross") and not hasattr(model,"view_bridge")
    # The extra initialization must not perturb posterior/image-noise streams.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(293011+int(seed))
        model.view_bridge=ViewMessage(model.state_dim,mode)
    model._sensor_predict=model.predict
    model.predict=types.MethodType(_predict,model)
    return model
