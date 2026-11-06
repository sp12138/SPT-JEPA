"""A goal forecast restricted to the learned forward model's control span.

The only fitting target is an observed NEXT feature. The goal branch never
reads a current/future demonstrated command or an inferred action label.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn

from as_jepa_lewm.predictive_density import PredictiveDensity,quadratic_box


class AttainableForecast(nn.Module):
    def __init__(self,parent,feature_seed=1,boundary=True,window=2):
        super().__init__();self.window=window;self.renderer=None
        parent=Path(parent);cfg=json.loads((parent/'config.json').read_text())
        assert cfg['model_class']=='predictive_density'and cfg['model']['feature_seed']==feature_seed
        origin=json.loads((parent/'ORIGIN.json').read_text())
        assert origin['no_reconstruction']and origin['no_action_targets_or_references']
        self.world=PredictiveDensity(**cfg['model'])
        weights=torch.load(parent/'model-step0010000/weights.pt',map_location='cpu',weights_only=True)
        self.world.load_state_dict(weights,strict=True);self.world.requires_grad_(False)
        # All inherited tensors originate in our own same-data,strict future-
        # observation training. Its entire10k-update budget must count.
        self.goal_body=copy.deepcopy(self.world.predictor).requires_grad_(True)
        self.goal_context=copy.deepcopy(self.world.context).requires_grad_(True)
        width=self.world.mean_head.in_channels
        self.control_coordinates=nn.Sequential(nn.Flatten(),nn.Linear(width*16*16,256),nn.SiLU(),nn.Linear(256,4))
        nn.init.zeros_(self.control_coordinates[-1].weight);nn.init.zeros_(self.control_coordinates[-1].bias)

    @property
    def encoder(self):return self.world.encoder

    def coefficients(self,previous,current,goal,valid,past):
        x=self.goal_body(torch.cat([previous,current,goal-current],1))
        context=torch.cat([past if self.world.actuation_history else torch.zeros_like(past),valid.new_ones(len(valid),1),valid[:,None]],-1)
        gamma,beta=self.goal_context(context.float()).chunk(2,-1)
        return self.control_coordinates(x*(1+.1*gamma[:,:,None,None])+beta[:,:,None,None]).float()

    def components(self,previous,current,valid,past):
        with torch.no_grad():
            zero=current.new_zeros(len(current),4);flag=valid.new_zeros(len(valid))
            return self.world.fields(previous,current,torch.zeros_like(current),zero,flag,valid,past)

    def loss(self,batch,step=0,ramp_steps=0):
        # There deliberately is no executed_control/action_target field here.
        assert 'executed_control'not in batch and 'action_target'not in batch and 'future_actions'not in batch
        previous,current,goal=[batch[k]for k in['previous_z','current_z','goal_z']]
        valid,past=batch['history_valid'],batch['previous_control']
        drift,effect,_=self.components(previous,current,valid,past)
        coordinates=self.coefficients(previous,current,goal,valid,past)
        prediction=current+drift+torch.einsum('na,nachw->nchw',coordinates,effect)
        error=(prediction-batch['next_z']).square().flatten(1).mean(1)
        mask=batch['goal_valid'];loss=(.5*error*mask).mean()
        return loss,dict(loss=loss,future_mse=(error*mask).sum()/mask.sum().clamp_min(1),
            coefficient_rms=coordinates.square().mean().sqrt())

    @torch.no_grad()
    def command(self,previous,current,goal,valid,objective,previous_control=None):
        past=current.new_zeros(len(current),4)if previous_control is None else previous_control
        if objective=='marginal':return self.world.command(previous,current,goal,valid,'marginal',past)
        assert objective=='projection'
        drift,effect,variance=self.components(previous,current,valid,past)
        coordinates=self.coefficients(previous,current,goal,valid,past)
        desired=drift+torch.einsum('na,nachw->nchw',coordinates,effect)
        incremental=self.world.passive in['velocity','transport']
        center=past if incremental else torch.zeros_like(past)
        if incremental:drift=drift-torch.einsum('na,nachw->nchw',past,effect)
        basis=effect.flatten(2).transpose(1,2).double();weight=variance.flatten(1).double().reciprocal()
        H=torch.einsum('npi,np,npj->nij',basis,weight,basis)
        g=torch.einsum('npi,np,np->ni',basis,weight,(desired-drift).flatten(1).double())
        action=quadratic_box(H,g,center)
        future=current+drift+torch.einsum('na,nachw->nchw',action,effect)
        return action,future

    def training_optimizer_config(self,max_steps):
        return PredictiveDensity.training_optimizer_config(self,max_steps)

    def audit_expected_gradients(self):
        return dict(world='frozen',goal_body='active',goal_context='active',control_coordinates='active')
