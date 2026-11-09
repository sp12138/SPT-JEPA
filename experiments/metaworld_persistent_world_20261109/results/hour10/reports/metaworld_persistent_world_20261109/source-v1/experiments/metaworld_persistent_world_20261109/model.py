"""Controlled latent persistence, without paths, action targets, or goal training.

Parallel residual and sequential reversible coupling use identical parameter
values and operations. Only whether G reads u or updated u' differs. The
sequential form has an exact algebraic inverse and unit Jacobian determinant;
this is a standard additive-coupling principle, not a novelty claim.
"""
import torch
from torch import nn
from experiments.metaworld_transition_only_20261109.model import ConvGRU
from experiments.metaworld_boundary_world_20261109.model import BoundaryWorld,random_terms


class Coupling(nn.Module):
    def __init__(self,reversible):
        super().__init__();self.reversible=reversible
        def net():
            value=nn.Sequential(nn.Conv2d(64+96,128,3,padding=1),nn.GroupNorm(8,128),nn.SiLU(),nn.Conv2d(128,64,3,padding=1))
            nn.init.normal_(value[-1].weight,std=.001);nn.init.zeros_(value[-1].bias);return value
        self.f=net();self.g=net()

    def forward(self,hidden,context):
        u,v=hidden.chunk(2,1);updated=u+self.f(torch.cat([v,context],1))
        last=v+self.g(torch.cat([updated if self.reversible else u,context],1))
        return torch.cat([updated,last],1)

    def invert_state(self,hidden,context):
        assert self.reversible
        updated,last=hidden.chunk(2,1);v=last-self.g(torch.cat([updated,context],1));u=updated-self.f(torch.cat([v,context],1))
        return torch.cat([u,v],1)


class PersistentWorld(BoundaryWorld):
    def __init__(self,origin,kind):
        nn.Module.__init__(self);assert kind in ['gru','residual','reversible']
        self.parent=origin.parent;self.direction=origin.direction;self.readout=origin.readout;self.mode='residual';self.dynamics_kind=kind
        self.parent.transition=ConvGRU(128,96) if kind=='gru' else Coupling(kind=='reversible')

    def objective(self,batch,terms):
        assert not {'goal_pixels','boundary','desired_goal','reward','state','task_id','expert_actions'}&set(batch)
        durations=terms[0];direction=torch.zeros_like(durations)
        hidden,current=self.prepare(batch,direction);last=hidden
        for t in range(int(durations.max())):
            hidden=self.advance(hidden,batch['actions'][:,t],direction)
            last=torch.where((durations==t+1)[:,None,None,None],hidden,last)
        prediction=self.render(last,current);target=self.parent.pixels(batch['endpoint_pixels'])
        per=(prediction-target).square().mean((1,2,3));loss=per.mean()
        return loss,dict(loss=loss,RGB_MSE=loss,duration=durations.float().mean(),endpoint_hidden_rms=last.square().mean().sqrt())


def backbone_sha(model):
    import hashlib
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('parent.transition.'):continue
        h.update(name.encode());h.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
