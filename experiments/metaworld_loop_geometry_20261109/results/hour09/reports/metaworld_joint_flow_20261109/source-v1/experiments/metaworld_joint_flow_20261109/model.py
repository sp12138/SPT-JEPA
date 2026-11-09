"""Joint factual dynamics and endpoint flow; no goal or policy training."""
import torch
from experiments.metaworld_belief_state_20261109.model import BeliefWorld
from experiments.metaworld_endpoint_density_20261109.model import advance


class JointFlow(BeliefWorld):
    def __init__(self,parent,renderer,mode):
        super().__init__(parent,renderer,'joint_anchor');self.flow_mode=mode
        assert mode in ['mean','uniform','highnoise']

    def factual(self,batch,durations):
        assert not {'goal_pixels','desired_goal','boundary','reward','state','task_id','expert_actions'}&set(batch)
        assert torch.all((durations>=1)&(durations<=batch['future_valid'].sum(1)))
        hidden,current=self.prepare(batch)
        for t in range(int(durations.max())):
            next_hidden=self.call(lambda h,a:advance(self.parent,h,a),hidden,batch['actions'][:,t])
            hidden=torch.where((t<durations)[:,None,None,None],next_hidden,hidden)
        target=self.parent.pixels(batch['endpoint_pixels'])*2-1
        return hidden,current,target

    def objective(self,batch,random):
        durations,index,noise0,noise1,latent_noise,uniform=random
        prior,current,target=self.factual(batch,durations)
        logp=self.prior_scale(prior).float().clamp(-5,1)
        z=prior+logp.exp()*latent_noise
        time=torch.ones_like(uniform) if self.flow_mode=='mean' else uniform
        if self.flow_mode=='highnoise':time=4*time/(1+3*time)
        shape=time[:,None,None,None];loss=0
        for noise in [noise0,noise1]:
            bridge=(1-shape)*target+shape*noise
            velocity=self.velocity(bridge,time,z,current,durations)
            loss=loss+.5*(velocity-(target-noise)).square().flatten(1).mean(1)
        loss=loss.mean()
        return loss,dict(loss=loss,noise_time=time.mean(),prior_std=logp.exp().mean(),duration=durations.float().mean())


def random_terms(batch):
    from experiments.metaworld_belief_state_20261109.model import random_terms as previous
    terms=previous(batch)
    uniform=torch.rand(len(batch['future_valid']),device=batch['future_valid'].device).clamp(1e-5,1-1e-5)
    return (*terms,uniform)
