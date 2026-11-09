"""CPU contract checks on real native windows, with no parameter updates."""
import argparse
import copy
from pathlib import Path
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import read,save,now,tensor_sha
from experiments.metaworld_transition_only_20261109.data import FactualWindows
from .train import setup
from .model import factual_condition
from .energy import log_discretized


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not torch.cuda.is_available();cfg=read(a.config);_,parent,head,_,_=setup(cfg,False)
    data=FactualWindows(cfg['data'],'train',maximum_horizon=60);batch=default_collate([data[0],data[31]])
    h=torch.minimum(torch.tensor([1,3]),batch['future_valid'].sum(1))
    first=factual_condition(parent,batch,h);poison=copy.deepcopy(batch)
    for i,duration in enumerate(h):
        poison['actions'][i,int(duration):]=float('nan');poison['future_pixels'][i,int(duration):]=255
    second=factual_condition(parent,poison,h)
    assert all(torch.equal(x,y) for x,y in zip(first,second))
    hidden,current,target=first;noise=torch.randn_like(target);time=torch.tensor([.2,.9])
    with torch.no_grad():
        two=head(noise,time,hidden,current,h)
        one=head(noise[:1],time[:1],hidden[:1],current[:1],h[:1])
        changed=head(torch.cat([noise[:1],noise[1:]*100]),time,torch.cat([hidden[:1],hidden[1:]*7]),current,h)
    assert torch.allclose(two[:1],one,atol=1e-6,rtol=1e-5) and torch.equal(two[:1],changed[:1])
    grid=torch.linspace(-1,1,256)[:,None];means=torch.tensor([[-2.,-.3,0.,.7,2.]])
    mass=log_discretized(grid,means,.1).exp().sum(0)
    assert torch.allclose(mass,torch.ones_like(mass),atol=2e-5,rtol=0),mass
    save(a.output,dict(passed=True,utc=now(),train_starts=len(data),episodes=len(data.episode_ids),
        after_duration_RGB_and_action_poison_invariance=True,actual_window_sample_independence=True,
        normalized_discrete_Gaussian_endpoint_mass=mass.tolist(),
        parent_frozen=True,head_random_init=True,model_updates=0,gpu_contexts=0,
        supervision='OnlyobservedfutureRGBtargetsandknownexecutedinputs. Desiredgoal/path/policy/action-targetbranchesabsent.'))
