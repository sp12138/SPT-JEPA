"""Read-only gradient/saturation audit at the last sealed mean checkpoint."""
import argparse
from pathlib import Path
import itertools
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.data import WindowSampler
from experiments.metaworld_joint_flow_20261109.data import EndpointWindows,add_endpoint
from experiments.metaworld_joint_flow_20261109.model import random_terms
from experiments.metaworld_joint_flow_20261109.train import setup,sources


def fp32_nonlinear(self,hidden,context):
    update,reset=self.gates(torch.cat([hidden,context],1)).float().sigmoid().chunk(2,1)
    candidate=self.candidate(torch.cat([hidden.float()*reset,context.float()],1)).float().tanh()
    return hidden.float()*(1-update)+candidate*update


def run(config,output):
    cfg=read(config);device,model,_,_=setup(cfg);model.train();assert torch.cuda.device_count()==4
    path=Path(cfg['output'])/'checkpoint-0001500.pt';p=torch.load(path,map_location='cpu',weights_only=False)
    assert p['source']==sources() and p['config']==cfg and p['step']==1500
    model.load_state_dict(p['model'],strict=True);before=tensor_sha(model)
    data=EndpointWindows(cfg['data'],'train',maximum_horizon=60)
    # Reconstruct the original step1600 random terms, without replaying SGD.
    # The model is still step1500: this is NOT a reproduction of its1599 state.
    torch.cuda.set_rng_state_all(p['rng']['cuda'])
    for step in range(1501,1601):
        indices=list(itertools.islice(WindowSampler(data,(step-1)*64,cfg['seed'],64),64))
        lengths=[]
        for index in indices:
            record,t=data.locate(index);lengths.append(min(60,record['length']-1-t))
        small={'future_valid':torch.arange(60,device=device)[None]<torch.tensor(lengths,device=device)[:,None]}
        terms=random_terms(small)
    batch={k:v.to(device) for k,v in default_collate([data[i] for i in indices]).items()};add_endpoint(data,batch,terms[0],device)
    cls=type(model.parent.transition);original=cls.forward;reports={}
    for mode in ['bf16_original','fp32_all','bf16_convs_fp32_recurrent_nonlinearity']:
        model.zero_grad(set_to_none=True);cls.forward=fp32_nonlinear if mode.endswith('nonlinearity') else original
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode!='fp32_all'):
            loss,metrics=torch.nn.DataParallel(model,device_ids=[0,1,2,3])(batch,terms);loss=loss.mean()
        assert torch.isfinite(loss);loss.backward();g={}
        for name,module in dict(vision=model.parent.vision,observer=model.parent.observer,control=model.parent.control,
                transition=model.parent.transition,renderer=model.renderer,prior_scale=model.prior_scale).items():
            grads=[v.grad.detach() for v in module.parameters() if v.grad is not None]
            g[name]=dict(norm_FP32=sum(float(v.float().square().sum()) for v in grads)**.5,
                norm_FP64=sum(float(v.double().square().sum()) for v in grads)**.5,
                nonzero_coordinates=sum(int(torch.count_nonzero(v)) for v in grads),
                finite=all(bool(torch.isfinite(v).all()) for v in grads))
        reports[mode]=dict(loss=float(loss.detach()),gradients=g)
    cls.forward=original;assert before==tensor_sha(model)
    save(output,dict(passed=True,utc=now(),weights_step=1500,batch_and_random_terms_step=1600,
        not_reproduced_1599_weights=True,checkpoint_sha256=sha(path),state_before=before,state_after=before,
        reports=reports,durations=terms[0].cpu().tolist(),episodes=batch['episode'].cpu().tolist(),starts=batch['start'].cpu().tolist(),
        action_nonzero_fraction=float((batch['actions']!=0).float().mean()),
        optimizer_updates=0,physics_steps=0,scope='Read-only numeric audit. Allthree useSAMEweights,batch,randomterms. No modificationoflearnedmodels or originalfrozenruntimefiles.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.config,a.output)
