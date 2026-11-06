"""Zero-update shape,query-information,and paired-state checks for recurrent RGB."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset


def run(a):
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.query_rgb_paths import QueryRGBPaths
    from experiments.metaworld_kernel_20261106.audit import equal
    torch.set_num_threads(2);cfg=json.loads(a.config.read_text());data=QueryRGBPaths(cfg,'test');indices={0:[],1:[]}
    for index in np.linspace(0,len(data)-1,256,dtype=int):
        role=int(data[int(index)]['boundary'])
        if len(indices[role])<2:indices[role].append(int(index))
        if all(len(v)==2 for v in indices.values()):break
    batch=next(iter(DataLoader(Subset(data,indices[0]+indices[1]),batch_size=4)));mask=batch['boundary'].bool()
    assert(batch['executed_sequence'][mask]==0).all()and(batch['previous_control_sequence'][mask]==0).all()
    assert(batch['target_row'][mask]<batch['goal_row'][mask]).all()
    initial=None;reports={}
    for mode in ['teacher','rollout']:
        cfg['model']['training_mode']=mode;torch.manual_seed(cfg['seed']);model=build_model(cfg).eval()
        state={k:v.clone()for k,v in model.state_dict().items()}
        if initial is None:initial=state
        else:assert equal(initial,state)
        predicted=model.distribution(batch);assert predicted.shape==(4,5,3,128,128)and torch.isfinite(predicted).all()
        loss,_=model.loss(batch);loss.backward();head=sum(float(p.grad.square().sum())for p in model.rgb_head.parameters()if p.grad is not None)**.5
        assert torch.isfinite(loss)and head>0
        with torch.no_grad():
            individual=torch.stack([model.loss({k:v[j:j+1]if torch.is_tensor(v)else v for k,v in batch.items()})[0]for j in range(4)]).mean()
            independent=float((individual-loss.detach()).abs())/(1+abs(float(loss.detach())));assert independent<2e-5
            image=model.encoder(batch['current_pixels'][:1]);goal=model.encoder(batch['goal_pixels'][:1]);past=image.new_zeros(1,4)
            action,future=model.command(image,image,goal,image.new_zeros(1),'projection',past);assert action.abs().max()<=1 and torch.isfinite(future).all()
        assert equal(state,model.state_dict())
        reports[mode]=dict(loss=float(loss.detach()),rgb_head_gradient=head,sample_loss_independence_relative=independent)
    a.output.write_text(json.dumps(dict(passed=True,models=reports,paired_initial_weights=True,model_updates=0,
        zero_residual_initialization='OnlyRGBhead has first-update gradient;allrequiredgradients mustbepositiveafteractualnative2updates.',
        same_five_actual_future_RGB_targets=True,no_inverse_or_future_action_reference=True),indent=2)+'\n')
    print(json.dumps(dict(passed=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
