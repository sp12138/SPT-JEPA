"""Observation-path integration gate,including15step backward,0updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset


def run(a):
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.goal_rgb_paths import GoalRGBPaths
    from experiments.metaworld_kernel_20261106.audit import equal
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    cfg=json.loads(a.config.read_text());reports={};initial=None
    for mode,horizon in [('point1',1),('jump5',5),('compose5',5),('compose15',15)]:
        cfg['model'].update(mode=mode,horizon=horizon);torch.manual_seed(cfg['seed'])
        data=GoalRGBPaths(cfg,'test');batch=next(iter(DataLoader(Subset(data,np.linspace(0,len(data)-1,4,dtype=int).tolist()),batch_size=4)))
        assert(batch['target_row']<batch['goal_row']).all()and(batch['executed_control']==0).all()
        model=build_model(cfg).to(a.device).eval();state={k:v.cpu().clone()for k,v in model.state_dict().items()}
        if initial is None:initial=state
        else:assert equal(initial,state)
        batch={k:v.to(a.device)if torch.is_tensor(v)else v for k,v in batch.items()};loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward();gradients={}
        for prefix,contract in model.audit_expected_gradients().items():
            params=[p for n,p in model.named_parameters()if n.startswith(prefix+'.')]
            norm=sum(float(p.grad.float().square().sum())for p in params if p.grad is not None)**.5
            assert all(torch.isfinite(p.grad).all()for p in params if p.grad is not None)
            if contract=='active':assert norm>0,(mode,prefix,norm)
            else:assert not any(p.requires_grad for p in params)
            gradients[prefix]=norm
        with torch.no_grad():
            single=torch.stack([model.loss({k:v[j:j+1]if torch.is_tensor(v)else v for k,v in batch.items()})[0]for j in range(4)]).mean()
            relative=float((single-loss.detach()).abs())/(1+abs(float(loss.detach())));assert relative<2e-5,relative
            image=model.encoder(batch['current_pixels'][:1]);goal=model.encoder(batch['goal_pixels'][:1])
            action,flow=model.choose_control(image,goal,image.new_zeros(1),image.new_zeros(1,4))
            assert torch.isfinite(flow).all()and action.abs().max()<=1
        assert equal(state,{k:v.cpu()for k,v in model.state_dict().items()})
        reports[mode]=dict(loss=float(loss.detach()),gradients=gradients,sample_loss_independence_relative=relative)
        del model
    a.output.write_text(json.dumps(dict(passed=True,models=reports,no_model_updates=True,paired_initial_weights=True,
        no_command_arrays_opened_by_generator_dataset=True,world_frozen=True,only_future_rgb_targets=True),indent=2)+'\n');print(json.dumps(dict(passed=True,modes=list(reports))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');run(p.parse_args())
