"""Own-forward ancestry,initial query parity,and observation-path fit,0updates."""
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
    cfg=json.loads(a.config.read_text());cfg['model_class']='goal_transport_warm';cfg['model'].update(mode='compose5',horizon=5)
    data=GoalRGBPaths(cfg,'test');batch=next(iter(DataLoader(Subset(data,np.linspace(0,len(data)-1,16,dtype=int).tolist()),batch_size=16)))
    torch.manual_seed(cfg['seed']);model=build_model(cfg).eval();before={k:v.clone()for k,v in model.state_dict().items()}
    current=model.encoder(batch['current_pixels']);goal=model.encoder(batch['goal_pixels']);zero=current.new_zeros(16,4);flag=current.new_zeros(16)
    with torch.no_grad():
        _,_,parent,_=model.world.fields(current,goal,flag+1,zero,flag)
        query_error=float((parent-model.velocity(current,goal)).abs().max());assert query_error<2e-5,query_error
        prediction=model.distribution(batch);target=model.encoder(batch['next_pixels']);mse=float((prediction-target).square().mean());copy=float((current-target).square().mean())
    loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward();gradients={}
    for prefix,contract in model.audit_expected_gradients().items():
        params=[p for n,p in model.named_parameters()if n.startswith(prefix+'.')]
        norm=sum(float(p.grad.float().square().sum())for p in params if p.grad is not None)**.5
        assert all(torch.isfinite(p.grad).all()for p in params if p.grad is not None)
        if contract=='active':assert norm>0,(prefix,norm)
        else:assert not any(p.requires_grad for p in params)
        gradients[prefix]=norm
    with torch.no_grad():
        individual=torch.stack([model.loss({k:v[j:j+1]if torch.is_tensor(v)else v for k,v in batch.items()})[0]for j in range(16)]).mean()
        independent=float((individual-loss.detach()).abs())/(1+abs(float(loss.detach())));assert independent<2e-5,independent
    assert equal(before,model.state_dict())
    a.output.write_text(json.dumps(dict(passed=True,model_updates=0,parent_forward_updates=20000,all_generator_parameters_finetunable=True,
        parent_goal_query_max_abs=query_error,initial_goal5_mse=mse,copy_mse=copy,gradients=gradients,sample_loss_independence_relative=independent,
        scope='Own same-data scratch future-observation parent only;allparentcostcounts.No inverse ancestry/actionlabels/futureactionreferences/externalpretraining.'),indent=2)+'\n')
    print(json.dumps(dict(passed=True,query_error=query_error,mse=mse,copy=copy)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
