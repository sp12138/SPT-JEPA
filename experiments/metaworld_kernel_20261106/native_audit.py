"""Native full-state restore,future-image loss,and forward-only information gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from experiments.metaworld_kernel_20261106.audit import equal


def run(a):
    from as_jepa_lewm.train import TrainModule
    from as_jepa_tools.native_rgb_pairs import NativeRGBPairs
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((a.run/'config.json').read_text());saved=torch.load(a.run/'last.ckpt',map_location='cpu',weights_only=False)
    module=TrainModule(cfg).cpu();module.load_state_dict(saved['state_dict'],strict=True)
    optimizer=module.configure_optimizers();optimizer['optimizer'].load_state_dict(saved['optimizer_states'][0]);optimizer['lr_scheduler']['scheduler'].load_state_dict(saved['lr_schedulers'][0])
    restored=dict(model=equal(module.state_dict(),saved['state_dict']),optimizer=equal(optimizer['optimizer'].state_dict(),saved['optimizer_states'][0]),
        scheduler=equal(optimizer['lr_scheduler']['scheduler'].state_dict(),saved['lr_schedulers'][0]))
    assert all(restored.values())and saved['global_step']==a.step
    weights=torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cpu',weights_only=True);assert equal(module.model.state_dict(),weights)
    del saved,optimizer,weights
    if cfg['dataset_kind']=='goal_rgb_paths':
        from as_jepa_tools.goal_rgb_paths import GoalRGBPaths
        data=GoalRGBPaths(cfg,'test');chosen=np.linspace(0,len(data)-1,8,dtype=int).tolist()
        assert equal(module.model.world.state_dict(),torch.load(cfg['model']['world_checkpoint'],map_location='cpu',weights_only=True))
    else:
        data=NativeRGBPairs(cfg,'test');indices={0:[],1:[]}
        for index in np.linspace(0,len(data)-1,min(256,len(data)),dtype=int):
            role=int(data[int(index)]['boundary'])
            if len(indices[role])<4:indices[role].append(int(index))
            if all(len(v)==4 for v in indices.values()):break
        assert all(len(v)==4 for v in indices.values());chosen=indices[0]+indices[1]
    batch=next(iter(DataLoader(Subset(data,chosen),batch_size=8)))
    assert 'action_target'not in batch and 'future_actions'not in batch
    mask=batch['boundary'].bool();assert(batch['target_row'][mask]!=batch['goal_row'][mask]).all()and(batch['executed_control'][mask]==0).all()
    model=module.model.cuda().eval();batch={k:v.cuda()if torch.is_tensor(v)else v for k,v in batch.items()}
    loss,metrics=model.loss(batch);assert torch.isfinite(loss);loss.backward();gradients={}
    for prefix,contract in model.audit_expected_gradients().items():
        params=[p for n,p in model.named_parameters()if n.startswith(prefix+'.')]
        norm=sum(float(p.grad.float().square().sum())for p in params if p.grad is not None)**.5
        assert all(torch.isfinite(p.grad).all()for p in params if p.grad is not None)
        if contract=='active':assert norm>0,(prefix,norm)
        else:assert not any(p.requires_grad for p in params)
        gradients[prefix]=dict(contract=contract,norm=norm)
    with torch.no_grad():
        single=[model.loss({k:v[j:j+1]if torch.is_tensor(v)else v for k,v in batch.items()})[0]for j in range(8)]
        independent=float((torch.stack(single).mean()-loss.detach()).abs())
        relative=independent/(1+abs(float(loss.detach())));assert relative<2e-5,(independent,relative)
        image=model.encoder(batch['current_pixels'][:1]);identity=float((model.warp(image,torch.zeros(1,2,128,128,device='cuda'))-image).abs().max());assert identity<1e-6,identity
        action,future=model.command(image,image,model.encoder(batch['goal_pixels'][:1]),image.new_ones(1),'projection',batch['previous_control'][:1])
        assert action.abs().max()<=1 and torch.isfinite(future).all()
    report=dict(passed=True,step=a.step,actual_cpu_restore=restored,portable_exact=True,gradients=gradients,
        sample_loss_independence_max_abs=independent,sample_loss_independence_relative=relative,identity_warp_max_abs=identity,loss=float(loss.detach()),metrics={k:float(v.detach())for k,v in metrics.items()},
        heldout_query_indices=chosen,no_inverse_or_control_coordinate_objective=True,no_future_action_reference=True,model_updates=0,
        scope='Only observed future RGB is the target.Known current command inputs forward prediction only;goal query commands zeroed.'
              'Planning is never called in the training loss.No external pretrained model/state/reward/success/taskID input.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,step=a.step,loss=float(loss.detach()))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
