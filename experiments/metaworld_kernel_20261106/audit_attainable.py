"""Actual native state restore and no-action-target audit for attainable forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from experiments.metaworld_kernel_20261106.audit import equal


def run(a):
    from as_jepa_lewm.train import TrainModule
    from as_jepa_tools.attainable_pairs import AttainablePairs
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((a.run/'config.json').read_text());saved=torch.load(a.run/'last.ckpt',map_location='cpu',weights_only=False)
    module=TrainModule(cfg).cpu();module.load_state_dict(saved['state_dict'],strict=True)
    optimizer=module.configure_optimizers();optimizer['optimizer'].load_state_dict(saved['optimizer_states'][0])
    optimizer['lr_scheduler']['scheduler'].load_state_dict(saved['lr_schedulers'][0])
    restored=dict(model=equal(module.state_dict(),saved['state_dict']),optimizer=equal(optimizer['optimizer'].state_dict(),saved['optimizer_states'][0]),
        scheduler=equal(optimizer['lr_scheduler']['scheduler'].state_dict(),saved['lr_schedulers'][0]))
    assert all(restored.values())and saved['global_step']==a.step
    portable=torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cpu',weights_only=True)
    assert equal(module.model.state_dict(),portable)
    parent=Path(cfg['model']['parent']);parent_weights=torch.load(parent/'model-step0010000/weights.pt',map_location='cpu',weights_only=True)
    assert equal(module.model.world.state_dict(),parent_weights)
    del saved,optimizer,portable,parent_weights
    model=module.model.cuda().eval();data=AttainablePairs(cfg,'test');indices=np.linspace(0,len(data)-1,8,dtype=int).tolist()
    batch=next(iter(DataLoader(Subset(data,indices),batch_size=8)))
    assert not any(k in batch for k in['executed_control','action_target','future_actions'])
    assert (batch['target_row'][batch['goal_valid'].bool()]!=batch['goal_row'][batch['goal_valid'].bool()]).all()
    batch={k:v.cuda()if torch.is_tensor(v)else v for k,v in batch.items()}
    loss,metrics=model.loss(batch);assert torch.isfinite(loss);loss.backward();gradients={}
    for prefix,contract in model.audit_expected_gradients().items():
        params=[p for name,p in model.named_parameters()if name.startswith(prefix+'.')]
        norm=sum(float(p.grad.float().square().sum())for p in params if p.grad is not None)**.5
        assert all(torch.isfinite(p.grad).all()for p in params if p.grad is not None)
        if contract=='active':assert norm>0,(prefix,norm)
        else:assert not any(p.requires_grad for p in params)
        gradients[prefix]=dict(contract=contract,norm=norm)
    with torch.no_grad():
        independent=[model.loss({k:v[j:j+1]if torch.is_tensor(v)else v for k,v in batch.items()})[0]for j in range(len(batch['current_z']))]
        difference=float((torch.stack(independent).mean()-loss.detach()).abs());assert difference<2e-5,difference
        command,future=model.command(batch['previous_z'][:2],batch['current_z'][:2],batch['goal_z'][:2],batch['history_valid'][:2],'projection',batch['previous_control'][:2])
        assert torch.isfinite(future).all()and command.abs().max()<=1
    out=dict(passed=True,step=a.step,actual_cpu_restore=restored,portable_exact=True,parent_world_exact_and_frozen=True,
        loss=float(loss.detach()),metrics={k:float(v.detach())for k,v in metrics.items()},gradients=gradients,
        sample_loss_independence_max_abs=difference,current_future_action_fields_absent=True,boundary_target_not_in_inputs=True,
        scope='Only future-observation prediction targets.No current/future action target,pseudolabel,or reference.'
              'Own same-corpus reconstruction-free world pretraining cost must count.NoNNupdate during audit.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,step=a.step,loss=float(loss.detach()))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
