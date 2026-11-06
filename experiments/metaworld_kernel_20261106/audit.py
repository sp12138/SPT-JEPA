"""Native full-state/gradient/independence and forbidden-information audits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b)and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys()and all(equal(a[k],b[k])for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    return a==b


def run(a):
    from as_jepa_lewm.train import TrainModule
    from as_jepa_tools.predictive_kernel_data import KernelTransitions
    from as_jepa_tools.predictive_density_data import DensityTransitions
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((a.run/'config.json').read_text());saved=torch.load(a.run/'last.ckpt',map_location='cpu',weights_only=False)
    module=TrainModule(cfg).cpu();module.load_state_dict(saved['state_dict'],strict=True)
    optim=module.configure_optimizers();optim['optimizer'].load_state_dict(saved['optimizer_states'][0]);optim['lr_scheduler']['scheduler'].load_state_dict(saved['lr_schedulers'][0])
    restored=dict(model=equal(module.state_dict(),saved['state_dict']),optimizer=equal(optim['optimizer'].state_dict(),saved['optimizer_states'][0]),scheduler=equal(optim['lr_scheduler']['scheduler'].state_dict(),saved['lr_schedulers'][0]))
    assert all(restored.values())and saved['global_step']==a.step
    weights=torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cpu',weights_only=True)
    assert equal(module.model.state_dict(),weights)
    initial=torch.load(a.run/'initial-model/weights.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,weights[k])for k,v in initial.items()if k.startswith('encoder.'))
    del saved,optim,weights,initial
    model=module.model.cuda().eval();model.requires_grad_(True);model.encoder.requires_grad_(False)
    if not model.gaussian:model.variance_head.requires_grad_(False)
    density=cfg['model_class']in['predictive_density','signed_density']
    data=(DensityTransitions if density else KernelTransitions)(cfg,'test');indices=np.linspace(0,len(data)-1,8,dtype=int).tolist()
    batch=next(iter(DataLoader(Subset(data,indices),batch_size=8)))
    assert 'action_target'not in batch and 'future_actions'not in batch
    mask=batch['goal_valid'].bool()if density else batch['boundary'].bool()
    assert (batch['target_row'][mask]!=batch['goal_row'][mask]).all()
    if not density:assert (batch['executed_control'][mask]==0).all()
    batch={k:v.cuda()if torch.is_tensor(v)else v for k,v in batch.items()}
    loss,metrics=model.loss(batch);loss.backward();assert torch.isfinite(loss)
    gradients={}
    for prefix,contract in model.audit_expected_gradients().items():
        params=[p for n,p in model.named_parameters()if n.startswith(prefix+'.')]
        norm=sum(float(p.grad.float().square().sum())for p in params if p.grad is not None)**.5
        assert all(torch.isfinite(p.grad).all()for p in params if p.grad is not None)
        if contract=='active':assert norm>0,(prefix,norm)
        else:assert not any(p.requires_grad for p in params)
        gradients[prefix]=dict(contract=contract,norm=norm)
    with torch.no_grad():
        if density:
            def query(b):return model.fields(b['previous_z'],b['current_z'],b['goal_z'],b['executed_control'],b['goal_valid'],b['history_valid'],b['previous_control'])
            full=query(batch);single=query({k:v[:1]if torch.is_tensor(v)else v for k,v in batch.items()})
        else:full=model.distribution(batch);single=model.distribution({k:v[:1]if torch.is_tensor(v)else v for k,v in batch.items()})
        independence=max(float((x[:1]-y).abs().max())for x,y in zip(full,single));assert independence<2e-5,independence
        if density:
            independent_losses=[model.loss({k:v[j:j+1]if torch.is_tensor(v)else v for k,v in batch.items()})[0]for j in range(len(batch['current_z']))]
            loss_independence=float((torch.stack(independent_losses).mean()-loss.detach()).abs())
            assert loss_independence<2e-5,loss_independence
        else:loss_independence=None
        if model.affine:
            z=batch['current_z'][:2];drift,effect,var=model.fields(batch['previous_z'][:2],z,torch.zeros_like(z),z.new_zeros(2,4),z.new_zeros(2),z.new_ones(2))
            command=model.box_projection(drift,effect,drift,var);assert command.abs().max()<1e-5
    report=dict(passed=True,step=a.step,actual_cpu_restore=restored,portable_exact=True,frozen_random_encoder_unchanged=True,
        loss=float(loss),metrics={k:float(v)for k,v in metrics.items()},gradients=gradients,sample_independence_max_abs=independence,loss_independence_max_abs=loss_independence,
        forbidden_information_audit=True,boundary_target_not_in_inputs=True,executed_action_input_only=True,
        scope='No reconstruction,no action labels/reference or oldweights.All targets observedNEXTfeature;noNNupdate in audit.Not performance.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,step=a.step,loss=float(loss))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
