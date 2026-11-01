"""Native full-state restore/backward and frozen causal diagnostics on3 tasks."""
import argparse
from contextlib import nullcontext
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import dataset, read, same, sha, state_hash, write
from as_jepa_lewm.factory import build_model, gradient_expectations
from as_jepa_lewm.train import TrainModule, WindowSampler
from experiments.ideas_validate_20261031.audit import evaluate_latents, move_batch
from experiments.ideas_validate_20261031.preflight import checkpoint_report, compare_tensors, finite, gradient_report
from vforge.ops import OpsConfig, initialize_ops


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--diagnostics',action='store_true');a=p.parse_args()
    root=Path(a.root);cfg=read(root/a.id/'config.json');out=root/'audits'/f'{a.id}-step{a.step}'
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    run=root/a.id;checkpoint=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    portable=torch.load(run/f'model-step{a.step:07d}/weights.pt',map_location='cpu',weights_only=True)
    initial=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True)
    check=checkpoint_report(checkpoint,portable,cfg,a.step);assert all(check['checks'].values()),check
    torch.manual_seed(cfg['seed']);module=TrainModule(cfg)
    assert same(module.model.state_dict(),initial),'Fresh initial tensors differ from native constructor'
    module.load_state_dict(checkpoint['state_dict'],strict=True)
    configured=module.configure_optimizers();opt=configured['optimizer'];sched=configured['lr_scheduler']['scheduler']
    opt.load_state_dict(checkpoint['optimizer_states'][0]);sched.load_state_dict(checkpoint['lr_schedulers'][0])
    assert same(module.state_dict(),checkpoint['state_dict'])
    assert same(opt.state_dict(),checkpoint['optimizer_states'][0])
    assert same(sched.state_dict(),checkpoint['lr_schedulers'][0])
    params=[p for p in module.model.parameters() if p.requires_grad]
    actual=[p for g in opt.param_groups for p in g['params']]
    assert len(params)==len(actual) and {id(p) for p in actual}=={id(p) for p in params}
    assert all(p in opt.state for p in params)
    tr=dataset(cfg,'train');cursor=checkpoint['campaign_samples'];assert cursor==a.step*128
    iterator=iter(WindowSampler(tr,start=cursor,seed=cfg['seed'],batch_size=128))
    next16=[next(iterator) for _ in range(16)]
    expectations=gradient_expectations(module.model,cfg)
    frozen=[k for k,v in expectations.items() if v=='frozen']+cfg['audit_expectations'].get('initial_unchanged_prefixes',[])
    frozen_proof=compare_tensors(initial,portable,frozen) if frozen else dict(equal=True,compared_tensors=0)
    assert frozen_proof['equal']
    model=module.model
    del actual,params,opt,sched,configured,module,checkpoint,portable,initial,tr
    gc.collect()
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model=model.cuda().eval();before=state_hash(model)
    ds=dataset(cfg,'validation');task=next(r['task'] for r in read(root/'protocol.json')['runs'] if r['id']==a.id)
    cohort=read(root/'cohorts'/f'{task}.json')
    batch=next(iter(DataLoader(Subset(ds,[r['index'] for r in cohort[:2]]),batch_size=2,num_workers=0)))
    assert batch['episode'].tolist()==[r['episode'] for r in cohort[:2]]
    batch=move_batch(batch,'cuda');batch.pop('state',None)
    torch.manual_seed(1042);loss,metrics=model.loss(batch,step=a.step,ramp_steps=cfg.get('ramp_steps',0))
    assert finite(loss) and finite(metrics);loss.backward()
    gradients=gradient_report(model,expectations);assert all(r['passed'] for r in gradients.values()),gradients
    if cfg['model_class'] in ['local_readout','spatial_native']:
        groups={f'predictor.transformer.layers.{i}':'active' for i in range(6)}
        if cfg['model_class']=='local_readout':
            groups.update({'predictor.readout.0':'active','predictor.readout.2':'active'})
        else:
            groups.update({'predictor.transformer.input_proj':'active','predictor.transformer.output_proj':'active',
                          'predictor.transformer.cond_proj':'active'})
        layers=gradient_report(model,groups);assert all(r['passed'] for r in layers.values()),layers
    else:layers={}
    assert all(finite(p.grad) for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    diag=None;observed=[];actions=[]
    if a.diagnostics:
        with torch.no_grad():
            for batch in DataLoader(Subset(ds,[r['index'] for r in cohort]),batch_size=4,num_workers=0):
                batch=move_batch(batch,'cuda');pixels,_,_=model.prepare(batch)
                observed.append(model.encode(dict(pixels=pixels))['emb'].float());actions.append(batch['action'].float())
            observed,actions=torch.cat(observed),torch.cat(actions)
            diag,states=evaluate_latents(model,observed,actions,batch_size=8)
            np.savez_compressed(out/'state-audit.npz',observed=observed.cpu().numpy(),actions=actions.cpu().numpy(),
                                **{k:v.cpu().numpy() for k,v in states.items()})
            if model.renderer is not None:
                # One prespecified matched-noise renderer check. This is not image
                # generation or likelihood; no inference from zero sensitivity alone.
                from vforge.models.as_jepa.pixels import pack
                if hasattr(model,'render_query_batch'):
                    render_batch,address=model.render_query_batch(batch,a.step)
                    query_context=model.renderer.queried(address.repeat_interleave(batch['pixels'].shape[1]-1,0))
                else:render_batch,query_context=batch,nullcontext()
                pixels,render,anchor=model.prepare(render_batch)
                real=model.encode(dict(pixels=pixels))['emb'];acts=model.action_encoder(batch['action'])
                def forecast(acts):
                    zs=[real[:,0]]
                    for t in range(real.shape[1]-1):zs.append(model.predict(torch.stack(zs,1),acts[:,:t+1])[:,-1])
                    return torch.stack(zs[1:],1)
                good=forecast(acts);wrong=forecast(model.action_encoder(batch['action'].roll(1,1)))
                y=pack(render[:,1:].flatten(0,1));anc=pack(anchor).repeat_interleave(real.shape[1]-1,0)
                torch.manual_seed(1042);eps=torch.randn_like(y);noised=.05*y+.95*eps;t=y.new_full((len(y),1),.95)
                energies={}
                with query_context:
                    for name,z in [('forecast',good),('wrong_actions',wrong),('other_state',good.roll(1,0)),('zero',torch.zeros_like(good))]:
                        with model.renderer.anchored(anc):v=model.renderer(noised,t,condition=z.reshape(-1,model.state_slots,model.state_width))
                        energies[name]=float((y-eps-v).square().mean())
                diag['renderer_t095_last4_windows']=energies
        assert finite(diag)
        write(out/'state-audit.json',dict(step=a.step,cohort=cohort,results=diag,
              scope='Frozen real-input prediction/state/action diagnostic;no optimizer,physical labels or success claim.'))
    after=state_hash(model);assert before==after
    write(out/'audit.json',dict(passed=True,step=a.step,checkpoint=check,
          native_cpu_model_optimizer_scheduler_restore=True,fresh_initial_exact=True,
          next16_sampler_indices=next16,saved_rng_fields_checked=True,uninterrupted_rng_replay_claim=False,
          frozen_tensors=frozen_proof,gradients=gradients,layers=layers,loss=float(loss.detach()),
          state_before=before,state_after=after,config_sha256=sha(run/'config.json'),
          full_checkpoint_sha256=sha(run/'last.ckpt'),portable_sha256=sha(run/f'model-step{a.step:07d}/weights.pt'),
          optimizer_steps_added_by_audit=0,diagnostics=bool(a.diagnostics)))
    print('Audit passed',a.id,a.step,flush=True)


if __name__=='__main__':main()
