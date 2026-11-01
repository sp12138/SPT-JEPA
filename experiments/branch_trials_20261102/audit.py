"""Actual native CPU restoration,GPU backward and deployed-interface diagnostics."""
import argparse
import gc
import inspect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from common import B,dataset,read,write,sha
from as_jepa_lewm.train import TrainModule,WindowSampler
from as_jepa_lewm.factory import gradient_expectations
from experiments.unified_three_20261101.common import same
from experiments.trajectory_search_20261101.common import state_hash
from experiments.ideas_validate_20261031.preflight import checkpoint_report,compare_tensors,finite,gradient_report
from experiments.ideas_validate_20261031.audit import move_batch
from vforge.ops import OpsConfig,initialize_ops


def main(root,ident,step,diagnostics):
    root=Path(root);protocol=read(root/'protocol.json');row=next(r for r in protocol['runs'] if r['id']==ident)
    run=Path(row['run']);cfg=read(run/'config.json');out=root/'audits'/f'{ident}-step{step}';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    checkpoint=torch.load(run/'full-archives'/f'step{step:07d}.ckpt',map_location='cpu',weights_only=False)
    portable=torch.load(run/f'model-step{step:07d}/weights.pt',map_location='cpu',weights_only=True)
    initial=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True)
    health=checkpoint_report(checkpoint,portable,cfg,step);assert all(health['checks'].values()),health
    torch.manual_seed(cfg['seed']);module=TrainModule(cfg)
    initial_encoder=Path(cfg['initial_encoder_checkpoint'])/'weights.pt';assert sha(initial_encoder)==cfg['frozen_encoder_sha256']
    declared=torch.load(initial_encoder,map_location='cpu',weights_only=True,mmap=True)
    module.model.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in declared.items() if k.startswith('encoder.')},strict=True)
    del declared
    calibration=None
    if cfg.get('calibrate_state_std'):
        stats=read(run/'initial_scale.json');assert stats['target_std']==cfg['calibrate_state_std']
        head=module.model.pool.head
        mean=torch.tensor(stats['mean_before'],dtype=head.bias.dtype).reshape(module.model.state_slots,module.model.state_width).mean(0)
        with torch.no_grad():head.weight.mul_(stats['scale']);head.bias.sub_(mean).mul_(stats['scale'])
        actual=initial['pool.head.bias'];expected=head.bias.detach()
        # One-time calibration was done onGPU. Its four-slot FP32 mean may
        # round differently onCPU;this is bounded initialization arithmetic.
        assert torch.equal(head.weight.detach(),initial['pool.head.weight'])
        assert torch.allclose(actual,expected,rtol=2e-6,atol=1e-7)
        calibration=dict(metadata_sha256=sha(run/'initial_scale.json'),weight_exact=True,
                         bias_max_error=float((actual-expected).abs().max()),bias_rtol=2e-6,bias_atol=1e-7)
    current=module.model.state_dict()
    names=[n for n in current if not(calibration and n=='pool.head.bias')]
    assert set(current)==set(initial) and all(torch.equal(current[n],initial[n]) for n in names),'Undeclared initialization mismatch'
    module.load_state_dict(checkpoint['state_dict'],strict=True)
    configured=module.configure_optimizers();optimizer=configured['optimizer'];scheduler=configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer_states'][0]);scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    assert same(module.state_dict(),checkpoint['state_dict']) and same(optimizer.state_dict(),checkpoint['optimizer_states'][0])
    assert same(scheduler.state_dict(),checkpoint['lr_schedulers'][0])
    trainable=[p for p in module.model.parameters() if p.requires_grad];actual=[p for g in optimizer.param_groups for p in g['params']]
    assert len(actual)==len(trainable) and {id(p) for p in actual}=={id(p) for p in trainable} and all(p in optimizer.state for p in trainable)
    tr=dataset(cfg,'train');cursor=checkpoint['campaign_samples'];assert cursor==step*128
    sampler=iter(WindowSampler(tr,start=cursor,seed=cfg['seed'],batch_size=128));next16=[next(sampler) for _ in range(16)]
    expectations=gradient_expectations(module.model,cfg)
    frozen=[k for k,v in expectations.items() if v=='frozen']
    frozen+=cfg.get('audit_expectations',{}).get('initial_unchanged_prefixes',[])
    proof=compare_tensors(initial,portable,frozen) if frozen else dict(equal=True,compared_tensors=0)
    assert proof['equal']
    runtime=read(run/f'stage-runtime-source-{step:07d}.json')
    assert runtime['cwd']==protocol['source']['source'] and runtime['PYTHONPATH']==protocol['source']['pythonpath']
    assert Path(runtime['paths']['model']).resolve().is_relative_to(Path(protocol['source']['source']).resolve())
    for key,path in runtime['paths'].items():assert sha(path)==runtime['sha256'][key]
    model=module.model
    del configured,optimizer,scheduler,actual,trainable,module,checkpoint,portable,initial,current,tr;gc.collect()
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model=model.cuda().eval();before=state_hash(model)
    ds=dataset(cfg,'validation');cohort=read(root/'cohorts'/f'{row["task"]}.json')
    ids=[x['index'] for x in cohort]
    batch=next(iter(DataLoader(Subset(ds,ids[:2]),batch_size=2,num_workers=0)));batch=move_batch(batch,'cuda');batch.pop('state',None)
    torch.manual_seed(1042);loss,metrics=model.loss(batch,step=step,ramp_steps=0);assert finite(loss) and finite(metrics);loss.backward()
    gradients=gradient_report(model,expectations);assert all(r['passed'] for r in gradients.values()),gradients
    assert all(p.grad is None or finite(p.grad) for p in model.parameters())
    model.zero_grad(set_to_none=True)
    diagnostic=None
    if diagnostics:
        arrays={};statistics=[]
        with torch.no_grad():
            for batch in DataLoader(Subset(ds,ids),batch_size=4,num_workers=0):
                assert set(batch['episode'].tolist()).issubset({r['episode'] for r in cohort})
                batch=move_batch(batch,'cuda');batch.pop('state',None)
                pixels=model.normalize(batch['pixels']);observed=model.encode(dict(pixels=pixels))['emb'];actions=batch['action']
                def keep(name,value):arrays.setdefault(name,[]).append(value.detach().float().cpu().numpy())
                keep('observed',observed);keep('actions',actions)
                if cfg['model_class']=='goal_query':
                    reference=model.encode(dict(pixels=model.normalize(batch['reference_pixels'])))['emb']
                    act=model.action_encoder(actions);q=model.query_predictions(observed[:,:3],act[:,:7])
                    positive=model.query_logits(q,observed[:,3:]);negative=model.query_logits(q,reference)
                    wrong_raw=actions.clone();wrong_raw[:,2:7]=actions.flip(0)[:,2:7]
                    wrong_q=model.query_predictions(observed[:,:3],model.action_encoder(wrong_raw)[:,:7])
                    other_q=model.query_predictions(observed[:,:3].flip(0),act[:,:7])
                    for name,value in [('queries',q),('positive',positive),('reference',negative),
                                       ('wrong_action_score',model.query_logits(wrong_q,observed[:,3:])),
                                       ('other_context_score',model.query_logits(other_q,observed[:,3:]))]:keep(name,value)
                else:
                    info=dict(pixels=pixels[:,:3,None].transpose(1,2),emb=observed[:,:3,None].transpose(1,2),
                              action_history=actions[:,:2,None].transpose(1,2))
                    predicted=model.rollout(info,actions[:,None,2:7])['predicted_emb'][:,0,3:]
                    keep('forecast',predicted);keep('target',observed[:,3:]);keep('copy',observed[:,2:3].expand_as(predicted))
                    if cfg['model_class']=='decoder_free':
                        all_observed=model.encode(dict(pixels=model.frames(batch)))['emb']
                        mu=all_observed[:,:8].reshape(len(pixels),8,model.state_slots,model.state_width)
                        anchor=all_observed[:,8].reshape(len(pixels),model.state_slots,model.state_width)
                        statistics.append({k:float(v) for k,v in model.information_diagnostics(mu[:,1:],anchor).items()})
        joined={k:np.concatenate(v) for k,v in arrays.items()};np.savez_compressed(out/'diagnostics.npz',**joined)
        diagnostic=dict(kind=cfg['model_class'],cohort=cohort,cohort_frames=64*8,precision='FP32 eval',
                        scope='Cohort size,not an exact forward count;reference/anchor/health encodes are additional.',
                        observer_std=float(joined['observed'].reshape(-1,joined['observed'].shape[-1]).std(0).mean()))
        if cfg['model_class']=='goal_query':
            diagnostic.update(positive_logit=float(joined['positive'].mean()),reference_logit=float(joined['reference'].mean()),
                              action_score_gap=float((joined['positive']-joined['wrong_action_score']).mean()),
                              context_score_gap=float((joined['positive']-joined['other_context_score']).mean()))
        else:
            errors=np.square(joined['forecast']-joined['target']).mean((0,2));copy=np.square(joined['copy']-joined['target']).mean((0,2))
            diagnostic.update(forecast_mse=errors.tolist(),copy_mse=copy.tolist(),forecast_copy_ratio=(errors/np.maximum(copy,1e-30)).tolist(),information_batch4=statistics)
        write(out/'diagnostics.json',diagnostic)
    after=state_hash(model);assert before==after
    report=dict(passed=True,step=step,checkpoint=health,native_cpu_model_optimizer_scheduler_restore=True,
                fresh_initial_noncalibrated_tensors_exact=len(names),calibration=calibration,
                declared_random_encoder_sha256=cfg['frozen_encoder_sha256'],frozen_tensors=proof,
                gradients=gradients,next16_sampler_indices=next16,saved_rng_fields_checked=True,
                uninterrupted_rng_replay_claim=False,state_before=before,state_after=after,
                optimizer_steps_added_by_audit=0,
                actual_runtime_source=runtime,config_sha256=sha(run/'config.json'),
                full_checkpoint_sha256=sha(run/'full-archives'/f'step{step:07d}.ckpt'),
                portable_sha256=sha(run/f'model-step{step:07d}/weights.pt'),loss=float(loss.detach()),diagnostics=diagnostic)
    write(out/'audit.json',report);print(dict(passed=True,id=ident,step=step))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True);p.add_argument('--step',required=True,type=int);p.add_argument('--diagnostics',action='store_true');a=p.parse_args();main(a.root,a.id,a.step,a.diagnostics)
