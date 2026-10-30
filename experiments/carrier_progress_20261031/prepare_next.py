"""Prepare the reviewed8 Carrier +4 TF continuations to6000; no launches."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import shutil
import time

from experiments.carrier_progress_20261031.prepare_learning import (
    B, PYTHON, activate_source, checkpoint_semantic, digest, read, rng_digest,
    semantic, verify_sources, write,
)

F_ROOT=B/'runs/carrier_learn2000_20261031'
TF_ROOT=B/'runs/hardtasks_learn2000_20261030'
D_ROOT=B/'runs/carrier_diagnostics_20261031'
ROOT=B/'runs/carrier_learn6000_20261031'
SELF='experiments/carrier_progress_20261031/prepare_next.py'
_CPU_OPS_READY=False


def allocations():
    rows=[]
    for seed in (1,2):
        for arm in ('identity','unconstrained','bounded'):
            rows.append(dict(task='tworoom',training_seed=seed,family='carrier',arm=arm))
    rows.extend(dict(task='cube',training_seed=s,family='carrier',arm='identity') for s in (1,2))
    rows.extend(dict(task=t,training_seed=s,family='temporal-forecast',arm='temporal-forecast')
                for t in ('tworoom','cube') for s in (1,2))
    for i,row in enumerate(rows):
        row.update(host='dev2' if i<8 else 'dev13',gpu=i if i<8 else i-8)
        middle='carrier-'+row['arm'] if row['family']=='carrier' else 'temporal-forecast'
        row['id']=f'{row["task"]}-{middle}-seed{row["training_seed"]}'
    return rows


def resume_sequence(dataset,cursor,seed,batch_size,count=32):
    from as_jepa_lewm.train import WindowSampler
    kwargs=dict(seed=seed,batch_size=batch_size)
    length=len(WindowSampler(dataset,**kwargs))
    assert length>0
    resumed=list(itertools.islice(iter(WindowSampler(dataset,start=cursor,**kwargs)),count))
    # Concatenate native loader epochs; start0 alone ends before the6000 cursor
    # on TwoRoom. No sampled image or action is read here.
    epochs=(cursor+count+length-1)//length
    reference=itertools.chain.from_iterable(
        iter(WindowSampler(dataset,start=i*length,**kwargs)) for i in range(epochs))
    expected=list(itertools.islice(reference,cursor,cursor+count))
    return dict(next_indices=resumed,native_loader_length=length,
                passed=len(resumed)==count and resumed==expected)


def audit_state(run,step,parent,initial_weights=None):
    """Existing checkpoint/sampler contracts plus native CPU load roundtrips."""
    import torch
    from experiments.ideas_validate_20261031.preflight import checkpoint_report,compare_tensors
    from as_jepa_lewm.train import TrainModule
    from as_jepa_lewm.factory import gradient_expectations
    from as_jepa_tools.benchmark_data import make_training_dataset
    from vforge.ops import OpsConfig,initialize_ops

    global _CPU_OPS_READY
    torch.set_num_threads(2)
    if not _CPU_OPS_READY:
        initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)
        _CPU_OPS_READY=True
    run,parent=Path(run),Path(parent);cfg=read(run/'config.json');oldcfg=read(parent/'config.json')
    assert semantic(cfg)==semantic(oldcfg)
    state=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    portable_path=run/f'model-step{step:07d}/weights.pt'
    portable=torch.load(portable_path,map_location='cpu',weights_only=True)
    report=checkpoint_report(state,portable,cfg,step)
    checks=report['checks']
    checks['complete_model_keys']=set(state['state_dict'])=={'model.'+k for k in portable}
    checks['saved_training_semantics']=checkpoint_semantic(state['hyper_parameters'])==semantic(cfg)
    checks['scheduler_step']=all(s['last_epoch']==step for s in state['lr_schedulers'])
    checks['optimizer_scheduler_lr_equal']=([g['lr'] for g in state['optimizer_states'][0]['param_groups']]
                                          ==state['lr_schedulers'][0]['_last_lr'])
    module=TrainModule(cfg)
    module.load_state_dict(state['state_dict'],strict=True)
    if cfg.get('model_class')!='carrier':
        # Audited bf1849 TF configs select exactly the unchanged arithmetic and
        # RNG path through all optional branches introduced before a499.
        assert cfg.get('robust_variant') and not cfg.get('model_class')
        expected=dict(rate_prior='one-step',posterior_noise_scale=1.,flow_force_t_one=False,
            response_query_mode='none',response_query_encoded_fraction=0.,
            renderer_anchor_mode='observed',pool_partition='full',rho=.8,forecast_fraction=.5)
        actual={k:getattr(module.model,k) for k in expected}
        checks['reviewed_tf_loss_path']=actual==expected
        report['reviewed_tf_loss_path']=actual
    optimizer_config=module.configure_optimizers()
    optimizer=optimizer_config['optimizer'];scheduler=optimizer_config['lr_scheduler']['scheduler']
    named={id(p):n for n,p in module.model.named_parameters()}
    current_names=[[named[id(p)] for p in g['params']] for g in optimizer.param_groups]
    trainable={n for n,p in module.model.named_parameters() if p.requires_grad}
    ordered=[n.removeprefix('model.') for n in state['state_dict']
             if n.removeprefix('model.') in trainable]
    serialized_names=[[n for n in ordered if not n.startswith('renderer.')],
                      [n for n in ordered if n.startswith('renderer.')]]
    checks['optimizer_parameter_order']=current_names==serialized_names
    optimizer.load_state_dict(state['optimizer_states'][0]);scheduler.load_state_dict(state['lr_schedulers'][0])
    def equal(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    checks['native_optimizer_roundtrip']=equal(state['optimizer_states'][0],optimizer.state_dict())
    checks['native_scheduler_roundtrip']=equal(state['lr_schedulers'][0],scheduler.state_dict())
    checks['native_model_strict_load']=True
    expectations=gradient_expectations(module.model,cfg)
    frozen=[name for name,value in expectations.items() if value=='frozen']
    if cfg.get('model_class')=='carrier':
        frozen+=['carrier_projection','carrier_mean','carrier_scale']
        assert initial_weights is not None
    if frozen:
        initial=torch.load(initial_weights,map_location='cpu',weights_only=True)
        assert all(any(n==p or n.startswith(p+'.') for n in portable) for p in frozen)
        report['frozen_tensors']=compare_tensors(initial,portable,frozen)
        checks['frozen_unchanged']=report['frozen_tensors']['equal']
    else:
        report['frozen_tensors']=dict(prefixes=[],compared_tensors=0,equal=True,
                                     scope='Native TF has no declared frozen encoder or chart.')
    sequence=resume_sequence(make_training_dataset(cfg),state['campaign_samples'],cfg['seed'],cfg['batch_size'])
    checks['sampler_cursor_sequence']=sequence['passed']
    report.update(run=str(run),parent=str(parent),config_sha256=digest(run/'config.json'),
        checkpoint_sha256=digest(run/'last.ckpt'),portable_sha256=digest(portable_path),
        initial_weights=str(initial_weights) if initial_weights else None,
        gradient_expectations=expectations,sampler=sequence,
        optimizer_parameter_tensors=[len(g) for g in current_names],
        rng_component_sha256={k:rng_digest(v) for k,v in state['campaign_rng'].items()},
        passed=all(checks.values()),scope='CPU-only saved-state and native model/optimizer/scheduler load audit. '
            'No update, GPU, or Trainer replacement. Does not assert bitwise resumed/uninterrupted GPU trajectories.')
    assert report['passed'],report
    return report


def archive(args):
    src=read(args.source_record);activate_source(src);run=Path(args.audit_run);out=Path(args.output)
    assert not (out/'fullstate.json').exists()
    report=audit_state(run,6000,args.parent_run,args.initial_weights)
    stage=read(run/'stage-complete.json');assert stage['initial_step']==2000 and stage['stop_step']==6000
    path=run/'checkpoint-step0006000.ckpt';assert not path.exists()
    shutil.copy2(run/'last.ckpt',path);assert digest(path)==report['checkpoint_sha256']
    report.update(archived_checkpoint=str(path),native_stage_complete=stage,source_commit=src['source_commit'])
    out.mkdir(parents=True,exist_ok=True);write(out/'fullstate.json',report)
    print(json.dumps(dict(passed=True,step=6000,archive=str(path))),flush=True)


def prepare(args):
    root=Path(args.root);assert not root.exists()
    src=read(args.source_record);oldsrc=read(F_ROOT/'source-record.json')
    assert oldsrc['source_commit']=='a499fc2b543214173ea3866b76df6636a859faa0'
    compatibility=verify_sources(src,oldsrc)
    assert digest(__file__)==digest(Path(src['source'])/SELF)==read(src['manifest'])['files'][SELF]
    activate_source(src)
    fp=read(F_ROOT/'plan.json');tp=read(TF_ROOT/'plan.json');dp=read(D_ROOT/'plan.json')
    for j in fp['jobs']:assert read(F_ROOT/'exits'/(j['id']+'.json'))['returncode']==0
    for j in dp['jobs']:assert read(D_ROOT/'exits'/(j['id']+'.json'))['returncode']==0
    tfjobs={(j['task'],j['training_seed']):j for j in tp['jobs']
            if j.get('phase')=='learn2000' and j.get('method')=='temporal-forecast'}
    fresh={}
    for task in ('tworoom','cube'):
        j=next(j for j in dp['jobs'] if j['task']==task and j['phase']=='fresh-validation-planning')
        command=j['command'];path=Path(command[command.index('--manifest')+1]);manifest=read(path)
        assert manifest['task']==task and manifest['campaign_split']=='validation' and len(manifest['pairs'])==50
        fresh[task]=dict(path=str(path),sha256=digest(path))
    records=[]
    for rec in allocations():
        carrier=rec['family']=='carrier'
        oldjob=tfjobs[rec['task'],rec['training_seed']] if not carrier else None
        parent=F_ROOT/rec['id'] if carrier else Path(oldjob['output'])
        cfg=read(parent/'config.json');stage=read(parent/'stage-complete.json')
        assert stage['stop_step']==2000 and cfg['seed']==rec['training_seed']
        assert cfg['max_steps']==124910 and cfg['batch_size']==128
        if oldjob:
            assert digest(parent/'config.json')==oldjob['config_sha256']
            for prefix in ('','health-','planning-'):
                assert read(TF_ROOT/'exits'/(prefix+oldjob['id']+'.json'))['returncode']==0
        initial=(Path(read(parent/'pre-resume-fullstate.json')['parent'])/'initial-model/weights.pt') if carrier else None
        pre=audit_state(parent,2000,parent,initial)
        newcfg=copy.deepcopy(cfg);newcfg.update(output=str(root/rec['id']),checkpoint_every=1000,retain_steps=[4000,6000])
        assert semantic(cfg)==semantic(newcfg)
        primary=Path(cfg['data'])/'manifests/validation-strict.json';fm=read(fresh[rec['task']]['path'])
        assert fm['metadata_card_sha256']==digest(Path(cfg['data'])/'dataset_card.json')
        assert fm['fresh_selection']['primary_manifest_sha256']==digest(primary)
        records.append(dict(**rec,parent=str(parent),initial_weights=str(initial) if initial else None,
            parent_source=oldsrc if carrier else oldjob['source'],config=newcfg,pre_audit=pre,
            primary_manifest=dict(path=str(primary),sha256=digest(primary))))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True,audits_passed=12,carrier=8,temporal_forecast=4,
                              allocations=allocations()),indent=2));return
    root.mkdir(parents=True);write(root/'source-record.json',src);write(root/'source-compatibility.json',compatibility)
    jobs=[];helper=str(Path(src['source'])/SELF)
    for rec in records:
        name=rec['id'];out=root/name;out.mkdir();config=out/'config.json';write(config,rec['config'])
        shutil.copy2(Path(rec['parent'])/'last.ckpt',out/'last.ckpt')
        assert digest(out/'last.ckpt')==rec['pre_audit']['checkpoint_sha256']
        write(out/'pre-resume-fullstate.json',dict(rec['pre_audit'],copy_byte_identical=True,
            preserved_parent_checkpoint=str(Path(rec['parent'])/'last.ckpt'),parent_source=rec['parent_source']))
        common={k:rec[k] for k in ('host','gpu','task','training_seed','family','arm')};common['source']=src
        ident='learn6000-'+name
        jobs.append(dict(**common,id=ident,output=str(out),config=str(config),config_sha256=digest(config),
            phase='learn6000',command=[PYTHON,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(config),'--stop-step','6000','--resume']))
        dest=root/('fullstate6000-'+name)
        command=[PYTHON,'-B','-u',helper,'--source-record',str(root/'source-record.json'),'--audit-run',str(out),
                 '--parent-run',rec['parent'],'--output',str(dest)]
        if rec['initial_weights']:command+=['--initial-weights',rec['initial_weights']]
        jobs.append(dict(**common,id=dest.name,output=str(dest),phase='fullstate6000',depends_on=[ident],command=command));previous=dest.name
        dest=root/('health-'+name)
        jobs.append(dict(**common,id=dest.name,output=str(dest),phase='health',depends_on=[previous],
            command=[PYTHON,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(out),'--output',str(dest)]));previous=dest.name
        if rec['family']=='carrier':
            dest=root/('transition-'+name)
            jobs.append(dict(**common,id=dest.name,output=str(dest),phase='transition',depends_on=[previous],
                command=[PYTHON,'-B','-u',str(Path(src['source'])/'experiments/ideas_validate_20261031/audit.py'),
                    '--run',str(out),'--output',str(dest),'--cases','64','--manifest',
                    str(Path(src['source'])/f'experiments/ideas_validate_20261031/audit-windows-{rec["task"]}.json')]));previous=dest.name
        for phase,manifest in [('planning',rec['primary_manifest']),('fresh-planning',fresh[rec['task']])]:
            dest=root/(phase+'-'+name)
            jobs.append(dict(**common,id=dest.name,output=str(dest),phase=phase,depends_on=[previous],
                environment={'MUJOCO_EGL_DEVICE_ID':str(rec['gpu'])},manifest_sha256=manifest['sha256'],
                command=[PYTHON,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',manifest['path'],
                         '--checkpoint',str(out/'model-step0006000'),'--data',rec['config']['data'],'--output',str(dest)]));previous=dest.name
    assert len(jobs)==68 and len({j['id'] for j in jobs})==68
    write(root/'plan.json',dict(root=str(root),deadline=None,environment=fp['environment'],jobs=jobs))
    write(root/'protocol.json',dict(created_at=time.time(),source_commit=src['source_commit'],root=str(root),
        model_training_source_equivalent_to=oldsrc['source_commit'],initial_step=2000,stop_step=6000,
        scheduler_horizon=124910,warmup_steps=1249,runs=12,jobs=68,allocations=allocations(),
        fresh_manifests=fresh,parents=[{k:r[k] for k in ('id','parent','parent_source','initial_weights','primary_manifest')} for r in records],
        tf_source_review=dict(path=str(Path(src['source'])/'experiments/carrier_progress_20261031/TF-SOURCE-COMPATIBILITY.md'),
            sha256=digest(Path(src['source'])/'experiments/carrier_progress_20261031/TF-SOURCE-COMPATIBILITY.md')),
        decision='Reviewed continuation: TwoRoom all three Carrier arms, Cube identity only, both seeds; '
            'four native TF2000 controls receive the same6000 total updates. No automatic later stage.',
        evidence='Fixed identity-coordinate causal errors fell strongly600→2000 in both tasks; late losses '
            'still decrease. Bound necessity and Cube planning superiority remain unproven.',
        state_preservation='Each held parent full2000 checkpoint remains unchanged and is byte-copied. '
            'Native periodic saves at3000/4000/5000/6000; portable4000/6000 retained, full6000 archived.',
        departures=['Fresh run directories and fixed new host/GPU assignment; no historical holds are cleared.',
            'TF source moves bf1849→a499-equivalent native code; identical config, strict full-model load, '
            'optimizer ordering/roundtrip and scheduler roundtrip are CPU-verified before preparation.',
            'Checkpoint cadence becomes1000; no extra1200-style intermediate resume boundary.',
            'No calibration refresh, loss change, optimizer reset, scheduler shortening or new expert labels.']))
    print(json.dumps(dict(root=str(root),runs=12,jobs=len(jobs),stop_step=6000)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-record',required=True)
    parser.add_argument('--root',default=str(ROOT));parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--audit-run');parser.add_argument('--parent-run');parser.add_argument('--initial-weights');parser.add_argument('--output')
    args=parser.parse_args()
    if args.audit_run:
        assert args.parent_run and args.output and not args.dry_run;archive(args)
    else:prepare(args)


if __name__=='__main__':main()
