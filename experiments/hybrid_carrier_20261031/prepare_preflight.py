"""Prepare the reviewed model's eight two-step candidates; never activate."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

B=Path('/efs/prescottsun/SPT-JEPA')
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
MODEL_COMMIT='f2ae93d4f0da2fb32021bfade7426b11ea06602b'
PREFIX='experiments/hybrid_carrier_20261031/'
STAT_KEYS=['hybrid_projection','hybrid_fixed_mean','hybrid_fixed_scale','hybrid_query_mean','hybrid_query_scale']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')


def allocations():
    slots=[('dev2',0),('dev2',1),('dev2',2),('dev13',0),('dev13',1),('dev13',2),('dev15',0),('dev15',1)]
    return [dict(task=t,training_seed=s,arm=a,host=h,gpu=g,id=f'{t}-hybrid-{a}-seed{s}')
        for (t,s,a),(h,g) in zip(itertools.product(('tworoom','cube'),(1,2),('fixed','adaptive')),slots)]


def config(alloc,root,calibration_path):
    cal=read(calibration_path);cfg=copy.deepcopy(read(cal['config']))
    assert cal['source_commit']==MODEL_COMMIT and cal['measured_frames']==128 and cal['model_seed']==alloc['training_seed']
    assert sha(cal['config'])==cal['config_sha256'] and cfg['seed']==alloc['training_seed']
    assert cfg['batch_size']==128 and cfg['max_steps']==124910
    assert not cfg.get('initial_weights') and not cfg.get('initial_encoder_checkpoint')
    assert cfg.get('calibrate_state_std') is None and cfg.get('freeze_encoder_steps',0)==0
    assert cfg['model']['beta']==.25 and cfg['model']['rho']==.8 and cfg['model']['forecast_fraction']==.5
    assert cfg['model'].get('sigma0',.1)==.1 and cfg['model'].get('rate_prior','one-step')=='one-step'
    assert cfg['model'].get('flow_logit_shift',0.)==0. and not cfg['model'].get('flow_force_t_one',False)
    cfg.update(output=str(root/alloc['id']),model_class='hybrid_carrier',checkpoint_every=2,
        retain_steps=[2],record_initial_weights=True,
        audit_expectations=dict(gradients={'action_encoder':'active'},initial_unchanged_prefixes=STAT_KEYS))
    cfg['model'].update(state_slots=8,state_width=16,hybrid_query=alloc['arm'],**cal['model_kwargs'])
    cfg['hybrid_calibration']=dict(path=str(calibration_path),sha256=sha(calibration_path))
    return cfg


def plan_jobs(root,model_source,helper_source,configs):
    jobs=[]
    for a in allocations():
        name=a['id'];out=root/name;cfgpath=out/'config.json'
        peer=f'{a["task"]}-hybrid-{"adaptive" if a["arm"]=="fixed" else "fixed"}-seed{a["training_seed"]}'
        common={k:v for k,v in a.items() if k!='id'};common['source']=model_source
        jobs.append(dict(**common,id='train2-'+name,phase='train2',output=str(out),config=str(cfgpath),
            config_sha256=hashlib.sha256((json.dumps(configs[name],indent=2,allow_nan=False)+'\n').encode()).hexdigest(),
            command=[PY,'-B','-u',str(Path(helper_source['source'])/PREFIX/'train_preflight.py'),
                     '--config',str(cfgpath),'--stop-step','2']))
        command=[PY,'-B','-u',str(Path(helper_source['source'])/PREFIX/'preflight.py'),
            '--run',str(out),'--output',str(root/('fullstate-'+name)),'--expected-step','2','--gradient-cases','2',
            '--reference-initial',str(root/peer/'initial-model/weights.pt')]
        for prefix in ['encoder','pool','predictor','action_encoder','renderer','sigreg',*STAT_KEYS]:
            command+=['--reference-prefix',prefix]
        jobs.append(dict(**common,id='fullstate-'+name,phase='fullstate',output=str(root/('fullstate-'+name)),
            depends_on=['train2-'+name,'train2-'+peer],command=command))
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--helper-source-record',required=True)
    p.add_argument('--root',default=str(B/'runs/hybrid_carrier_preflight_20261031'))
    args=p.parse_args();root=Path(args.root);assert not root.exists()
    calroot=B/'runs/hybrid_carrier_calibration_20261031'
    closed=read(calroot/'closed.json');assert closed['hold'] and closed['owned_gpus_empty'] and (calroot/'HOLD').exists()
    assert len(closed['exits'])==4 and all(x['returncode']==0 for x in closed['exits'])
    model=read(calroot/'source-record.json');assert model['source_commit']==MODEL_COMMIT
    helper=read(args.helper_source_record)
    manifests=[]
    for src in [model,helper]:
        manifest=read(src['manifest']);assert manifest['source']==src['source'] and manifest['source_commit']==src['source_commit']
        for n,digest in manifest['files'].items():assert sha(Path(src['source'])/n)==digest,n
        manifests.append(manifest)
    assert sha(__file__)==manifests[1]['files'][PREFIX+'prepare_preflight.py']
    proof=read(Path(helper['source'])/PREFIX/'cpu/preflight-contracts.json')
    assert proof['passed'] and proof['tests']==5
    for name,digest in proof['source_sha256'].items():assert manifests[1]['files'][PREFIX+name]==digest
    for name in ['as_jepa_lewm/hybrid_carrier.py','as_jepa_lewm/factory.py','as_jepa_lewm/model.py',
                 'as_jepa_lewm/robust.py','as_jepa_lewm/train.py']:
        assert manifests[0]['files'][name]==manifests[1]['files'][name],name
    configs={a['id']:config(a,root,calroot/f'calibrate-{a["task"]}-seed{a["training_seed"]}'/'calibration.json') for a in allocations()}
    jobs=plan_jobs(root,model,helper,configs)
    assert len(jobs)==16 and len({j['id'] for j in jobs})==16
    ids={j['id'] for j in jobs}
    assert all(set(j.get('depends_on',[]))<=ids for j in jobs)
    for a in allocations():
        c=copy.deepcopy(configs[a['id']]);c.pop('output');c['model'].pop('hybrid_query')
        peer=f'{a["task"]}-hybrid-{"adaptive" if a["arm"]=="fixed" else "fixed"}-seed{a["training_seed"]}'
        other=copy.deepcopy(configs[peer]);other.pop('output');other['model'].pop('hybrid_query')
        assert c==other
    root.mkdir()
    for name,c in configs.items():write(root/name/'config.json',c)
    for j in jobs:
        if j['phase']=='train2':assert sha(j['config'])==j['config_sha256']
    write(root/'model-source-record.json',model);write(root/'helper-source-record.json',helper)
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,
        environment=read(calroot/'plan.json')['environment'],jobs=jobs))
    write(root/'protocol.json',dict(model_source=model,helper_source=helper,allocations=allocations(),
        runs=8,real_native_updates_per_run=2,endpoints=16,batch_size=128,scheduler_total_steps=124910,
        candidate_only=True,all_555_initial_tensors_paired=True,
        fixed_Q='Nominal requires_grad true, but9 exact initial tensors/gradNone/no saved native AdamW state required.',
        trace='Only2 steps: F/Q means and centered/total RMS; native clipping once with raw before/after group norms.',
        additional_Q_assay='Initial Q on already-computed8 tokens/repeated1024; high/no-grad/grad; no extra encoding or fitting.',
        stopping_rule='No600 or any further training without root actual-preflight review.'))
    (root/'HOLD').write_text('Candidate-only technical2step plan; root review and actual-free checks required.\n')
    print(json.dumps(dict(root=str(root),runs=8,endpoints=16,active_plan=False)))


if __name__=='__main__':main()
