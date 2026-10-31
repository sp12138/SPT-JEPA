"""Prepare8 authorized fresh600 chains; unchanged immutable1018 model source."""
import argparse
import copy
import json
from pathlib import Path

from prepare_preflight import B,PY,PREFIX,OLD,read,sha,write,allocations,configure

PRE=B/'runs/patch_token_preflight_20261031'
MODEL_COMMIT='1018ebbe9beb3e5f9b7c91018f03ff8a4f17884d'
def fresh_config(old,a,output):
    cfg=configure(old,a,output)
    cfg.update(checkpoint_every=200,retain_steps=[200,400,600])
    return cfg
def make_jobs(root,model,helper,configs):
    jobs=[]
    for a in allocations():
        name=a['id'];out=root/name;common={k:v for k,v in a.items() if k!='id'};common['source']=model
        peer=f'{a["task"]}-patch-{"orthogonal" if a["arm"]=="identity" else "identity"}-seed{a["training_seed"]}'
        old=OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}';cfg=configs[name]
        jobs.append(dict(**common,id='train600-'+name,phase='train600',output=str(out),config=str(out/'config.json'),
            command=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(out/'config.json'),'--stop-step','600']))
        previous='fullstate-'+name;dest=root/previous
        jobs.append(dict(**common,id=previous,phase='fullstate',output=str(dest),depends_on=['train600-'+name,'train600-'+peer],
            command=[PY,'-B','-u',str(Path(helper['source'])/PREFIX/'preflight600.py'),'--run',str(out),
                '--output',str(dest),'--reference-initial',str(old/'initial-model/weights.pt'),
                '--paired-initial',str(root/peer/'initial-model/weights.pt'),'--expected-step','600']))
        for phase,command in [
            ('health',[PY,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(out),'--output',str(root/('health-'+name))]),
            ('transition',[PY,'-B','-u',str(Path(model['source'])/'experiments/ideas_validate_20261031/audit.py'),
                 '--run',str(out),'--output',str(root/('transition-'+name)),'--cases','64','--manifest',
                 str(Path(model['source'])/f'experiments/ideas_validate_20261031/audit-windows-{a["task"]}.json')]),
            ('planning',[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',
                 str(Path(cfg['data'])/'manifests/validation-strict.json'),'--checkpoint',str(out/'model-step0000600'),
                 '--data',cfg['data'],'--output',str(root/('planning-'+name))])]:
            ident=phase+'-'+name;extra={}
            if phase=='planning':extra=dict(environment=dict(MUJOCO_EGL_DEVICE_ID=str(a['gpu'])),
                manifest_sha256=sha(Path(cfg['data'])/'manifests/validation-strict.json'))
            jobs.append(dict(**common,id=ident,phase=phase,output=str(root/ident),depends_on=[previous],command=command,**extra));previous=ident
    return jobs
def main():
    p=argparse.ArgumentParser();p.add_argument('--helper-source-record',required=True)
    p.add_argument('--root',default=str(B/'runs/patch_token600_20261031'));args=p.parse_args()
    root=Path(args.root);assert not root.exists()
    closed=read(PRE/'closed.json');assert closed['all16_actual_exit0'] and closed['owned_gpus_empty']
    assert (PRE/'HOLD').exists() and (PRE/'HOLD_DISPATCH').exists()
    preplan=read(PRE/'plan.json');assert len(preplan['jobs'])==16
    for job in preplan['jobs']:assert read(PRE/'exits'/(job['id']+'.json'))['returncode']==0
    summary_path=B/'reports/patch-token-preflight-20261031/summary.json';summary=read(summary_path)
    assert summary['passed'] and summary['completed']==16 and summary['all_8_native_optimizers_restored']
    model=read(PRE/'source-record.json');assert model['source_commit']==MODEL_COMMIT
    helper=read(args.helper_source_record);manifests=[]
    for source in [model,helper]:
        manifest=read(source['manifest']);assert manifest['source_commit']==source['source_commit']
        for name,digest in manifest['files'].items():assert sha(Path(source['source'])/name)==digest,name
        manifests.append(manifest)
    assert sha(__file__)==manifests[1]['files'][PREFIX+'prepare600.py']
    for name in ['as_jepa_lewm/patch_carrier.py','as_jepa_lewm/factory.py','as_jepa_lewm/model.py','as_jepa_lewm/robust.py',
                 'as_jepa_lewm/spatial_carrier.py','as_jepa_lewm/train.py','experiments/ideas_validate_20261031/preflight.py']:
        assert manifests[0]['files'][name]==manifests[1]['files'][name],name
    proof=read(Path(helper['source'])/PREFIX/'cpu/prepare600.json');assert proof['passed'] and proof['endpoints']==40
    for name,digest in proof['source_sha256'].items():assert manifests[1]['files'][PREFIX+name]==digest,name
    configs={};records=[]
    for a in allocations():
        pre=PRE/a['id'];old=OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}'
        cfg=fresh_config(read(old/'config.json'),a,root/a['id'])
        cal=cfg['spatial_calibration'];assert sha(cal['path'])==cal['sha256']
        reference=copy.deepcopy(read(pre/'config.json'));reference.update(output=cfg['output'],checkpoint_every=200,retain_steps=[200,400,600])
        assert reference==cfg,'Fresh600 differs from reviewed2step beyond operational cap/output'
        assert not cfg.get('is_resuming',False) and not cfg.get('initial_weights')
        configs[a['id']]=cfg;records.append(dict(**a,calibration=cal,reviewed_preflight=str(pre),
             primary_manifest_sha256=sha(Path(cfg['data'])/'manifests/validation-strict.json')))
    jobs=make_jobs(root,model,helper,configs);assert len(jobs)==40
    ids={j['id'] for j in jobs};assert len(ids)==40 and all(set(j.get('depends_on',[]))<=ids for j in jobs)
    for name,cfg in configs.items():write(root/name/'config.json',cfg)
    for job in jobs:
        if job['phase']=='train600':job['config_sha256']=sha(job['config'])
    env=copy.deepcopy(preplan['environment'])
    env.update(PYOPENGL_PLATFORM='egl',LD_LIBRARY_PATH=str(B/'environments/egl-lib/usr/lib/x86_64-linux-gnu')+':/usr/lib64',
        __EGL_VENDOR_LIBRARY_FILENAMES=str(B/'environments/egl-lib/10_nvidia.json'))
    write(root/'source-record.json',model);write(root/'helper-source-record.json',helper)
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=env,jobs=jobs))
    write(root/'protocol.json',dict(model_source=model,helper_source=helper,runs=records,steps=600,fresh_from_zero=True,
        preflight_summary_sha256=sha(summary_path),endpoints=40,native_batch=128,max_steps=124910,
        stages=['train600','strictfullstate','nativehealth16','existingtransition64','originalCEMprimary50'],
        controls='Two token arms same parameters/initialization, raw local1024/copy/zero coordinates. Historical flat models have different predictor capacity/compute.',
        not_in_scope=['resume2step','extra seeds','fresh cases','correlated controller','NGC','parameter grid','automatic2000'],
        stopping_rule='All600 stages stop; inspect task/copy/zero/teacher/state-action-use and per-layer gradients before any further budget. Preserve old holds/failures.'))
    (root/'HOLD').write_text('Prepared fresh600 bounded plan; activate only after frozen helper and actual GPU-free checks.\n')
    print(json.dumps(dict(root=str(root),endpoints=40,activated=False)))
if __name__=='__main__':main()
