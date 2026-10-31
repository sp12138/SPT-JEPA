"""Four reviewed raw-public charts: native600->1200 archive->2000 archive."""
import argparse,copy,itertools,json,shutil,time
from pathlib import Path
from experiments.local_carrier_20261031 import prepare2000 as core
from experiments.local_carrier_20261031.prepare6000 import fresh_references

B,PY=core.B,core.PYTHON
read,write,sha,semantic=core.read,core.write,core.digest,core.semantic
ROOT=B/'runs/local_raw_chart2000_20261031'
PARENT=B/'runs/local_raw_chart_repaired600_20261031'
PRIOR=B/'runs/raw_chart_public_termination_gate600_retry_20261031'
IDENTITY=B/'runs/local_isotropic2000_20261031'
SELF='experiments/local_carrier_20261031/raw_chart/prepare2000.py'
MODEL_COMMIT='6e1693345786f2c190d0191685e4b485d24b4ab8'
PREFIXES=['encoder','pool','predictor','action_encoder','renderer','spatial_projection','spatial_mean','spatial_scale','carrier_chart_module','sigreg']
SIGREG=['sigreg.t','sigreg.phi','sigreg.weights']


def allocations():
    for gpu,(task,seed) in enumerate(itertools.product(('tworoom','cube'),(1,2))):
        yield dict(id=f'{task}-raw-chart-bounded-seed{seed}',host='dev13',gpu=gpu,task=task,training_seed=seed,arm='local-raw-bounded',family='spatial_carrier')


def sources(helper,model):
    assert model['source_commit']==MODEL_COMMIT
    for s in [model,helper]:
        m=read(s['manifest']);assert m['source']==s['source'] and m['source_commit']==s['source_commit']
        for name,digest in m['files'].items():assert sha(Path(s['source'])/name)==digest,name
    assert sha(__file__)==read(helper['manifest'])['files'][SELF]
    assert Path(core.__file__).is_relative_to(Path(model['source']))
    assert helper['runtime_source']==model['runtime_source'] and helper['pythonpath']==model['pythonpath']
    return dict(model=model,helper=helper,model_manifest_sha256=sha(model['manifest']),helper_manifest_sha256=sha(helper['manifest']))


def config(old,out):
    assert old['model']['spatial_state_interface']=='raw' and old['model']['spatial_chart']=='bounded'
    assert old['model']['spatial_mode']=='local' and old['model'].get('spatial_readout_mode','none')=='none'
    assert old['model'].get('spatial_basis','identity')=='identity'
    return core.continuation_config(old,out)


def audit(run,step,parent,initial,model,portable_override=None):
    r=core.audit_state(run,step,parent,initial,model,portable_override)
    import torch
    from as_jepa_lewm.carrier import ResidualCarrierChart
    p=torch.load(r['portable_path'],map_location='cpu',weights_only=True)
    z=torch.load(initial,map_location='cpu',weights_only=True)
    assert len(p)==len(z)==557 and r['frozen_tensors']['compared_tensors']==210
    assert all(torch.equal(p[k],z[k]) for k in SIGREG)
    chart=ResidualCarrierChart(16,64,'bounded',.5)
    w={k.removeprefix('carrier_chart_module.'):v for k,v in p.items() if k.startswith('carrier_chart_module.')}
    assert len(w)==4 and sum(v.numel() for v in w.values())==2128
    chart.load_state_dict(w,strict=True);cert=chart.norm_certificate()
    r['checks']['three_sigreg_buffers_unchanged']=True
    r['checks']['raw_shared_chart_bound']=cert['residual_lipschitz_upper']<=.500001
    r.update(chart=dict(parameters=2128,tensors=4,certificate=cert),sigreg_buffers=SIGREG,passed=all(r['checks'].values()))
    assert r['passed'];return r


def archive(args):
    helper,model=read(args.helper_record),read(args.model_record);provenance=sources(helper,model)
    run=Path(args.run);stage=read(run/'stage-complete.json');assert args.step in [1200,2000]
    assert stage['stage_complete'] and stage['stop_step']==args.step and stage['initial_step']==(600 if args.step==1200 else 1200)
    r=audit(run,args.step,args.parent,args.initial,model)
    target=run/f'checkpoint-step{args.step:07d}.ckpt';assert not target.exists()
    shutil.copy2(run/'last.ckpt',target);assert sha(target)==r['checkpoint_sha256']
    r.update(archived_checkpoint=str(target),native_stage_complete=stage,source_provenance=provenance)
    out=Path(args.output);assert not (out/'fullstate.json').exists();write(out/'fullstate.json',r)
    print(json.dumps(dict(step=args.step,passed=True,archive=str(target))),flush=True)


def jobs_for(rec,root,helper,model):
    root=Path(root);name=rec['id'];run=root/name;cfg=run/'config.json';jobs=[];deps=[]
    common={k:rec[k] for k in ['host','gpu','task','training_seed','arm','family']};common['source']=model
    def add(phase,out,command,**extra):
        nonlocal deps
        ident=phase+'-'+name;jobs.append(dict(**common,id=ident,phase=phase,output=str(out),depends_on=deps,command=command,**extra));deps=[ident]
    for step in [1200,2000]:
        add('learn'+str(step),run,[PY,'-B','-u','-v','-m','as_jepa_tools.train_stage','--config',str(cfg),'--stop-step',str(step),'--resume'],config=str(cfg),config_sha256=sha(cfg))
        out=root/('archive'+str(step)+'-'+name)
        add('archive'+str(step),out,[PY,'-B','-u',str(Path(helper['source'])/SELF),'archive','--helper-record',str(root/'helper-source-record.json'),
            '--model-record',str(root/'source-record.json'),'--run',str(run),'--parent',rec['parent'],'--initial',rec['initial'],'--step',str(step),'--output',str(out)])
    out=root/('fullstate2000-'+name)
    cmd=[PY,'-B','-u',str(Path(model['source'])/'experiments/ideas_validate_20261031/preflight.py'),'--run',str(run),'--output',str(out),
        '--expected-step','2000','--reference-initial',rec['original_initial']]
    for prefix in PREFIXES:cmd+=['--reference-prefix',prefix]
    add('fullstate2000',out,cmd)
    out=root/('health-'+name);add('health',out,[PY,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(run),'--output',str(out)])
    out=root/('transition-'+name);add('transition',out,[PY,'-B','-u',str(Path(model['source'])/'experiments/ideas_validate_20261031/audit.py'),
        '--run',str(run),'--output',str(out),'--cases','64','--manifest',str(Path(model['source'])/f'experiments/ideas_validate_20261031/audit-windows-{rec["task"]}.json')])
    for phase,key in [('planning','primary'),('fresh-planning','fresh')]:
        out=root/(phase+'-'+name);manifest=rec[key+'_manifest']
        add(phase,out,[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',manifest['path'],
            '--checkpoint',str(run/'model-step0002000'),'--data',rec['config']['data'],'--output',str(out)],
            manifest_sha256=manifest['sha256'],environment=dict(MUJOCO_EGL_DEVICE_ID=str(rec['gpu'])))
    assert len(jobs)==9;return jobs


def prepare(args):
    import torch
    torch.set_num_threads(2);root=Path(args.root);assert not root.exists()
    helper,model=read(args.helper_record),read(PARENT/'source-record.json');provenance=sources(helper,model)
    review=read(B/'reports/local-raw-chart-repaired600-20261031/summary.json')
    assert review['new_endpoints']==17 and review['prior_successful_endpoints']==3 and review['total_evidence_endpoints']==20
    assert sha(review['full_results']['path'])==review['full_results']['sha256']
    proto=read(PARENT/'protocol.json');plan=read(PARENT/'plan.json')
    assert (PARENT/'HOLD_DISPATCH').exists() and (PRIOR/'HOLD_DISPATCH').exists()
    for j in plan['jobs']:assert read(PARENT/'exits'/(j['id']+'.json'))['returncode']==0
    for j in proto['prior_successful_endpoints']:assert sha(j['path'])==j['sha256'] and read(j['path'])['returncode']==0
    fresh,evidence=fresh_references(B/'runs/followup_fresh2000_20261031',IDENTITY)
    records=[]
    for rec in allocations():
        name=rec['id'];old=(PRIOR if rec['task']=='tworoom' and rec['training_seed']==1 else PARENT)/name
        cfg=read(old/'config.json');initial=old/'initial-model/weights.pt'
        original=B/'runs/local_raw_chart600_20261031'/name/'initial-model/weights.pt'
        refname=f'{rec["task"]}-spatial-local-seed{rec["training_seed"]}'
        identity=B/'runs/local_isotropic600_20261031'/refname/'initial-model/weights.pt'
        values=[torch.load(p,map_location='cpu',weights_only=True) for p in [initial,original,identity]]
        assert len(values[0])==len(values[1])==557 and len(values[2])==553
        assert all(torch.equal(v,values[0][k]) for x in values[1:] for k,v in x.items());del values
        pre=audit(old,600,old,initial,model)
        proof=next(x for x in review['checks'] if x['task']==rec['task'] and x['seed']==rec['training_seed'])
        assert pre['checkpoint_sha256']==proof['fullstate_sha256']
        comparable=copy.deepcopy(cfg);del comparable['model']['spatial_chart'];del comparable['model']['spatial_state_interface']
        assert semantic(comparable)==semantic(read(IDENTITY/refname/'config.json'))
        primary=Path(cfg['data'])/'manifests/validation-strict.json';p=read(IDENTITY/('planning-'+refname)/'metrics.json')
        prior=read(PARENT/('planning-'+name)/'metrics.json')
        for key in ['manifest_sha256','pairs','solver','plan','protocol','split','normalization']:assert p[key]==prior[key]
        assert p['manifest_sha256']==sha(primary) and p['checkpoint_audit']['strict_equal'] and len(p['pairs'])==50
        cal=cfg['spatial_calibration'];assert sha(cal['path'])==cal['sha256']
        records.append(dict(**rec,parent=str(old),initial=str(initial),original_initial=str(original),identity_initial=str(identity),
            config=config(cfg,root/name),pre=pre,primary_manifest=dict(path=str(primary),sha256=sha(primary)),fresh_manifest=fresh[rec['task']],
            identity2000_primary=str(IDENTITY/('planning-'+refname)/'metrics.json'),identity2000_fresh=evidence[refname],
            old_chart2000=str(B/'runs/local_spatial_chart2000_20261031'/f'{rec["task"]}-spatial-chart-bounded-seed{rec["training_seed"]}')))
        print(json.dumps(dict(pre600_passed=name,parent=str(old),initial557_identity553=True)),flush=True)
    root.mkdir();write(root/'source-record.json',model);write(root/'helper-source-record.json',helper);write(root/'source-provenance.json',provenance)
    jobs=[];copies=[]
    for rec in records:
        out=root/rec['id'];out.mkdir();write(out/'config.json',rec['config']);old=Path(rec['parent'])
        shutil.copy2(old/'last.ckpt',out/'last.ckpt');shutil.copytree(old/'initial-model',out/'initial-model')
        assert sha(out/'last.ckpt')==rec['pre']['checkpoint_sha256']
        post=audit(out,600,old,out/'initial-model/weights.pt',model,old/'model-step0000600/weights.pt')
        assert post['checkpoint_sha256']==rec['pre']['checkpoint_sha256'] and post['sampler']==rec['pre']['sampler']
        assert post['rng_component_sha256']==rec['pre']['rng_component_sha256']
        write(out/'pre-resume-fullstate.json',rec['pre']);write(out/'post-copy-fullstate.json',post)
        copies.append(dict(id=rec['id'],parent=rec['parent'],complete600_byte_equal=True,pre_post_native_restore=True,checkpoint_sha256=post['checkpoint_sha256']))
        jobs.extend(jobs_for(rec,root,helper,model));print(json.dumps(dict(postcopy600_passed=rec['id'])),flush=True)
    assert len(jobs)==36 and len({j['id'] for j in jobs})==36 and all(j['source']==model for j in jobs)
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=plan['environment'],jobs=jobs))
    write(root/'resume-audit.json',copies)
    write(root/'protocol.json',dict(at=time.time(),runs=4,endpoints=36,model_source=model,helper_source=helper,allocations=[{k:v for k,v in r.items() if k not in ['pre','config']} for r in records],
        initial_step=600,intermediate_step=1200,stop_step=2000,scheduler_horizon=124910,warmup_steps=1249,archive_steps=[1200,2000],
        complete600_review=str(B/'reports/local-raw-chart-repaired600-20261031/summary.json'),source_600_accounting='3 prior real successes plus17 real successes; parent paths mapped per run.',
        initial_contract='All557 original raw and553 identity tensors exact, including3 sigreg buffers; sigreg remains inactive in proposed loss.',
        unchanged='Model6e/metadata-v2,raw public coordinates,2128 chart parameters,data,calibration,batch128,seed,sampler/RNG/optimizer/scheduler and native objectives/CEM unchanged. No extra labels/head/loss/cohort.',
        operations='New outputs; native600->1200->2000 boundaries; checkpoint cadence1200,retain1200/2000. Every job source remains6e; archive scripts explicitly frozen/self-verified. Python-v only records model import.',
        coordinates='Public/cached states already RAW; do not inverse again. Compare both original identity2000 and old chart2000 in common raw coordinates.',
        authorization='One reviewed maturity discrimination beyond1249 warmup. dev13g0-3 only after actual availability. Failed1200 archive blocks2000. No automatic budget>2000; no task/substantial-prediction gain at2000 means stop.',
        candidate_only=True))
    print(json.dumps(dict(candidate=str(root/'candidate-plan.json'),endpoints=36,launched=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','archive']);p.add_argument('--helper-record',required=True);p.add_argument('--model-record')
    p.add_argument('--root',default=str(ROOT));p.add_argument('--run');p.add_argument('--parent');p.add_argument('--initial');p.add_argument('--step',type=int,choices=[1200,2000]);p.add_argument('--output');a=p.parse_args()
    if a.mode=='archive':assert all([a.model_record,a.run,a.parent,a.initial,a.step,a.output]);archive(a)
    else:prepare(a)
