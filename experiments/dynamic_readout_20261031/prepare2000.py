"""Bounded native600→1200→2000 continuation, immutable d6 model source."""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

B=Path('/efs/prescottsun/SPT-JEPA')
PARENT=B/'runs/dynamic_readout600_20261031'
FULL=B/'runs/dynamic_readout600_recovery_20261031'
EVAL=B/'runs/dynamic_readout600_health_recovery_20261031'
ROOT=B/'runs/dynamic_readout2000_20261031'
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
SELF='experiments/dynamic_readout_20261031/prepare2000.py'
HEALTH='experiments/dynamic_readout_20261031/health_compat.py'
AUDITOR_RECORD=B/'runs/local_isotropic2000_20261031/helper-source-record.json'
AUDITOR_FILES=['experiments/local_carrier_20261031/prepare2000.py',
               'experiments/carrier_progress_20261031/prepare_learning.py',
               'experiments/carrier_progress_20261031/prepare_next.py']
MODEL_COMMIT='d6d0a9f85127a8af3e65bd31b29c161c3584867c'
_AUDITOR=None


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def semantic(c):return {k:v for k,v in c.items()if k not in{'output','checkpoint_every','retain_steps','is_resuming'}}
def allocations():
    for gpu,(task,seed,arm)in enumerate(itertools.product(['tworoom','cube'],[1,2],['static','state'])):
        yield dict(id=f'{task}-readout-{arm}-seed{seed}',host='dev15',gpu=gpu,task=task,training_seed=seed,arm=arm,family='spatial_carrier')
def config(old,out):
    c=copy.deepcopy(old);c.update(output=str(out),checkpoint_every=1200,retain_steps=[1200,2000])
    assert semantic(c)==semantic(old)and c['batch_size']==128 and c['max_steps']==124910
    assert c['model_class']=='spatial_carrier'and c['model']['spatial_mode']=='local'
    assert c['model'].get('spatial_chart','none')=='none'and c['model'].get('spatial_basis','identity')=='identity'
    assert c['model']['spatial_readout_mode']in{'static','state'}
    return c


def checked_sources(helper,model):
    assert model['source_commit']==MODEL_COMMIT
    for s in[helper,model]:
        m=read(s['manifest']);assert m['source_commit']==s['source_commit']
        for rel,d in m['files'].items():assert sha(Path(s['source'])/rel)==d
    assert sha(__file__)==read(helper['manifest'])['files'][SELF]
    auditor=read(AUDITOR_RECORD);m=read(auditor['manifest'])
    assert auditor['source_commit']=='0934d3c445e6c754a6ab547d801890c34740b330'
    for rel in AUDITOR_FILES:assert sha(Path(auditor['source'])/rel)==m['files'][rel]
    return dict(model=model,helper=helper,auditor=auditor,
        reused_auditor_files={p:m['files'][p]for p in AUDITOR_FILES})


def audit(run,step,parent,initial,model,portable=None):
    global _AUDITOR
    if _AUDITOR is None:
        source=read(AUDITOR_RECORD)
        path=Path(source['source'])/AUDITOR_FILES[0]
        spec=importlib.util.spec_from_file_location('pinned_native_spatial_resume_audit',path)
        _AUDITOR=importlib.util.module_from_spec(spec);spec.loader.exec_module(_AUDITOR)
    # Reused native auditor explicitly activates d6 and asserts actual model imports.
    result=_AUDITOR.audit_state(run,step,parent,initial,model,portable)
    assert result['passed']and result['frozen_tensors']['compared_tensors']==210
    return result


def archive(args):
    helper,model=read(args.helper_record),read(args.model_record);sources=checked_sources(helper,model)
    assert args.step in[1200,2000]
    run=Path(args.run);stage=read(run/'stage-complete.json')
    assert stage['stage_complete']and stage['stop_step']==args.step
    assert stage['initial_step']==(600 if args.step==1200 else 1200)
    result=audit(run,args.step,args.parent,args.initial,model)
    target=run/f'checkpoint-step{args.step:07d}.ckpt';assert not target.exists()
    shutil.copy2(run/'last.ckpt',target);assert sha(target)==result['checkpoint_sha256']
    result.update(archived_checkpoint=str(target),native_stage_complete=stage,sources=sources)
    write(Path(args.output)/'fullstate.json',result)
    print(json.dumps(dict(step=args.step,passed=True,archive=str(target))),flush=True)


def existing_references(task,seed,data):
    ident=f'{task}-spatial-local-seed{seed}'
    primary=B/'runs/local_isotropic2000_20261031'/('planning-'+ident)/'metrics.json'
    fresh=B/'runs/followup_fresh2000_20261031'/('fresh-planning-'+ident)/'metrics.json'
    p,f=read(primary),read(fresh)
    for x in[p,f]:assert len(x['pairs'])==50 and x['checkpoint_audit']['strict_equal']
    primary_manifest=Path(data)/'manifests/validation-strict.json'
    fresh_manifest=Path(f['manifest'])
    assert sha(primary_manifest)==p['manifest_sha256']and sha(fresh_manifest)==f['manifest_sha256']
    assert p['normalization']==f['normalization']=='train episodes'
    return dict(primary_metrics=str(primary),primary_metrics_sha256=sha(primary),
        fresh_metrics=str(fresh),fresh_metrics_sha256=sha(fresh),
        primary_manifest=str(primary_manifest),primary_manifest_sha256=sha(primary_manifest),
        fresh_manifest=str(fresh_manifest),fresh_manifest_sha256=sha(fresh_manifest),
        scope='Existing identity2000 controls and exact existing50primary/50fresh;no newcase generation')


def jobs_for(rec,root,helper,model):
    root=Path(root);name=rec['id'];run=root/name;cfg=run/'config.json';deps=[];jobs=[]
    common={k:v for k,v in rec.items()if k in['host','gpu','task','training_seed','arm','family']};common['source']=model
    def add(phase,out,command,**extra):
        nonlocal deps
        ident=phase+'-'+name;jobs.append(dict(**common,id=ident,phase=phase,output=str(out),depends_on=deps,command=command,**extra));deps=[ident]
    for step in[1200,2000]:
        add('learn'+str(step),run,[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(cfg),'--stop-step',str(step),'--resume'],config=str(cfg),config_sha256=sha(cfg))
        out=root/('archive'+str(step)+'-'+name)
        add('archive'+str(step),out,[PY,'-B','-u',str(Path(helper['source'])/SELF),'archive',
            '--helper-record',str(root/'helper-source-record.json'),'--model-record',str(root/'source-record.json'),
            '--run',str(run),'--parent',rec['parent'],'--initial',rec['initial'],'--step',str(step),'--output',str(out)])
    full=root/('fullstate2000-'+name)
    add('fullstate2000',full,[PY,'-B','-u',str(Path(model['source'])/'experiments/dynamic_readout_20261031/preflight.py'),
        '--run',str(run),'--output',str(full),'--expected-step','2000','--reference-initial',rec['reference_initial']])
    out=root/('health-'+name)
    add('health',out,[PY,'-B','-u',str(Path(helper['source'])/HEALTH),'--run',str(run),'--output',str(out),
        '--cases','16','--fullstate',str(full),'--source-record',str(root/'source-record.json'),
        '--helper-manifest',helper['manifest'],'--expected-step','2000'])
    out=root/('transition-'+name)
    add('transition',out,[PY,'-B','-u',str(Path(model['source'])/'experiments/ideas_validate_20261031/audit.py'),
        '--run',str(run),'--output',str(out),'--cases','64','--manifest',str(Path(model['source'])/f'experiments/ideas_validate_20261031/audit-windows-{rec["task"]}.json')])
    for phase,key in[('planning','primary'),('fresh-planning','fresh')]:
        out=root/(phase+'-'+name);manifest=rec['references'][key+'_manifest']
        add(phase,out,[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',manifest,
            '--checkpoint',str(run/'model-step0002000'),'--data',rec['config']['data'],'--output',str(out)],
            manifest_sha256=rec['references'][key+'_manifest_sha256'],environment=dict(MUJOCO_EGL_DEVICE_ID=str(rec['gpu'])))
    assert len(jobs)==9
    return jobs


def prepare(args):
    root=Path(args.root);assert not root.exists()
    model=read(PARENT/'source-record.json');helper=read(args.helper_record);sources=checked_sources(helper,model)
    review=read(B/'reports/dynamic-readout600-20261031/summary.json')
    assert review['completion']==dict(training_stages_complete=8,training_os_exits_unknown=8,
        first_recovery_fullstate_exits0=8,preserved_health_exits1=8,first_recovery_blocked_pending=16,second_recovery_actual_exits0=24)
    assert sha(review['full_results']['path'])==review['full_results']['sha256']
    for r in[PARENT,FULL,EVAL]:assert(r/'HOLD_DISPATCH').exists()
    assert not list((PARENT/'exits').glob('*.json'))
    for j in read(EVAL/'plan.json')['jobs']:assert read(EVAL/'exits'/(j['id']+'.json'))['returncode']==0
    records=[]
    for a in allocations():
        old=PARENT/a['id'];cfg=read(old/'config.json');stage=read(old/'stage-complete.json')
        assert stage['stage_complete']and stage['stop_step']==600 and stage['config_sha256']==sha(old/'config.json')
        cal=cfg['spatial_calibration'];assert sha(cal['path'])==cal['sha256']and read(cal['path'])['normalization']=='global_rms'
        native=read(FULL/('fullstate-'+a['id'])/'readout-audit.json');assert native['passed']and sha(old/'last.ckpt')==native['checkpoint_sha256']
        initial=old/'initial-model/weights.pt';pre=audit(old,600,old,initial,model)
        assert pre['checkpoint_sha256']==native['checkpoint_sha256']
        reference=B/'runs/local_isotropic600_20261031'/f'{a["task"]}-spatial-local-seed{a["training_seed"]}'/'initial-model/weights.pt'
        assert sha(reference)==native['reference_initial_sha256']
        records.append(dict(**a,parent=str(old),initial=str(initial),reference_initial=str(reference),
            config=config(cfg,root/a['id']),pre=pre,references=existing_references(a['task'],a['training_seed'],cfg['data'])))
        print(json.dumps(dict(pre600_native_verified=a['id'])),flush=True)
    root.mkdir();write(root/'source-record.json',model);write(root/'helper-source-record.json',helper);write(root/'source-provenance.json',sources)
    jobs=[];copies=[]
    for rec in records:
        old=Path(rec['parent']);run=root/rec['id'];run.mkdir();write(run/'config.json',rec['config'])
        shutil.copy2(old/'last.ckpt',run/'last.ckpt');assert sha(run/'last.ckpt')==rec['pre']['checkpoint_sha256']
        shutil.copytree(old/'initial-model',run/'initial-model')
        assert sha(run/'initial-model/weights.pt')==sha(old/'initial-model/weights.pt')
        post=audit(run,600,old,rec['initial'],model,old/'model-step0000600/weights.pt')
        assert post['checkpoint_sha256']==rec['pre']['checkpoint_sha256']
        assert post['rng_component_sha256']==rec['pre']['rng_component_sha256']and post['sampler']==rec['pre']['sampler']
        write(run/'pre-resume-fullstate.json',rec['pre']);write(run/'post-copy-fullstate.json',post)
        copies.append(dict(id=rec['id'],passed=True,checkpoint_byte_identical=True,initial_byte_identical=True,
            parent=str(old),checkpoint_sha256=post['checkpoint_sha256'],sampler=post['sampler'],rng=post['rng_component_sha256']))
        jobs.extend(jobs_for(rec,root,helper,model));print(json.dumps(dict(postcopy600_native_verified=rec['id'])),flush=True)
    assert len(jobs)==72 and len({j['id']for j in jobs})==72
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=read(EVAL/'plan.json')['environment'],jobs=jobs))
    write(root/'resume-audit.json',copies)
    write(root/'protocol.json',dict(at=time.time(),model_source=model,helper_source=helper,initial_step=600,intermediate_step=1200,stop_step=2000,
        runs=8,endpoints=72,allocations=list(allocations()),references=[dict(id=r['id'],**r['references'])for r in records],
        parent_review_sha256=sha(B/'reports/dynamic-readout600-20261031/summary.json'),training_parent_os_exits_unknown=8,
        original_roots_preserved=[str(PARENT),str(FULL),str(EVAL)],native_archives=[1200,2000],retained_portables=[1200,2000],
        unchanged='d6 model/data/batch128/max_steps124910/warmup1249/sampler/calibration/loss/primary50/existingfresh50;botharms/tasks/seeds.',
        helpers='Reused093 nativeCPU model/optimizer/scheduler/RNG/sampler audit;d6 directp.grad fullstate2000;stricthealth explicitexpected-step2000;dispatcher22e.',
        stop_policy='All chains stop2000. Failed1200 archive blocks2000. No automaticlonger budget;STATE lacks pairedtaskgain at2000 => stopdynamicextension.',
        logger='Preserve four artificialzero exact-parameter tracekeys;aggregate moduletrace and actualsamecheckpointp.grad are separate evidence.',
        candidate_only=True,authorization='Root explicitly approved activation after immutablehelpers/CPUcopyaudits and actualdev15free check. No additionalpermission needed.'))
    print(json.dumps(dict(root=str(root),endpoints=72,prepared=True,launched=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','archive']);p.add_argument('--helper-record',required=True)
    p.add_argument('--model-record');p.add_argument('--root',default=str(ROOT));p.add_argument('--run');p.add_argument('--parent');p.add_argument('--initial')
    p.add_argument('--step',type=int);p.add_argument('--output');a=p.parse_args()
    archive(a)if a.phase=='archive'else prepare(a)
