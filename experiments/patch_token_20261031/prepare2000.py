"""Only the eight approved native600→1200→2000 patch continuations; no launch."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from prepare_preflight import B,PY,read,sha,write,allocations
from resume_audit import audit_state,RECORD,FILES

PARENT=B/'runs/patch_token600_20261031'
ROOT=B/'runs/patch_token2000_20261031'
SELF='experiments/patch_token_20261031/prepare2000.py'
FULL='experiments/patch_token_20261031/preflight2000.py'
MODEL_COMMIT='1018ebbe9beb3e5f9b7c91018f03ff8a4f17884d'


def semantic(c):return {k:v for k,v in c.items()if k not in{'output','checkpoint_every','retain_steps','is_resuming'}}
def config(old,out):
    c=copy.deepcopy(old);c.update(output=str(out),checkpoint_every=1200,retain_steps=[1200,2000])
    assert semantic(c)==semantic(old)and c['batch_size']==128 and c['max_steps']==124910
    assert c['model_class']=='patch_carrier'and c['model']['patch_basis']in['identity','orthogonal']
    assert c['model']['patch_predictor_seed']==c['seed']
    return c


def checked_sources(helper,model):
    assert model['source_commit']==MODEL_COMMIT
    for source in[helper,model]:
        m=read(source['manifest']);assert m['source_commit']==source['source_commit']
        for rel,d in m['files'].items():assert sha(Path(source['source'])/rel)==d,rel
    assert sha(__file__)==read(helper['manifest'])['files'][SELF]
    # The original native fullstate body and all model/training files remain1018.
    hm,mm=read(helper['manifest']),read(model['manifest'])
    for rel in mm['files']:
        if rel.startswith(('as_jepa_lewm/','as_jepa_tools/'))or rel=='experiments/ideas_validate_20261031/preflight.py':
            assert hm['files'][rel]==mm['files'][rel],rel
    auditor=read(RECORD);m=read(auditor['manifest'])
    assert auditor['source_commit']=='0934d3c445e6c754a6ab547d801890c34740b330'
    for rel in FILES:assert sha(Path(auditor['source'])/rel)==m['files'][rel]
    proof=read(Path(helper['source'])/'experiments/patch_token_20261031/cpu/prepare2000.json')
    assert proof['passed']and proof['endpoints']==72
    for rel,d in proof['source_sha256'].items():assert hm['files'][rel]==d
    return dict(model=model,helper=helper,pinned_native_helpers=auditor,
        reused_helper_files={p:m['files'][p]for p in FILES})


def archive(args):
    helper,model=read(args.helper_record),read(args.model_record);sources=checked_sources(helper,model)
    assert args.step in[1200,2000]
    run=Path(args.run);stage=read(run/'stage-complete.json')
    assert stage['stage_complete']and stage['stop_step']==args.step
    assert stage['initial_step']==(600 if args.step==1200 else 1200)
    result=audit_state(run,args.step,args.parent,args.initial,model)
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
    primary_manifest=Path(data)/'manifests/validation-strict.json';fresh_manifest=Path(f['manifest'])
    assert sha(primary_manifest)==p['manifest_sha256']and sha(fresh_manifest)==f['manifest_sha256']
    assert p['normalization']==f['normalization']=='train episodes'
    assert p['solver']==f['solver']and p['plan']==f['plan']and p['protocol']==f['protocol']
    return dict(primary_metrics=str(primary),primary_metrics_sha256=sha(primary),
        fresh_metrics=str(fresh),fresh_metrics_sha256=sha(fresh),
        primary_manifest=str(primary_manifest),primary_manifest_sha256=sha(primary_manifest),
        fresh_manifest=str(fresh_manifest),fresh_manifest_sha256=sha(fresh_manifest),
        scope='Existing flat local1024 identity2000 control; same raw states and cases, different predictor capacity/compute. No newcase generation.')


def jobs_for(rec,root,helper,model):
    root=Path(root);name=rec['id'];run=root/name;cfg=run/'config.json';deps=[];jobs=[]
    common={k:v for k,v in rec.items()if k in['host','gpu','task','training_seed','arm']};common['source']=model
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
    peer=f'{rec["task"]}-patch-{"orthogonal"if rec["arm"]=="identity"else"identity"}-seed{rec["training_seed"]}'
    add('fullstate2000',full,[PY,'-B','-u',str(Path(helper['source'])/FULL),
        '--run',str(run),'--output',str(full),'--expected-step','2000',
        '--reference-initial',rec['reference_initial'],'--paired-initial',str(root/peer/'initial-model/weights.pt')])
    out=root/('health-'+name)
    add('health',out,[PY,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(run),'--output',str(out),'--cases','16'])
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
    review=read(B/'reports/patch-token600-20261031/summary.json')
    assert review['passed']and review['all40_actual_exit0']and len(review['checks'])==8
    assert sha(review['full_results']['path'])==review['full_results']['sha256']
    assert(PARENT/'HOLD').exists()and(PARENT/'HOLD_DISPATCH').exists()
    assert read(PARENT/'closed.json')['all40_actual_exit0']
    plan=read(PARENT/'plan.json');assert len(plan['jobs'])==40
    for j in plan['jobs']:assert read(PARENT/'exits'/(j['id']+'.json'))['returncode']==0
    records=[]
    for a in allocations():
        old=PARENT/a['id'];cfg=read(old/'config.json');stage=read(old/'stage-complete.json')
        assert stage['stage_complete']and stage['stop_step']==600 and stage['config_sha256']==sha(old/'config.json')
        cal=cfg['spatial_calibration'];assert sha(cal['path'])==cal['sha256']and read(cal['path'])['normalization']=='global_rms'
        native=read(PARENT/('fullstate-'+a['id'])/'preflight.json');extra=read(PARENT/('fullstate-'+a['id'])/'patch-audit.json')
        assert native['passed']and extra['passed']and sha(old/'last.ckpt')==native['checkpoint_sha256']
        initial=old/'initial-model/weights.pt';pre=audit_state(old,600,old,initial,model)
        assert pre['checkpoint_sha256']==native['checkpoint_sha256']
        reference=B/'runs/local_isotropic600_20261031'/f'{a["task"]}-spatial-local-seed{a["training_seed"]}'/'initial-model/weights.pt'
        assert sha(reference)==extra['binding'][str(reference)]
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
        # Parent600 remains retained; preserve an additional explicit archive here.
        shutil.copy2(run/'last.ckpt',run/'checkpoint-step0000600.ckpt')
        assert sha(run/'checkpoint-step0000600.ckpt')==rec['pre']['checkpoint_sha256']
        post=audit_state(run,600,old,rec['initial'],model,old/'model-step0000600/weights.pt')
        assert post['checkpoint_sha256']==rec['pre']['checkpoint_sha256']
        assert post['rng_component_sha256']==rec['pre']['rng_component_sha256']and post['sampler']==rec['pre']['sampler']
        write(run/'pre-resume-fullstate.json',rec['pre']);write(run/'post-copy-fullstate.json',post)
        copies.append(dict(id=rec['id'],passed=True,checkpoint_byte_identical=True,initial_byte_identical=True,
            parent=str(old),checkpoint_sha256=post['checkpoint_sha256'],sampler=post['sampler'],rng=post['rng_component_sha256']))
        jobs.extend(jobs_for(rec,root,helper,model));print(json.dumps(dict(postcopy600_native_verified=rec['id'])),flush=True)
    assert len(jobs)==72 and len({j['id']for j in jobs})==72
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=plan['environment'],jobs=jobs))
    write(root/'resume-audit.json',copies)
    write(root/'protocol.json',dict(at=time.time(),model_source=model,helper_source=helper,initial_step=600,intermediate_step=1200,stop_step=2000,
        runs=8,endpoints=72,allocations=list(allocations()),references=[dict(id=r['id'],**r['references'])for r in records],
        parent_review_sha256=sha(B/'reports/patch-token600-20261031/summary.json'),
        original_roots_preserved=[str(PARENT)],native_archives=[600,1200,2000],retained_portables=[1200,2000],
        unchanged='1018 model/data/batch128/max_steps124910/warmup1249/sampler/calibration/loss/primary50/existingfresh50;botharms/tasks/seeds.',
        helpers='Pinned093 CPU native restore primitives with patch class/basis assertions;strict original fullstate+patch per-layer gradients at2000;originalhealth16;dispatcher22e.',
        rng_boundary='Exact copied checkpoint and saved RNG-component hashes; actual model/AdamW/scheduler restore. Stateless sampler resume indices versus start0. No bitwise uninterrupted GPU trajectory claim.',
        stop_policy='All chains stop2000. Failed1200 archive blocks2000. No automaticlonger budget/newseed/newcase/controller/grid.',
        candidate_only=True,authorization='Root explicitly approved activation after immutablehelpers/CPUcopyaudits and actual reserved8GPUfree check.'))
    (root/'HOLD').write_text('Prepared bounded2000; activate only after all16 CPU native audits and reserved8GPU idle verification.\n')
    print(json.dumps(dict(root=str(root),endpoints=72,prepared=True,launched=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','archive']);p.add_argument('--helper-record',required=True)
    p.add_argument('--model-record');p.add_argument('--root',default=str(ROOT));p.add_argument('--run');p.add_argument('--parent');p.add_argument('--initial')
    p.add_argument('--step',type=int);p.add_argument('--output');a=p.parse_args()
    archive(a)if a.phase=='archive'else prepare(a)
