"""Candidate-only2000→4000→6000 maturity control with mandatory fresh evaluation."""
import argparse,copy,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.local_carrier_20261031 import prepare2000 as core
B,PYTHON=core.B,core.PYTHON
read,write,digest,semantic=core.read,core.write,core.digest,core.semantic
PARENT=B/'runs/local_isotropic2000_20261031'
ROOT=B/'runs/local_isotropic6000_20261031'
FRESH=B/'runs/followup_fresh2000_20261031'
SELF='experiments/local_carrier_20261031/prepare6000.py'


def source_checks(helper,training):
    report=core.source_checks(helper,training);manifest=read(helper['manifest'])
    assert digest(__file__)==digest(Path(helper['source'])/SELF)==manifest['files'][SELF]
    return report


def continuation_config(cfg,out):
    new=core.continuation_config(cfg,out);new.update(checkpoint_every=2000,retain_steps=[4000,6000])
    assert semantic(new)==semantic(cfg)
    return new


def fresh_references(freshroot,parent):
    freshroot,parent=Path(freshroot),Path(parent);plan=read(freshroot/'plan.json');assert len(plan['jobs'])==12
    for j in plan['jobs']:assert read(freshroot/'exits'/(j['id']+'.json'))['returncode']==0
    records={};manifests={}
    for rec in core.allocations():
        expected=parent/rec['id']/'model-step0002000'
        jobs=[j for j in plan['jobs'] if '--checkpoint' in j['command'] and
              Path(j['command'][j['command'].index('--checkpoint')+1])==expected]
        assert len(jobs)==1;job=jobs[0];cmd=job['command'];path=Path(cmd[cmd.index('--manifest')+1]);m=read(path)
        report=read(Path(job['output'])/'metrics.json');cfg=read(parent/rec['id']/'config.json')
        assert report['manifest_sha256']==digest(path) and report['checkpoint_audit']['strict_equal']
        assert Path(report['checkpoint'])==expected and len(report['pairs'])==len(m['pairs'])==50
        assert m['task']==rec['task'] and m['campaign_split']=='validation'
        assert m['metadata_card_sha256']==digest(Path(cfg['data'])/'dataset_card.json')
        assert m['fresh_selection']['primary_manifest_sha256']==digest(Path(cfg['data'])/'manifests/validation-strict.json')
        item=dict(path=str(path),sha256=digest(path))
        if rec['task'] in manifests:assert manifests[rec['task']]==item
        else:manifests[rec['task']]=item
        records[rec['id']]=dict(evaluation=str(Path(job['output'])/'metrics.json'),sha256=digest(Path(job['output'])/'metrics.json'),
            success_rate=report['metrics']['success_rate'],manifest=item)
    return manifests,records


def jobs_for(rec,root,helper,training):
    root=Path(root);name=rec['id'];out=root/name;cfg=out/'config.json';common={k:rec[k] for k in ('host','gpu','task','training_seed','family','arm')}
    jobs=[];dep=[]
    for step in (4000,6000):
        ident=f'learn{step}-{name}';jobs.append(dict(**common,source=training,id=ident,output=str(out),config=str(cfg),config_sha256=digest(cfg),
            phase=f'learn{step}',depends_on=dep,command=[PYTHON,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(cfg),'--stop-step',str(step),'--resume']))
        dest=root/f'fullstate{step}-{name}';jobs.append(dict(**common,source=helper,id=dest.name,output=str(dest),phase=f'fullstate{step}',depends_on=[ident],
            command=[PYTHON,'-B','-u',str(Path(helper['source'])/SELF),'--source-record',str(root/'helper-source-record.json'),
                '--training-source-record',str(root/'source-record.json'),'--audit-run',str(out),'--parent-run',rec['parent'],
                '--initial-weights',rec['initial_weights'],'--expected-step',str(step),'--output',str(dest)]));dep=[dest.name]
    dest=root/('health-'+name);jobs.append(dict(**common,source=training,id=dest.name,output=str(dest),phase='health',depends_on=dep,
        command=[PYTHON,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(out),'--output',str(dest)]));dep=[dest.name]
    dest=root/('transition-'+name);jobs.append(dict(**common,source=training,id=dest.name,output=str(dest),phase='transition',depends_on=dep,
        command=[PYTHON,'-B','-u',str(Path(training['source'])/'experiments/ideas_validate_20261031/audit.py'),'--run',str(out),'--output',str(dest),
            '--cases','64','--manifest',str(Path(training['source'])/f'experiments/ideas_validate_20261031/audit-windows-{rec["task"]}.json')]));dep=[dest.name]
    for phase,manifest in [('planning',rec['primary_manifest']),('fresh-planning',rec['fresh_manifest'])]:
        dest=root/(phase+'-'+name);jobs.append(dict(**common,source=training,id=dest.name,output=str(dest),phase=phase,depends_on=dep,
            manifest_sha256=manifest['sha256'],environment=dict(MUJOCO_EGL_DEVICE_ID=str(rec['gpu'])),
            command=[PYTHON,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',manifest['path'],
                '--checkpoint',str(out/'model-step0006000'),'--data',rec['config']['data'],'--output',str(dest)]));dep=[dest.name]
    return jobs


def archive(args):
    helper,training=read(args.source_record),read(args.training_source_record);source_checks(helper,training)
    assert args.expected_step in (4000,6000);out=Path(args.output);assert not (out/'fullstate.json').exists()
    report=core.audit_state(args.audit_run,args.expected_step,args.parent_run,args.initial_weights,training)
    stage=read(Path(args.audit_run)/'stage-complete.json')
    assert stage['stage_complete'] and stage['stop_step']==args.expected_step and stage['initial_step']==(2000 if args.expected_step==4000 else 4000)
    target=Path(args.audit_run)/f'checkpoint-step{args.expected_step:07d}.ckpt';assert not target.exists()
    shutil.copy2(Path(args.audit_run)/'last.ckpt',target);assert digest(target)==report['checkpoint_sha256']
    report.update(archived_checkpoint=str(target),native_stage_complete=stage,helper_source_commit=helper['source_commit'])
    out.mkdir(parents=True,exist_ok=True);write(out/'fullstate.json',report);print(json.dumps(dict(passed=True,step=args.expected_step,archive=str(target))),flush=True)


def prepare(args):
    root,parent=Path(args.root),Path(args.parent);assert not root.exists()
    helper,training=read(args.source_record),read(parent/'source-record.json');compatibility=source_checks(helper,training)
    plan=read(parent/'plan.json');assert len(plan['jobs'])==64
    for j in plan['jobs']:assert read(parent/'exits'/(j['id']+'.json'))['returncode']==0
    fresh,confirmation=fresh_references(args.fresh_root,parent);records=[]
    for rec in core.allocations():
        old=parent/rec['id'];cfg=read(old/'config.json');stage=read(old/'stage-complete.json')
        assert stage['stage_complete'] and stage['stop_step']==2000
        initial=read(old/'pre-resume-fullstate.json')['initial_weights'];cal=cfg['spatial_calibration'];assert digest(cal['path'])==cal['sha256']
        assert read(cal['path'])['normalization']=='global_rms' and 'spatial_basis' not in cfg['model']
        previous=read(parent/f'fullstate2000-{rec["id"]}/fullstate.json');assert previous['passed'] and all(previous['checks'].values())
        pre=core.audit_state(old,2000,old,initial,training)
        assert pre['checkpoint_sha256']==previous['checkpoint_sha256']
        primary=Path(cfg['data'])/'manifests/validation-strict.json'
        records.append(dict(**rec,parent=str(old),initial_weights=initial,config=continuation_config(cfg,root/rec['id']),pre_audit=pre,
            primary_manifest=dict(path=str(primary),sha256=digest(primary)),fresh_manifest=fresh[rec['task']]))
        print(json.dumps(dict(pre_resume_passed=rec['id'])),flush=True)
    if args.dry_run:print(json.dumps(dict(dry_run=True,audits_passed=8,allocations=list(core.allocations()))));return
    root.mkdir();write(root/'source-record.json',training);write(root/'helper-source-record.json',helper);write(root/'source-compatibility.json',compatibility)
    jobs=[];copies=[]
    for rec in records:
        out=root/rec['id'];out.mkdir();write(out/'config.json',rec['config']);old=Path(rec['parent'])
        shutil.copy2(old/'last.ckpt',out/'last.ckpt');assert digest(out/'last.ckpt')==rec['pre_audit']['checkpoint_sha256']
        report=dict(rec['pre_audit'],copy_byte_identical=True,copied_to=str(out/'last.ckpt'),
            copy_scope='One actual native2000 CPU resume audit; post-copy full-file hash proves all saved model/optimizer/scheduler/RNG/cursor bytes equal. No second native load claimed.')
        write(out/'pre-resume-fullstate.json',report);copies.append(dict(id=rec['id'],checkpoint_sha256=report['checkpoint_sha256'],passed=True))
        jobs.extend(jobs_for(rec,root,helper,training))
    assert len(jobs)==64 and len({j['id'] for j in jobs})==64
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=plan['environment'],jobs=jobs));write(root/'resume-audit.json',copies)
    write(root/'protocol.json',dict(created_at=time.time(),root=str(root),parent=str(parent),training_source_commit=training['source_commit'],
        helper_source_commit=helper['source_commit'],initial_step=2000,intermediate_step=4000,stop_step=6000,runs=8,jobs=64,
        allocations=list(core.allocations()),native_fullstate_archives=[4000,6000],scheduler_horizon=124910,warmup_steps=1249,
        semantic_config_unchanged=True,original_calibration_unchanged=True,validation_cases=50,transition_windows=64,
        fresh_manifests=fresh,fresh2000_evidence=confirmation,candidate_only=True,
        reason='Bounded maturity control, not winner promotion: Cube local primary2000 advantage did not reproduce on fresh cases. Retain both arms/tasks/seeds.',
        endpoints='Native4000/archive/6000/archive;health;transition;primary50;mandatory fresh50. Frozen observed information already checked, no duplicate probe.',
        source_scope='Original32fad0c training and GPU evaluations. New helper only CPU full-state audit/archive, importing original source explicitly. Orthogonal-basis control is not enabled.',
        departures=['Fresh independent directories; original2000/source/config/calibration/holds untouched.',
            'Common4000 native resume boundary; checkpoint cadence2000, retained portable/full4000/6000.',
            'No optimizer/scheduler reset, loss change, new images/labels or model forward in preparation.'],
        stop_policy='Failed4000 archive blocks6000. Stop6000; root must activate candidate-plan explicitly and review all primary/fresh results before further work.'))
    assert not (root/'plan.json').exists();print(json.dumps(dict(candidate=str(root/'candidate-plan.json'),runs=8,endpoints=64,launched=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-record',required=True);p.add_argument('--training-source-record')
    p.add_argument('--root',default=str(ROOT));p.add_argument('--parent',default=str(PARENT));p.add_argument('--fresh-root',default=str(FRESH));p.add_argument('--dry-run',action='store_true')
    p.add_argument('--audit-run');p.add_argument('--parent-run');p.add_argument('--initial-weights');p.add_argument('--expected-step',type=int,choices=[4000,6000]);p.add_argument('--output')
    a=p.parse_args()
    if a.audit_run:
        assert a.training_source_record and a.parent_run and a.initial_weights and a.expected_step and a.output and not a.dry_run;archive(a)
    else:prepare(a)
