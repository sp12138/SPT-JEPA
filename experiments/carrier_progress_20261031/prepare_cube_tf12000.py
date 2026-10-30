"""Prepare two audited Cube TF6000→12000 continuations; never launch a worker."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

REPO=next(p for p in Path(__file__).resolve().parents if (p/'experiments/carrier_progress_20261031/prepare_next.py').is_file())
sys.path.insert(0,str(REPO))
from experiments.carrier_progress_20261031.prepare_learning import (
    B,PYTHON,activate_source,digest,read,semantic,write)
from experiments.carrier_progress_20261031.prepare_next import audit_state

PARENT=B/'runs/carrier_learn6000_20261031'
ROOT=B/'runs/cube_tf12000_20261031'
SELF='experiments/carrier_progress_20261031/prepare_cube_tf12000.py'
PENDING_HELPER='__FROZEN_CUBE_TF12000_HELPER_REQUIRED__'


def archive(args):
    source=read(args.source_record);activate_source(source)
    run,out,parent=Path(args.audit_run),Path(args.output),Path(args.parent_run)
    assert not (out/'fullstate.json').exists()
    report=audit_state(run,12000,parent)
    stage=read(run/'stage-complete.json')
    assert stage['initial_step']==6000 and stage['stop_step']==12000 and stage['stage_complete']
    for step in [8000,10000,12000]:
        assert (run/f'model-step{step:07d}/weights.pt').is_file()
    path=run/'checkpoint-step0012000.ckpt';assert not path.exists()
    shutil.copy2(run/'last.ckpt',path);assert digest(path)==report['checkpoint_sha256']
    report.update(archived_checkpoint=str(path),native_stage_complete=stage,
                  source_commit=source['source_commit'],source_record_sha256=digest(args.source_record))
    out.mkdir(parents=True,exist_ok=True);write(out/'fullstate.json',report)
    print(json.dumps(dict(passed=True,step=12000,archive=str(path))),flush=True)


def prepare(args):
    root=Path(args.root);assert not root.exists(),'Preserve existing candidate preparations'
    source=read(args.source_record);oldsource=read(PARENT/'source-record.json')
    assert source==oldsource and source['source_commit']=='8e0f821bba84e7ed8c0f16500d7c880fde56d4cf'
    manifest=read(source['manifest'])
    for name in ['experiments/carrier_progress_20261031/prepare_next.py',
                 'experiments/carrier_progress_20261031/prepare_learning.py']:
        assert digest(REPO/name)==digest(Path(source['source'])/name)==manifest['files'][name]
    plan,protocol=read(PARENT/'plan.json'),read(PARENT/'protocol.json')
    oldjobs=[j for j in plan['jobs'] if j['host']=='dev13']
    assert len(oldjobs)==20
    for j in oldjobs:assert read(PARENT/'exits'/(j['id']+'.json'))['returncode']==0
    launched=read(PARENT/'launched-dev13.json')
    assert launched['host']=='dev13' and 'dev13' in __import__('socket').gethostname()
    assert not Path(f'/proc/{launched["pid"]}').exists(),'Old dispatcher must be closed first'
    closure=dict(host='dev13',old_pid=launched['pid'],pid_absent=True,checked_at=time.time(),
                 all20_endpoints_exit0=True,worker_record_scope='Historical heartbeat can remain stale; PID absence and completed exits are checked.')
    activate_source(source)
    records=[]
    for seed in [1,2]:
        name=f'cube-temporal-forecast-seed{seed}';parent=PARENT/name;cfg=read(parent/'config.json')
        assert read(parent/'stage-complete.json')['stop_step']==6000
        assert cfg['seed']==seed and cfg['max_steps']==124910 and cfg['batch_size']==128
        assert cfg.get('robust_variant') and not cfg.get('model_class')
        pre=audit_state(parent,6000,parent)
        assert pre['passed'] and all(pre['checks'].values())
        newcfg=copy.deepcopy(cfg);newcfg.update(output=str(root/name),checkpoint_every=2000,
                                              retain_steps=[8000,10000,12000])
        assert semantic(newcfg)==semantic(cfg)
        records.append(dict(id=name,seed=seed,parent=str(parent),config=newcfg,pre=pre,
                            host='dev13',gpu=seed-1,task='cube',family='temporal-forecast',arm='temporal-forecast'))
    root.mkdir(parents=True)
    write(root/'source-record.pending.json',source)
    write(root/'predecessor-closure.json',closure)
    jobs=[]
    for rec in records:
        name=rec['id'];out=root/name;out.mkdir();cfgpath=out/'config.json';write(cfgpath,rec['config'])
        shutil.copy2(Path(rec['parent'])/'last.ckpt',out/'last.ckpt')
        assert digest(out/'last.ckpt')==rec['pre']['checkpoint_sha256']
        write(out/'pre-resume-fullstate.json',dict(rec['pre'],copy_byte_identical=True,
            preserved_parent_checkpoint=str(Path(rec['parent'])/'last.ckpt'),parent_source=source))
        common={k:rec[k] for k in ['host','gpu','task','family','arm']}
        common.update(training_seed=rec['seed'],source=source)
        ident='learn12000-'+name
        jobs.append(dict(**common,id=ident,phase='learn12000',output=str(out),config=str(cfgpath),
            config_sha256=digest(cfgpath),command=[PYTHON,'-B','-u','-m','as_jepa_tools.train_stage',
                '--config',str(cfgpath),'--stop-step','12000','--resume']))
        dest=root/('fullstate12000-'+name)
        jobs.append(dict(**common,id=dest.name,phase='fullstate12000',output=str(dest),depends_on=[ident],
            command=[PYTHON,'-B','-u',PENDING_HELPER,'--source-record',str(root/'source-record.json'),
                     '--audit-run',str(out),'--parent-run',rec['parent'],'--output',str(dest)]))
        previous=dest.name;dest=root/('health-'+name)
        jobs.append(dict(**common,id=dest.name,phase='health',output=str(dest),depends_on=[previous],
            command=[PYTHON,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(out),'--output',str(dest)]))
        previous=dest.name
        for phase in ['planning','fresh-planning']:
            old=next(j for j in plan['jobs'] if j['id']==phase+'-'+name)
            command=old['command'];manifest_path=Path(command[command.index('--manifest')+1])
            assert digest(manifest_path)==old['manifest_sha256']
            dest=root/(phase+'-'+name)
            jobs.append(dict(**common,id=dest.name,phase=phase,output=str(dest),depends_on=[previous],
                environment={'MUJOCO_EGL_DEVICE_ID':str(rec['gpu'])},manifest_sha256=old['manifest_sha256'],
                command=[PYTHON,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest_path),
                    '--checkpoint',str(out/'model-step0012000'),'--data',rec['config']['data'],'--output',str(dest)]))
            previous=dest.name
    assert len(jobs)==10
    write(root/'plan.pending.json',dict(root=str(root),deadline=None,environment=plan['environment'],jobs=jobs,
        prepared_only=True,requires_frozen_helper=True))
    write(root/'protocol.pending.json',dict(created_at=time.time(),parent=str(PARENT),initial_step=6000,
        stop_step=12000,runs=2,jobs=10,allocations=[{k:r[k] for k in ['id','seed','host','gpu']} for r in records],
        original_training_source_commit=source['source_commit'],training_semantics_unchanged=True,
        scheduler_horizon=124910,warmup_steps=1249,checkpoint_every=2000,retained_steps=[8000,10000,12000],
        native_fullstate_archive=12000,helper_sha256=digest(__file__),predecessor_closed=closure,
        scope='Candidate only. Native full-state6000 byte-copies; optimizer/scheduler/model/sampler/RNG audits passed. '
              'No update or GPU launch. No automatic continuation beyond12000. All eight Carrier6000 runs remain held.',
        required_before_launch='Root freezes this helper, validates old8e model/trainer/evaluator bytes, replaces pending helper '
            'and source records, and writes the executable plan. No plan.json exists yet.'))
    print(json.dumps(dict(root=str(root),prepared_only=True,fullstate_audits=2,byte_identical_copies=2,
                         scheduler_horizon=124910,warmup_steps=1249,jobs=10),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(ROOT));p.add_argument('--source-record',default=str(PARENT/'source-record.json'))
    p.add_argument('--audit-run');p.add_argument('--parent-run');p.add_argument('--output');a=p.parse_args()
    if a.audit_run:
        assert a.parent_run and a.output;archive(a)
    else:prepare(a)
