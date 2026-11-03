"""Finite two-host recovery; shared archived states, no new training recipe."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.metaworld_eval_20261103.worker import digest, read, save


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def link_tree(source, destination):
    # Native checkpoint writers replace files atomically. Archives are immutable.
    shutil.copytree(source, destination, copy_function=os.link)


def prepare(old, root, helper):
    assert not root.exists(), root
    base=old.parent.parent; source=base/'reports/metaworld_eval_20261103/runtime-60c887e'
    parent_plan=read(old/'plan-paired-v2.json'); plan=copy.deepcopy(parent_plan)
    jobs=read(old/'evaluation-plan.json'); gate=read(old/'TRAINING_APPROVED.json')
    assert gate['passed'] and read(old/'RENDERER_WARMUP_APPROVED.json')['passed']
    root.mkdir(); (root/'workers').mkdir()
    mapping={}; config_hashes={}; migrations=[]
    for recipe in plan['recipes']:
        parent=Path(recipe['run']); name=recipe['name']
        if recipe['arm'] not in ['temporal-forecast','as-jepa']:
            assert read(parent/'complete.json')['passed']
            config_hashes[name]=digest(parent/'config.json')
            continue
        run=root/'training'/name; run.mkdir(parents=True)
        cfg=read(parent/'config.json'); changed=copy.deepcopy(cfg); changed['output']=str(run)
        assert {k for k in cfg if cfg[k]!=changed[k]}=={'output'}
        save(run/'config.json',changed)
        for step in [2,600,2472]:
            archive=parent/f'archive-step{step:07d}'; manifest=read(archive/'manifest.json')
            assert manifest['step']==step and all(digest(archive/k)==v for k,v in manifest['files'].items())
            link_tree(archive,run/archive.name)
            link_tree(parent/f'model-step{step:07d}',run/f'model-step{step:07d}')
            shutil.copy2(archive/'audit.json',run/(f'inherited-audit-step{step}.json' if step==2472 else f'audit-step{step}.json'))
        link_tree(parent/'initial-model',run/'initial-model')
        entry=parent/'archive-step0002472'
        os.link(entry/'native.ckpt',run/'last.ckpt')
        for name in ['status.json','stage-complete.json']:shutil.copy2(entry/name,run/name)
        config_hashes[recipe['name']]=digest(run/'config.json')
        proof=dict(passed=True,step=2472,parent_run=str(parent),parent_checkpoint=str(entry/'native.ckpt'),
                   checkpoint_sha256=digest(run/'last.ckpt'),config_sha256=config_hashes[recipe['name']],
                   parent_config_sha256=digest(parent/'config.json'),runtime_source=str(source),
                   config_changed_keys=['output'],created_utc=now(),
                   scope='Saved full model/AdamW/scheduler/RNG/sample cursor,not weights-only initialization. '
                         'Earlier archives/audits are inherited unchanged;new entry restore/backward audit precedes continuation. '
                         'Any uncheckpointed work on the excluded host is unrecoverable.')
        save(run/'MIGRATION.json',proof); migrations.append(dict(run=str(run),**proof))
        mapping[str(parent)]=str(run); recipe.update(run=str(run),parent_run=str(parent),
                host='dev13' if recipe['seed']==1 else 'dev15',
                gpu=0 if recipe['arm']=='temporal-forecast' else 1,
                config_sha256=config_hashes[recipe['name']])
    assert len(migrations)==4
    # Every completed endpoint retains its original path and exact metrics.
    # Only the unavailable mature checkpoints are redirected to continuations.
    for job in jobs['jobs']:
        if job['run'] in mapping and job['step']==8240:
            assert not any((Path(job['work'])/x).exists() for x in ['complete.json','failed.json','claim'])
            job['original_run']=job['run'];job['run']=mapping[job['run']]
            job['checkpoint']=str(Path(job['run'])/'model-step0008240')
    save(root/'plan.json',plan);save(root/'evaluation-plan.json',jobs)
    dispatch=read(old/'dispatch.json');dispatch.update(plan_path=str(root/'plan.json'),helper=str(helper),
              previous_root=str(old),allowed_hosts=['dev13','dev15'],
              scope='Same8 learners/192 primary endpoints/8240 update cap/original deadline. Four archived2472 continuations;'
                    'all128 completed endpoints reused. No new seeds,SGD recipes,planner grid or recurring LLM monitor.')
    save(root/'dispatch.json',dispatch)
    for name in ['DATA_IO_APPROVED.json','RENDERER_WARMUP_APPROVED.json','RENDERER_REPAIR_REQUESTS.json']:
        shutil.copy2(old/name,root/name)
    gate.update(config_sha256=config_hashes,parent_gate=str(old/'TRAINING_APPROVED.json'),
                migration_entry_audit_required=True,scope='Original training approval plus output-only migration;actual new restore audits gate SGD.')
    save(root/'TRAINING_APPROVED.json',gate)
    diagnostics=[]
    for recipe in parent_plan['recipes']:
        steps=[2472,8240] if recipe['arm'] in ['native-lewm','forecast-render'] else [2472]
        for step in steps:
            name=f"step{step:07d}-{recipe['name']}"
            diagnostics.append(dict(name=name,checkpoint=str(Path(recipe['run'])/f'model-step{step:07d}'),
                                    work=str(root/'diagnostics'/name),arm=recipe['arm'],seed=recipe['seed'],step=step))
    save(root/'diagnostic-plan.json',dict(jobs=diagnostics,seed=20261104,windows_per_task=32,
          scope='Read-only50-task held-out observed-future/action-order diagnostic;no simulation,SGD,optimal-action labels or planner changes.'))
    save(root/'PREPARED.json',dict(passed=True,created_utc=now(),migrations=migrations,allowed_hosts=['dev13','dev15'],
                                  reused_endpoints=128,remaining_primary_endpoints=64,diagnostic_jobs=len(diagnostics)))
    print(json.dumps(dict(passed=True,root=str(root),migrations=4,diagnostics=len(diagnostics))),flush=True)


def retire_controller(launcher, root, script):
    import psutil
    try:outer=psutil.Process(launcher)
    except psutil.NoSuchProcess:return dict(pid=launcher,already_exited=True)
    assert str(root) in outer.cmdline() and any(x.endswith('/'+script) for x in outer.cmdline())
    children=outer.children()
    controllers=[p for p in children if any(x.endswith('/'+script) for x in p.cmdline())]
    assert len(controllers)==1
    process=controllers[0];process.suspend()
    # Current controllers were observed waiting with no GPU jobs. Refuse a
    # race rather than killing any in-flight native trainer/evaluator.
    assert not process.children(recursive=True), (process.pid,'In-flight child; drain before replacement')
    record=dict(launcher=launcher,controller=process.pid,command=process.cmdline())
    process.kill();process.wait(timeout=10)
    return record


def activate(old,root,helper,host):
    assert host in ['dev13','dev15'] and read(root/'PREPARED.json')['passed']
    output=root/f'activation-{host}.json';assert not output.exists()
    base=old.parent.parent;source=base/'reports/metaworld_eval_20261103/runtime-60c887e'
    dispatch=read(root/'dispatch.json');python='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
    gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    assert len(gpu_state.splitlines())==8 and all(int(x.split(',')[1])<=1 for x in gpu_state.splitlines()),gpu_state
    retired=[retire_controller(j['new_launcher_pid'],old,'worker.py') for j in read(old/f'renderer-activation-{host}.json')['jobs']]
    if host=='dev13':
        collector=read(old/'renderer-v2-completion/launch.json')
        retired.append(retire_controller(collector['pid'],old/'renderer-v2-completion','finish.py'))
    assignments={r['gpu']:r for r in read(root/'plan.json')['recipes'] if r.get('parent_run') and r['host']==host}
    assert set(assignments)=={0,1}
    env=os.environ.copy();env.update(PYTHONPATH=str(helper),PYTHONDONTWRITEBYTECODE='1')
    rows=[]
    for gpu in range(8):
        recipe=assignments.get(gpu);mode='train' if recipe else 'evaluate'
        cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(max(1,int(dispatch['deadline_unix']-time.time()))),
             python,'-B','-u',str(helper/'experiments/metaworld_eval_20261103/worker.py'),
             '--root',str(root),'--mode',mode,'--source',str(source),'--helper',str(helper),
             '--base',str(base),'--gpu',str(gpu),'--host',host]
        if recipe:cmd+=['--run',recipe['run'],'--resume-step','2472']
        with (root/'workers'/f'{host}-g{gpu}.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rows.append(dict(gpu=gpu,mode=mode,run=recipe['run'] if recipe else None,pid=p.pid,command=cmd))
    if host=='dev13':
        cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(max(1,int(dispatch['deadline_unix']-time.time())+300)),
             python,'-B','-u',str(helper/'experiments/metaworld_eval_20261103/finish.py'),'--root',str(root)]
        with (root/'completion.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'completion-launch.json',dict(host=host,pid=p.pid,command=cmd))
    save(output,dict(passed=True,host=host,created_utc=now(),gpu_before=gpu_state,retired=retired,jobs=rows))
    print(json.dumps(dict(passed=True,host=host,pids=[r['pid'] for r in rows])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','activate'],required=True)
    for arg in ['old','root','helper']:parser.add_argument('--'+arg,required=True)
    parser.add_argument('--host',choices=['dev13','dev15']);args=parser.parse_args()
    if args.mode=='prepare':prepare(Path(args.old),Path(args.root),Path(args.helper))
    else:activate(Path(args.old),Path(args.root),Path(args.helper),args.host)
