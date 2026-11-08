"""One-shot CPU-only evidence snapshot; this does not schedule or launch work."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_transition_only_20261109')


def read(path):
    return json.loads(Path(path).read_text())


def snapshot(host,output):
    plan=read(ROOT/'plan.json');launch=read(ROOT/f'launch-{host}.json')
    processes={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            raw=(p/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
            processes[int(p.name)]=dict(pid=int(p.name),ppid=int(tail[1]),state=tail[0],
                start_ticks=int(tail[19]),argv=(p/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0'))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    owned=set();workers=[]
    for w in launch['workers']:
        p=processes.get(w['pid']);identity=p is not None and p['argv'][-5:]==['work','--host',host,'--gpu',str(w['gpu'])]
        children={w['pid']} if identity else set()
        while True:
            more={pid for pid,r in processes.items() if r['ppid'] in children}-children
            if not more:break
            children|=more
        owned|=children
        workers.append(dict(w,identity_matches=identity,processes=[processes[pid] for pid in sorted(children)]))
    def smi(args):
        return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=20)
    def rows(args):
        return [{k.strip().split(' [')[0]:v.strip() for k,v in r.items()}
            for r in csv.DictReader(io.StringIO(smi(args)))]
    gpus=rows(['--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,nounits'])
    apps=rows(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,nounits'])
    for app in apps:
        pid=int(app['pid'])
        app['owned_by_this_launch']=pid in owned
    statuses={j['name']:read(Path(j['output'])/'status.json') for j in plan['jobs'] if (Path(j['output'])/'status.json').exists()}
    errors={str(p.relative_to(ROOT)):read(p) for p in ROOT.rglob('ERROR*.json')}
    for p in ROOT.glob('evaluation-errors/*.json'):errors[str(p.relative_to(ROOT))]=read(p)
    source=Path(plan['source'])
    source_ok=all(hashlib.sha256((source/p).read_bytes()).hexdigest()==h for p,h in plan['source_hashes'].items())
    partial=[]
    for e in plan['evaluations']:
        p=Path(e['output'])/'progress.json'
        if p.exists():
            rows=read(p)['rows'];partial.append(dict(name=e['name'],episodes=len(rows),successes=sum(r['success'] for r in rows),
                by_task={t:dict(episodes=sum(r['task']==t for r in rows),successes=sum(r['task']==t and r['success'] for r in rows)) for t in sorted({r['task'] for r in rows})},
                complete=(p.parent/'metrics.json').exists()))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=host,workers=workers,gpus=gpus,compute_apps=apps,
        owned_compute_apps=sum(a['owned_by_this_launch'] for a in apps),statuses=statuses,errors=errors,
        preflights=len(list(ROOT.glob('training/*/PREFLIGHT.json'))),cpu_entry_audits=len(list(ROOT.glob('training/*/audit-0000002.json'))),
        technical_endpoints=len(list(ROOT.glob('training/*/technical-*/metrics.json'))),global_gate=read(ROOT/'TRAINING_APPROVED.json') if (ROOT/'TRAINING_APPROVED.json').exists() else None,
        source_hashes_match=source_ok,trainings_complete=len(list(ROOT.glob('training/*/TRAINING_COMPLETE.json'))),
        summary=read(ROOT/'results/summary.json') if (ROOT/'results/summary.json').exists() else None,
        partial_physical_evaluations=partial,closed=read(ROOT/'CLOSED.json') if (ROOT/'CLOSED.json').exists() else None,
        new_model_updates=0,new_physics_steps=0,new_gpu_contexts=0)
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_name(output.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,output)
    print(json.dumps(dict(utc=result['utc'],host=host,owned_compute_apps=result['owned_compute_apps'],
        utilization=[r['utilization.gpu'] for r in gpus],preflights=result['preflights'],
        trainings_complete=result['trainings_complete'],step_range=[min(r['step'] for r in statuses.values()),max(r['step'] for r in statuses.values())] if statuses else [],
        errors=len(errors),source_hashes_match=source_ok,evaluation_endpoints=result['summary']['evaluation_endpoints_complete'] if result['summary'] else 0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();snapshot(a.host,a.output)
