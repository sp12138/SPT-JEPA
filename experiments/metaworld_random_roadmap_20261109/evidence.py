"""One-shot session health/evidence capture, no scheduler or experiment launcher."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime,timezone

BASE=Path('/efs/prescottsun/SPT-JEPA')
REPORT=BASE/'reports/metaworld_persistent_world_20261109'
NAMES=['metaworld_boundary_world_20261109','metaworld_interventions_20261109',
       'metaworld_covariance_world_20261109','metaworld_persistent_world_20261109','metaworld_persistent_world_20261109_v2','metaworld_observer_probe_20261109','metaworld_random_roadmap_20261109','metaworld_random_inverse_audit_20261109','metaworld_causal_inverse_20261109']


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')


def snapshot(host,label):
    commands=[['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],
              ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],['ps','-eo','pid,ppid,etime,args']]
    results=[]
    for c in commands:
        r=subprocess.run(c,capture_output=True,text=True,timeout=30)
        output=r.stdout if c[0]!='ps' else '\n'.join(x for x in r.stdout.splitlines() if '/efs/prescottsun/SPT-JEPA/' in x)
        results.append(dict(command=c,returncode=r.returncode,stdout=output,stderr=r.stderr))
    states={}
    for name in NAMES:
        root=BASE/'runs'/name
        paths=list(root.glob('results/summary.json'))+list(root.glob('CLOSED.json'))+list(root.glob('training/*/status.json'))
        paths+=list(root.glob('recovery-lazy-v1/results/summary.json'))+list(root.glob('recovery-lazy-v1/CLOSED.json'))
        paths+=list(root.glob('evaluation-v1/results/summary.json'))+list(root.glob('evaluation-v1/CLOSED.json'))+list(root.glob('results/edges-summary.json'))+list(root.glob('GRAPH_READY.json'))
        states[name]={str(p.relative_to(root)):json.loads(p.read_text()) for p in paths}
    sources={}
    for root in [BASE/'runs/metaworld_covariance_world_20261109',BASE/'runs/metaworld_persistent_world_20261109_v2',BASE/'runs/metaworld_causal_inverse_20261109']:
        p=json.loads((root/'plan.json').read_text());s=Path(p['source'])
        actual={str(f.relative_to(s)):sha(f) for f in s.rglob('*.py')}
        sources[root.name]=dict(passed=actual==p['source_hashes'],files=len(actual))
    target=REPORT/f'{label}-{host}.json';assert not target.exists()
    save(target,dict(utc=now(),host=host,resources=results,states=states,source_hashes=sources,
        scope='Actualone-timesnapshot,NOTcontinuousutilization;24PHYSICALcards. No scheduler/newgoal/LLMmonitor.'))


def archive(label):
    stage=REPORT/'evidence'/label;stage.mkdir(parents=True,exist_ok=False);copied=[];skipped=[]
    for name in NAMES:
        for top in [BASE/'runs'/name,BASE/'reports'/name]:
            if not top.exists():continue
            for source in top.rglob('*'):
                if not source.is_file() or source.suffix not in ['.json','.log','.py','.md']:continue
                if REPORT/'evidence' in source.parents:continue
                try:
                    stat=source.stat()
                    if stat.st_size>2_000_000:continue
                    content=source.read_bytes();target=stage/source.relative_to(BASE);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
                    copied.append(dict(path=str(source.relative_to(BASE)),bytes=len(content),sha256=sha(target),source_mtime_ns=stat.st_mtime_ns))
                except FileNotFoundError:skipped.append(str(source))
    manifest=stage/'SOURCE_MANIFEST.json';save(manifest,dict(utc=now(),files=copied,skipped=skipped,
        scope='Per-filecontentssnapshot,NOTgloballyatomic/currentfinalevidence. Models/optimizer/banks/tracearraysremainEFS. No trainingdata images archived.'))
    output=REPORT/(label+'-evidence.tar.gz')
    with tarfile.open(output,'x:gz') as tar:
        for p in sorted(stage.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(stage)))
    seal=dict(utc=now(),path=str(output),sha256=sha(output),files=len(copied)+1,bytes=output.stat().st_size,skipped=len(skipped))
    save(REPORT/(label+'-ARCHIVE.json'),seal);print(json.dumps(seal))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['snapshot','archive']);a.add_argument('--host');a.add_argument('--label',required=True);args=a.parse_args()
    if args.mode=='snapshot':snapshot(args.host,args.label)
    else:archive(args.label)
