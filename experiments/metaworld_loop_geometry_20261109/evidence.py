"""One-shot resource/evidence snapshots; never schedules or launches experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime,timezone

BASE=Path('/efs/prescottsun/SPT-JEPA')
REPORT=BASE/'reports/metaworld_loop_geometry_20261109'
ROOTS=['metaworld_joint_flow_20261109','metaworld_contact_geometry_20261109',
       'metaworld_loop_geometry_20261109','metaworld_input_servo_20261109']


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')


def snapshot(host):
    commands=[['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],
              ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],
              ['ps','-eo','pid,ppid,etime,args']]
    results=[]
    for c in commands:
        r=subprocess.run(c,capture_output=True,text=True,timeout=30)
        out=r.stdout if c[0]!='ps' else '\n'.join(x for x in r.stdout.splitlines() if '/efs/prescottsun/SPT-JEPA/' in x)
        results.append(dict(command=c,returncode=r.returncode,stdout=out,stderr=r.stderr))
    status={}
    for name in ROOTS:
        root=BASE/'runs'/name
        candidates=list(root.glob('results/summary.json'))+list(root.glob('CLOSED.json'))+list(root.glob('recovery-v2/results/summary.json'))+list(root.glob('recovery-v2/CLOSED.json'))
        status[name]={str(p.relative_to(root)):json.loads(p.read_text()) for p in candidates}
    write(REPORT/f'hour09-{host}.json',dict(utc=now(),host=host,resources=results,status=status,
        scope='Actualone-timesnapshot;NOTcontinuousutilization. No scheduler/LLMmonitor/newgoal.'))


def archive():
    paths=[]
    for name in ROOTS:
        for top in [BASE/'runs'/name,BASE/'reports'/name]:
            if not top.exists():continue
            for p in top.rglob('*'):
                if p.is_file() and p.suffix in ['.json','.log','.py','.md'] and p.stat().st_size<2_000_000:paths.append(p)
    paths=sorted(set(paths));index={str(p.relative_to(BASE)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
    manifest=REPORT/'hour09-SOURCE_MANIFEST.json';write(manifest,dict(utc=now(),files=index,
        scope='Metadata/source/logarchive. Fullmodel/optimizer/bank/tracefiles remainonEFS;actualtracehashvalidation is recordedby finitecollectors.'))
    target=REPORT/'hour09-evidence-v1.tar.gz'
    with tarfile.open(target,'x:gz') as t:
        for p in paths+[manifest]:t.add(p,arcname=str(p.relative_to(BASE)))
    seal=dict(utc=now(),path=str(target),sha256=sha(target),files=len(index)+1,bytes=target.stat().st_size)
    write(REPORT/'hour09-ARCHIVE.json',seal);print(json.dumps(seal))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['snapshot','archive']);p.add_argument('--host');a=p.parse_args()
    if a.mode=='snapshot':snapshot(a.host)
    else:archive()
