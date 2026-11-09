"""One-shot CPU-only snapshot, invoked directly by the active session."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_contact_geometry_20261109'


def read(path):return json.loads(path.read_text()) if path.exists() else None


def query(fields,kind='gpu'):
    option='--query-gpu=' if kind=='gpu' else '--query-compute-apps='
    raw=subprocess.check_output(['nvidia-smi',option+','.join(fields),'--format=csv,noheader,nounits'],text=True)
    return [dict(zip(fields,[s.strip() for s in row])) for row in csv.reader(raw.splitlines()) if row]


def run(host,output):
    p=read(ROOT/'plan.json');source=Path(p['source'])
    actual={str(x.relative_to(source)):hashlib.sha256(x.read_bytes()).hexdigest() for x in source.rglob('*.py')}
    jobs={j['name']:dict(status=read(Path(j['output'])/'status.json'),last=read(Path(j['output'])/'last.json'),
        preflight=read(Path(j['output'])/'PREFLIGHT.json'),audits={x.name:read(x) for x in Path(j['output']).glob('AUDIT-*.json')}) for j in p['jobs']}
    apps=query(['pid','gpu_uuid','used_gpu_memory','process_name'],'compute')
    for app in apps:
        try:argv=Path('/proc',app['pid'],'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,PermissionError):argv='exited-or-unreadable'
        app['argv']=argv;app['owned_belief_world']=str(ROOT) in argv
    d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=host,
        gpus=query(['index','uuid','utilization.gpu','memory.used','power.draw']),compute_apps=apps,
        source_hashes_match=actual==p['source_hashes'],training_gate=read(ROOT/'TRAINING_APPROVED.json'),
        four_gpu_audit=read(ROOT/'FOUR_GPU_AUDIT.json'),jobs=jobs,summary=read(ROOT/'results/summary.json'),
        phase_completions={x.name:read(x) for x in ROOT.glob('COLLECTED-*.json')},closed=read(ROOT/'CLOSED.json'),
        errors={str(x.relative_to(ROOT)):read(x) for x in (ROOT/'errors').glob('*.json')},
        model_updates=0,physics_steps=0,new_cuda_contexts=0,
        scope='Instant snapshot only. Four-GPU training repeats one PID across four GPU UUIDs; not four independent learners or constantutilization.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(dict(utc=d['utc'],host=host,source_hashes_match=d['source_hashes_match'],
        utilization=[x['utilization.gpu'] for x in d['gpus']],owned_cuda_GPU_records=sum(x['owned_belief_world'] for x in apps),
        unique_owned_CUDA_PIDs=len({x['pid'] for x in apps if x['owned_belief_world']}),
        steps={k:(v['status'] or {}).get('step') for k,v in jobs.items()},
        endpoints=(d['summary'] or {}).get('endpoints'),errors=len(d['errors']))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.host,a.output)
