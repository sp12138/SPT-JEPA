"""One-shot, CPU-only ownership/resource snapshot, never a recurring monitor."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import datetime
import hashlib

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_endpoint_density_20261109'


def read(path):return json.loads(path.read_text()) if path.exists() else None


def query(fields,kind='gpu'):
    option='--query-gpu=' if kind=='gpu' else '--query-compute-apps='
    raw=subprocess.check_output(['nvidia-smi',option+','.join(fields),'--format=csv,noheader,nounits'],text=True)
    return [dict(zip(fields,[s.strip() for s in row])) for row in csv.reader(raw.splitlines()) if row]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    gpus=query(['index','uuid','utilization.gpu','memory.used','power.draw'])
    apps=query(['pid','gpu_uuid','used_gpu_memory','process_name'],'compute');classification={}
    for app in apps:
        try:argv=Path('/proc',app['pid'],'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,PermissionError):argv='exited-or-unreadable'
        app['argv']=argv
        if '/composition-v1/evaluation/' in argv:kind='composition'
        elif '/metaworld_transition_only_20261109/' in argv:kind='primary_factual'
        elif 'metaworld_world_vs_planner_20261109' in argv:kind='privileged_oracle_audit'
        elif str(ROOT) in argv:kind='endpoint_other'
        else:kind='other_or_exited'
        app['classification']=kind;classification[kind]=classification.get(kind,0)+1
    roots={}
    for name,path in [('primary',BASE/'runs/metaworld_transition_only_20261109'),('endpoint',ROOT),('expanded',ROOT/'expanded-v1'),('composition',ROOT/'composition-v1')]:
        plan=read(path/'plan.json');source_match=None
        if plan and 'source_hashes' in plan:
            source=Path(plan['source']);actual={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
            source_match=actual==plan['source_hashes']
        roots[name]=dict(summary=read(path/'results/summary.json'),closed=read(path/'CLOSED.json'),source_hashes_match=source_match,
            effective_errors={str(p.relative_to(path)):read(p) for p in (path/'errors').glob('*.json')})
    value=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=a.host,gpus=gpus,compute_apps=apps,
        classifications=classification,roots=roots,model_updates=0,physics_steps=0,new_gpu_contexts=0,
        scope='Instant physicalGPU/resource snapshot;old/new CUDA contexts may share a GPU. No continuousutilization claim.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(utc=value['utc'],host=a.host,classifications=classification,utilization=[x['utilization.gpu'] for x in gpus],
        primary_endpoints=(roots['primary']['summary'] or {}).get('evaluation_endpoints_complete'),
        composition_endpoints=(roots['composition']['summary'] or {}).get('endpoints'),source_matches={k:v['source_hashes_match'] for k,v in roots.items()})),flush=True)
