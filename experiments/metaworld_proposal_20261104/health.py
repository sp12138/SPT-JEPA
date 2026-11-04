"""One read-only startup snapshot;no scheduled/LLM monitoring."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

import psutil


def main(root, host):
    def read(p):return json.loads(p.read_text())
    plan = read(root/'plan.json')
    owners = read(root/f'launch-{host}.json')['jobs']
    rows = []
    for r in plan['recipes']:
        if r['host'] != host:continue
        work = Path(r['work'])
        owner = next(o for o in owners if o['gpu']==r['gpu'])
        pre = work/'preflight/output/metrics.json'
        d = read(pre) if pre.exists() else {}
        try:
            p = psutil.Process(owner['pid'])
            alive = p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
            children = p.children(recursive=True)
        except psutil.NoSuchProcess:
            alive, children = False, []
        actual = []
        for child in children:
            try:
                command = child.cmdline()
                if any(x.endswith('/evaluate.py') for x in command):
                    env = child.environ()
                    actual.append(dict(pid=child.pid, command=command,
                        CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES'), PYTHONPATH=env.get('PYTHONPATH')))
            except psutil.NoSuchProcess:pass
        checks = dict(actual_preflight=d.get('passed',False),
            NN_unchanged=bool(d) and d.get('model_state_before')==d.get('model_state_after'),
            weights=d.get('weights_sha256')==r['weights_sha256'],
            model_updates_zero=d.get('model_updates')==0,
            actual20steps=bool(d) and len(d['episodes'])==1 and d['budget']==20,
            actual_RGB=bool(d) and all(e['initial_rgb_replay']['passed'] for e in d['episodes']),
            no_effective_failure=not (work/'failed.json').exists(),
            parent_alive_or_complete=alive or (work/'complete.json').exists(),
            physical_audit=(work/'preflight/output/physical-audit.json').exists())
        if actual:
            checks['actual_GPU_and_source']=all(x['CUDA_VISIBLE_DEVICES']==str(r['gpu']) and plan['source'] in x['PYTHONPATH'].split(':') for x in actual)
        rows.append(dict(name=r['name'],gpu=r['gpu'],passed=all(checks.values()),checks=checks,
                         completed_endpoints=len(list(work.glob('*-v3-*/complete.json'))),actual_children=actual,
                         failed=read(work/'failed.json') if (work/'failed.json').exists() else None))
    report = dict(host=host,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  passed=len(rows)==8 and all(r['passed'] for r in rows),rows=rows,
                  gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
                  scope='Actual frozen-model startup/physics gates only;not highperformance evidence or constantGPUutilization.')
    output = root/f'health-{host}.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],host=host,
        rows=[dict(name=r['name'],passed=r['passed'],endpoints=r['completed_endpoints'],
                   failing=[k for k,v in r['checks'].items() if not v],error=r['failed']) for r in rows],gpu=report['gpu_state'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',choices=['dev13','dev15'],required=True)
    a=p.parse_args();main(Path(a.root),a.host)
