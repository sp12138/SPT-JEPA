"""Launch one verified new immutable native workflow on the specified host."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--host', choices=['dev2', 'dev13', 'dev15'], required=True)
    a = p.parse_args()
    planpath = Path(a.plan)
    plan = json.loads(planpath.read_text())
    root = Path(plan['root'])
    jobs = [j for j in plan['jobs'] if j['host'] == a.host]
    assert jobs and not (root/'HOLD_DISPATCH').exists()
    recordpath = root/f'launched-{a.host}.json'
    assert not recordpath.exists(), 'Preserve launch evidence; investigate before restart.'
    sources = {json.dumps(j['source'], sort_keys=True) for j in jobs}
    assert len(sources) == 1
    source = jobs[0]['source']
    for gpu in sorted({j['gpu'] for j in jobs}):
        apps = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        assert not apps, (a.host, gpu, apps)
    env = {**os.environ, **plan.get('environment', {}), 'PYTHONPATH': source['pythonpath'],
           'PYTHONDONTWRITEBYTECODE': '1'}
    py = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
    command = [py, '-B', '-u', '-m', 'as_jepa_tools.workflow', '--plan', str(planpath), '--host', a.host]
    with (root/f'worker-{a.host}.log').open('ab') as log:
        proc = subprocess.Popen(command, cwd=source['source'], env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(host=a.host, pid=proc.pid, launched_at=time.time(), command=command,
        source_commit=source['source_commit'], slots=sorted({j['gpu'] for j in jobs}), jobs=len(jobs))
    recordpath.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
