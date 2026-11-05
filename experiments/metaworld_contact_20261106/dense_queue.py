"""Fill released slots with complete contextual patch features,without SGD."""
import argparse
import json
from pathlib import Path
import shutil
import time

import psutil

from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_contact_20261106.launch import PYTHON, environment
from experiments.metaworld_contact_20261106.diagnose import save


def run(a):
    plan = json.loads((a.root/'bridge-plan-v3.json').read_text())
    own = next(r for r in json.loads((a.root/f'dense-cache-launch-{a.host}.json').read_text())['jobs'] if r['gpu'] == a.gpu)
    owner = own['wait_owner']
    while True:
        try:
            p = psutil.Process(owner['pid'])
            owned = p.is_running() and p.status() != psutil.STATUS_ZOMBIE and p.cmdline() == owner['command']
        except psutil.NoSuchProcess:
            owned = False
        if not owned:
            break
        assert time.time() < plan['deadline_unix']
        time.sleep(5)
    from experiments.metaworld_contact_20261106.launch import wait_gpu_release
    wait_gpu_release(a.gpu)
    output = Path(own['output'])
    env = environment(a.source, a.gpu)
    command([PYTHON, '-B', '-u', str(a.source/'experiments/metaworld_contact_20261106/cache.py'),
             '--data', plan['data'], '--checkpoint', own['checkpoint'], '--output', str(output),
             '--shard', str(own['shard']), '--shards', '4', '--dense'],
            a.root/f'dense-cache-{a.host}-gpu{a.gpu}.log', env, a.source, plan['deadline_unix'])
    result = json.loads((output/'manifest.json').read_text()); assert result['passed'] and result['model_unchanged']
    destination = a.root/'dense-cache-manifests'; destination.mkdir(exist_ok=True)
    shutil.copyfile(output/'manifest.json', destination/f"{a.host}-shard{own['shard']}.json")
    save(a.root/f'dense-cache-complete-{a.host}-gpu{a.gpu}.json', dict(passed=True, output=str(output),
         scope='Exact fullpatches from ownscratchTFencoder,all50tasks,train/testsplitretained,no SGD.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--gpu',type=int,required=True);run(p.parse_args())
