"""Full20physical-case/task confirmation of the frozen10000-step local models."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil

from experiments.metaworld_contact_20261106.bridge_campaign import command, TASKS
from experiments.metaworld_contact_20261106.launch import PYTHON, environment, wait_gpu_release
from experiments.metaworld_contact_20261106.diagnose import save, digest


def run(a):
    plan = json.loads((a.root/'bridge-plan-v3.json').read_text())
    seed = 1 if a.host == 'dev13' else 2
    arm = 'local-mse' if a.gpu == 6 else 'local-flow'
    record_file = 'dense-cache-launch' if a.gpu == 6 else 'feedback-launch'
    owner = next(x for x in json.loads((a.root/f'{record_file}-{a.host}.json').read_text())['jobs'] if x['gpu'] == a.gpu)
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
    wait_gpu_release(a.gpu)
    recipe = next(r for r in plan['recipes'] if r['host'] == a.host and r['arm'] == arm)
    learner = Path(recipe['run'])
    assert json.loads((learner/'audit-10000.json').read_text())['passed']
    output = a.root/'confirmation10k'/f'{arm}-seed{seed}'
    output.parent.mkdir(exist_ok=True)
    cmd = [PYTHON, '-B', '-u', str(a.source/'experiments/metaworld_contact_20261106/evaluate.py'),
           '--mode', 'bridge', '--run', str(learner), '--step', '10000', '--parent', recipe['parent'],
           '--data', plan['data'], '--manifest', plan['manifest'], '--output', str(output),
           '--execute', '3', '--seeds']+list(map(str, list(range(42,52))+list(range(1042,1052))))+['--tasks']+TASKS
    command(cmd, output.with_suffix('.log'), environment(a.source, a.gpu), a.source, plan['deadline_unix'])
    summary = json.loads((output/'summary.json').read_text())
    assert summary['passed'] and not summary['frozen_parent_used']
    previous = json.loads((learner/'eval-step0010000/summary.json').read_text())
    assert summary['learner_sha256'] == previous['learner_sha256']
    common = []
    for old in previous['rows']:
        new = next(x for x in summary['rows'] if x['task'] == old['task'] and x['seed'] == old['seed'])
        assert new['case_sha256'] == old['case_sha256']
        assert digest(old['trace']) == old['trace_sha256'] and digest(new['trace']) == new['trace_sha256']
        first, second = np.load(old['trace'])['actions'], np.load(new['trace'])['actions']
        length = min(15, len(first), len(second))
        common.append(dict(task=old['task'], seed=old['seed'], first_block_exact=bool(np.array_equal(first[:length], second[:length])),
                           first_block_max_abs=float(np.abs(first[:length]-second[:length]).max()), old_success=old['success'], new_success=new['success']))
    save(output/'parent-removal-proof.json', dict(passed=True, unchanged_learner=True, frozen_parent_used=False, common=common,
         scope='Local controller uses only current/goal RGB;world-model features were unused in its original implementation. '
               'Expanded20cases are known existing cases,not untouchedbenchmark confirmation.No SGD or new objective.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--host',choices=['dev13','dev15'],required=True);p.add_argument('--gpu',type=int,choices=[6,7],required=True)
    run(p.parse_args())
