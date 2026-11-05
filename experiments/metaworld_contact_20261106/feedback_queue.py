"""Finite matched feedback-frequency controls after frozen horizon controls."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time

import psutil

from experiments.metaworld_contact_20261106.bridge_campaign import command, TASKS
from experiments.metaworld_contact_20261106.launch import PYTHON, environment
from experiments.metaworld_contact_20261106.diagnose import save


def run(a):
    plan = json.loads((a.root/'bridge-plan-v3.json').read_text())
    seed = 1 if a.host == 'dev13' else 2
    parent = next(r['parent'] for r in plan['recipes'] if r['host'] == a.host)
    horizon = 12 if a.gpu == 6 else 24
    previous = a.root/'frozen'/f'cem-h{horizon}-seed{seed}'
    owner = next(r for r in json.loads((a.root/f'stage2-launch-{a.host}.json').read_text())['jobs'] if r['gpu'] == a.gpu)
    while True:
        try:
            process = psutil.Process(owner['pid'])
            owned = process.is_running() and process.status() != psutil.STATUS_ZOMBIE and str(a.root) in process.cmdline() and any(x.endswith('frozen_work.py') for x in process.cmdline())
        except psutil.NoSuchProcess:
            owned = False
        if not owned:
            break
        # Complete all24 manipulation endpoints. When none succeeds, eight
        # additional reaching episodes add little to this manipulation audit.
        progress = previous/'evaluation/progress.json'
        if a.gpu == 7 and progress.exists():
            rows = json.loads(progress.read_text())['rows']
            manip = [r for r in rows if r['task'] not in ['reach-v3', 'reach-wall-v3']]
            if len(manip) == 24 and not any(r['success'] for r in manip):
                assert os.getpgid(process.pid) == process.pid
                save(previous/'EARLY_STOP.json', dict(disposition='All24manipulationendpointscomplete0;remainingreachingnotexecuted/completed.',
                     completed_rows=rows, pending_not_negative=True, stopped_pid=process.pid,
                     utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                os.killpg(process.pid, signal.SIGTERM)
        assert time.time() < plan['deadline_unix']
        time.sleep(5)
    out = a.root/'feedback'/f'{a.host}-gpu{a.gpu}'; out.mkdir(parents=True, exist_ok=False)
    env = environment(a.source, a.gpu)
    common = [PYTHON, '-B', '-u', str(a.source/'experiments/metaworld_contact_20261106/evaluate.py'),
              '--parent', parent, '--data', plan['data'], '--manifest', plan['manifest'], '--execute', '1',
              '--seeds', '42', '1042', '--tasks']+TASKS
    if a.gpu == 6:
        requests = [('global-mse', 600), ('local-mse', 600), ('local-mse', 3000), ('local-mse', 10000)]
    else:
        requests = [('cem-control', 0), ('local-flow', 600), ('local-flow', 3000), ('local-flow', 10000)]
    for arm, step in requests:
        work = out/f'{arm}-step{step}'
        if arm == 'cem-control':
            cmd = common+['--mode', 'cem', '--horizon', '6', '--output', str(work)]
        else:
            learner = a.root/'bridge'/f'{arm}-seed{seed}'
            while not (learner/f'audit-{step}.json').exists():
                assert time.time() < plan['deadline_unix']
                if (learner/'execution-error.json').exists():
                    raise RuntimeError('Training stopped;do not wait indefinitely for unavailable checkpoint')
                time.sleep(5)
            assert json.loads((learner/f'audit-{step}.json').read_text())['passed']
            cmd = common+['--mode', 'bridge', '--run', str(learner), '--step', str(step), '--output', str(work)]
        command(cmd, out/f'{arm}-step{step}.log', env, a.source, plan['deadline_unix'])
    save(out/'complete.json', dict(passed=True, scope='Extra5-native-step feedback controls,not pooled with primary15-step results.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--host', choices=['dev13', 'dev15'], required=True); p.add_argument('--gpu', type=int, choices=[6, 7], required=True)
    run(p.parse_args())
