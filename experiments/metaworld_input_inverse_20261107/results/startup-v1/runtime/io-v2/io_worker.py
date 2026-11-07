"""Adopt native states at a stage boundary; change ONLY read-only data location."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.metaworld_input_inverse_20261107.campaign import (
    PYTHON, digest, environment, evaluate, execute, now, read, save)


def owned_alive(pid, marker):
    path = Path(f'/proc/{pid}/cmdline')
    try:
        return marker.encode() in path.read_bytes()
    except FileNotFoundError:
        return False


def work(root, source, helper, name, adoption):
    plan = read(root/'plan.json')
    recipe = next(x for x in plan['recipes'] if x['name'] == name)
    run = Path(recipe['run']); deadline = plan['deadline_unix']
    env = environment(source, recipe['gpu'])
    env['SPT_INPUT_INVERSE_NN_SOURCE'] = str(source)
    try:
        # Old native training/audit/physics children finish untouched.
        for pid in adoption['children']:
            while owned_alive(pid, str(root)):
                if time.time() >= deadline:
                    raise TimeoutError('Waiting for the previous native child')
                time.sleep(5)
        cache = Path('/tmp/prescottsun-mw-input-inverse-20261107-v1')
        while not (cache/'COMPLETE.json').exists():
            if time.time() >= deadline or (root/'STOP').exists():
                return
            time.sleep(5)
        proof = read(cache/'COMPLETE.json'); assert proof['passed']
        save(run/'DATA_IO_ADOPTION.json', dict(passed=True, old_parent=adoption['pid'],
            children_completed_without_interruption=adoption['children'], cache=str(cache),
            cache_proof_sha256=digest(cache/'COMPLETE.json'),
            nn_source=str(source), nn_sha256=digest(source/'as_jepa_lewm/executed_inverse.py'),
            trainer_sha256=digest(source/'as_jepa_lewm/train.py'),
            scope='Only equivalent read-only HDF5 source location; no completed SGD is repeated.'))
        from experiments.metaworld_eval_20261103.worker import free_gpu
        free_gpu(recipe['gpu'])
        assert digest(run/'config.json') == recipe['config_sha256']
        for step in [2, 1000, 5000]:
            if (root/'STOP').exists() or (run/'STOP').exists():
                return
            status = read(run/'status.json') if (run/'status.json').exists() else {}
            current = status.get('step', 0)
            if current > step:
                assert (run/f'archive-step{step:07d}/manifest.json').exists()
                continue
            if current < step:
                execute([PYTHON, '-B', '-u', str(helper/'io_stage.py'), 'train',
                         '--config', str(run/'config.json'), '--stop-step', str(step)]
                        + (['--resume'] if current else []), run/f'train-io-v2-{step}.log', env, source, deadline)
            else:
                assert current == step, (current, step)
            audit = run/f'audit-{step}.json'
            if not audit.exists() or not read(audit).get('passed'):
                output = run/f'audit-io-v2-{step}.json'
                execute([PYTHON, '-B', '-u', str(helper/'io_stage.py'), 'audit',
                         '--run', str(run), '--step', str(step), '--output', str(output)],
                        run/f'audit-io-v2-{step}.log', env, source, deadline, cap=1800)
                assert read(output)['passed']
            archive = run/f'archive-step{step:07d}'
            if not archive.exists():
                archive.mkdir(); os.link(run/'last.ckpt', archive/'native.ckpt')
                save(archive/'manifest.json', dict(step=step, sha256=digest(archive/'native.ckpt')))
            if step == 2 and not (run/'entry20/summary.json').exists():
                evaluate(root, source, run, step, Path(plan['screen_manifest']), run/'entry20', env, deadline, technical=True)
            if step == 5000 and not (run/'screen5000/summary.json').exists():
                evaluate(root, source, run, step, Path(plan['screen_manifest']), run/'screen5000', env, deadline)
            save(run/'progress.json', dict(step=step, state='stage_complete', updated_utc=now(), io_helper=str(helper)))
        save(run/'screen-complete.json', dict(passed=True, step=5000, ended_utc=now()))
        while not (root/'selection.json').exists():
            if time.time() >= deadline or (root/'STOP').exists():
                return
            time.sleep(15)
        selection = read(root/'selection.json')
        if recipe['arm'] not in selection['selected']:
            save(run/'HOLD', dict(reason='Not selected for long training', selection=str(root/'selection.json')))
            return
        for step in [20000, 41200]:
            if (root/'STOP').exists() or (run/'STOP').exists():
                return
            execute([PYTHON, '-B', '-u', str(helper/'io_stage.py'), 'train',
                     '--config', str(run/'config.json'), '--stop-step', str(step), '--resume'],
                    run/f'train-io-v2-{step}.log', env, source, deadline)
            output = run/f'audit-io-v2-{step}.json'
            execute([PYTHON, '-B', '-u', str(helper/'io_stage.py'), 'audit',
                     '--run', str(run), '--step', str(step), '--output', str(output)],
                    run/f'audit-io-v2-{step}.log', env, source, deadline)
            assert read(output)['passed']
            evaluate(root, source, run, step, Path(plan['full_manifest']), run/f'eval-{step}', env, deadline)
            save(run/'progress.json', dict(step=step, state='stage_complete', updated_utc=now(), io_helper=str(helper)))
        save(run/'complete.json', dict(passed=True, ended_utc=now(), step=41200))
    except BaseException as error:
        save(run/'execution-error-io-v2.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=now()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['root', 'source', 'helper', 'adoption']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--name', required=True)
    a = p.parse_args()
    work(a.root, a.source, a.helper, a.name, read(a.adoption)[a.name])
