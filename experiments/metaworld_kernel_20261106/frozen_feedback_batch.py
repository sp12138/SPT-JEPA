"""A finite comparison on existing cases, with immutable trained actor/worlds."""
import argparse
import json
from pathlib import Path
import time

from experiments.metaworld_contact_20261106.launch import environment, PYTHON, wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save, digest


def work(a):
    plan = json.loads((a.root/'plan.json').read_text())
    r = next(v for v in plan['recipes'] if v['host'] == a.host and v['gpu'] == a.gpu)
    out = Path(r['output']); out.mkdir()
    source = Path(plan['source']); deadline = plan['deadline_unix']
    try:
        if r.get('wait_output'):
            while not (Path(r['wait_output'])/'complete.json').exists():
                assert not (Path(r['wait_output'])/'execution-error.json').exists() and time.time()<deadline
                time.sleep(5)
        while not (Path(r['run'])/'complete.json').exists():
            assert not (Path(r['run'])/'execution-error.json').exists()
            assert time.time() < deadline
            time.sleep(5)
        assert json.loads((Path(r['run'])/f"audit-{plan['step']}.json").read_text())['passed']
        assert digest(Path(plan['evaluator'])) == plan['evaluator_sha256']
        wait_gpu_release(a.gpu); env = environment(source, a.gpu)
        common = ['--run', r['run'], '--step', str(plan['step']), '--manifest', plan['manifest']]
        if r['mode'] == 'actual':
            base = [PYTHON, '-B', '-u', str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py')]+common+['--feedback-native', '1']
        else:
            base = [PYTHON, '-B', '-u', plan['evaluator']]+common+['--mode', r['mode'], '--world', r['world']]
            if 'policy_draw' in r: base += ['--policy-draw', r['policy_draw']]
            if 'render_state' in r: base += ['--render-state', r['render_state']]
        command(base+['--output', str(out/'entry20'), '--budget', '20', '--tasks', 'bin-picking-v3',
                      '--seeds', str(plan['seeds'][0])], out/'entry20.log', env, source, deadline)
        command(base+['--output', str(out/'pilot'), '--tasks']+plan['tasks']+
                ['--seeds']+list(map(str, plan['seeds'])), out/'pilot.log', env, source, deadline)
        proof = json.loads((out/'pilot/summary.json').read_text())
        assert proof['passed'] and proof['model_unchanged'] and len(proof['rows']) == plan['episodes']
        save(out/'complete.json', dict(passed=True, new_model_updates=0, episodes=plan['episodes']))
    except BaseException as error:
        save(out/'execution-error.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--host', required=True); p.add_argument('--gpu', type=int, required=True)
    work(p.parse_args())
