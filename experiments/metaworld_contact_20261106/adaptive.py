"""Evidence-led complements: one fused state,noise weighting,horizon marginal."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from experiments.metaworld_contact_20261106.diagnose import save, digest
from experiments.metaworld_contact_20261106.launch import PYTHON, environment


def prepare(root, original, source):
    root.mkdir(parents=True, exist_ok=False)
    previous = json.loads((original/'bridge-plan-v3.json').read_text())
    recipes = []
    for host, seed in [('dev13', 1), ('dev15', 2)]:
        old = {r['arm']:r for r in previous['recipes'] if r['host'] == host}
        v3 = json.loads((original/f'train-recovery-v3-{host}.json').read_text())['jobs']
        probes = json.loads((original/f'probe-launch-{host}.json').read_text())['jobs']
        specs = [(0, 'high-noise-flow', 'local-flow', {'noise_odds':4.}, True),
                 (1, 'marginal-global', 'global-mse', {'condition_horizon':'none'}, True),
                 (4, 'hybrid-mse', 'local-mse', {'visual_source':'hybrid'}, False),
                 (5, 'hybrid-flow', 'local-flow', {'visual_source':'hybrid'}, False)]
        for gpu, arm, base, intervention, fork in specs:
            parent = Path(old[base]['run'])
            assert json.loads((parent/'audit-3000.json').read_text())['passed']
            config = json.loads((parent/'config.json').read_text()); config = copy.deepcopy(config)
            name = f'{arm}-seed{seed}'; work = root/'bridge'/name; work.mkdir(parents=True)
            config['output'] = str(work); config['model'].update(intervention)
            stages = [3002, 6000, 10000] if fork else [2, 600, 3000, 10000]
            config['retain_steps'] = sorted(set(stages+[3000] if fork else stages))
            save(work/'config.json', config)
            owner = next(r for r in (v3 if gpu < 4 else probes) if r['gpu'] == gpu)
            recipe = dict(name=name, arm=arm, seed=seed, host=host, gpu=gpu, run=str(work),
                          parent=old[base]['parent'], config_sha256=digest(work/'config.json'),
                          stages=stages, technical_step=stages[0], wait_owner=owner,
                          inherited_from=str(parent) if fork else None, intervention=intervention)
            if fork:
                archive = parent/'archive-step0003000'
                expected = json.loads((archive/'manifest.json').read_text())['native_sha256']
                assert digest(archive/'native.ckpt') == expected
                os.link(archive/'native.ckpt', work/'last.ckpt')
                shutil.copytree(parent/'initial-model', work/'initial-model', copy_function=os.link)
                portable = work/'model-step0003000'; portable.mkdir()
                os.link(parent/'model-step0003000/weights.pt', portable/'weights.pt')
                save(portable/'config.json', dict(_target_='as_jepa_lewm.temporal_bridge.TemporalBridge', **config['model']))
                save(work/'FORK.json', dict(parent=str(parent), entry_step=3000, entry_native_sha256=expected,
                     intervention=intervention, full_optimizer_scheduler_rng_cursor=True,
                     scope='Actual own-data3000-step full state;no completedSGD repeats.New objective applies only afterentry.'))
                recipe['entry_step'] = 3000
            else:
                save(work/'ORIGIN.json', dict(random_cnn_and_readout=True, frozen_features=old[base]['parent'],
                     source='Ownsame-data20epochTFonly,noexternalteacher;fullparentcostmustcount.'))
            recipes.append(recipe)
    plan = dict(previous, source=str(source), recipes=recipes, stages=[2,600,3000,10000],
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='New finite evidence-led tests within same user request. '
                      'Hybrid is one shared model with local patch memory plus two ownTFglobal tokens,no task-ID/gating. '
                      'Global/local complements observed at3000;oracle-to-estimated horizon factual error rises46%global,11-24%local. '
                      'Flow goal-conditioning benefits concentrated at highnoise;odds4 shared for all tasks. '
                      'Forks inherit actual3000 SGD and countcumulative10000;hybrids fresh auxiliaryrandommodels with ownTFfeatures. '
                      'No novelty/optimal-policy claim;newsimulator training data absent;samplelevel retained.')
    save(root/'bridge-plan-v3.json', plan)
    save(root/'bridge-plan.json', plan)
    save(root/'active-bridge-plan.json', dict(path=str(root/'bridge-plan-v3.json')))


def work(root, source, host, gpu):
    plan = json.loads((root/'bridge-plan-v3.json').read_text())
    recipe = next(r for r in plan['recipes'] if r['host'] == host and r['gpu'] == gpu)
    owner = recipe['wait_owner']
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
    from experiments.metaworld_eval_20261103.worker import free_gpu
    free_gpu(gpu)
    from experiments.metaworld_contact_20261106.bridge_campaign import work as train
    train(root, source, host, gpu)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--original',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.mode=='prepare': prepare(a.root,a.original,a.source)
    else: work(a.root,a.source,a.host,a.gpu)
