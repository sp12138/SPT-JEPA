"""Goal-free TRAIN interventions along the nontranslation actuator inputs.

The collection-only servo executes the requested rotation; native Reach ignores
that input. Evaluation keeps its native controller. No model is instantiated.
Only natural RGB and the actually issued current commands enter the corpus.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.evaluate import install_public, reset
from experiments.robocasa_refine.intervene import pixels
from experiments.robocasa_stable_scaling.rotation_facts import step_rotation
from experiments.robocasa_stable_scaling.stream import make_env


def collect(root, lane, lanes, limit=None):
    import cv2
    cv2.setNumThreads(1)
    plan = json.loads((root/'plan.json').read_text())
    assert 0 <= lane < lanes and plan['episodes'] % lanes == 0
    install_public(root)
    completed = 0
    for index in range(lane, plan['episodes'], lanes):
        ident = f'orientation-{index:06d}'
        directory = root/'records'/ident
        ready = directory/'ready.json'
        if ready.exists():
            proof = json.loads(ready.read_text())
            assert proof['passed'] and sha(Path(proof['rgb'])) == proof['rgb_sha256']
            assert sha(Path(proof['observations'])) == proof['observations_sha256']
            continue
        assert not directory.exists(), 'Preserve incomplete data before explicit retry'
        pair = plan['kitchens'][index % len(plan['kitchens'])]
        seed = plan['seed']+index
        rng = np.random.default_rng(seed)
        direction = rng.normal(size=(8, 3))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        # Same normalized input range as the ordinary translation corpus.
        radius = np.exp(rng.uniform(np.log(.01), 0., size=(8, 1)))
        forward = direction*radius*.5
        commands = np.zeros((16, 7), np.float32)
        commands[:8, 3:6] = forward
        commands[8:, 3:6] = -forward[::-1]
        began = time.time()
        env = make_env(pair, seed, 'reach')
        try:
            observation = reset(env, 'reach', seed, 'init')
            images = [pixels(observation)]
            for command in commands:
                observation = step_rotation(env, command)
                images.append(pixels(observation))
            images = np.stack(images)
        finally:
            env.close()
        assert images.shape == (17, 2, 3, 224, 224) and images.dtype == np.uint8
        assert np.all(commands[:, :3] == 0)
        render = np.stack([cv2.resize(x.transpose(1, 2, 0), (64, 64),
            interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
            for x in images.reshape(-1, 3, 224, 224)]).reshape(17, 2, 3, 64, 64)
        directory.mkdir(parents=True)
        rgb = directory/'rgb.npz'
        observations = directory/'observations.npz'
        np.savez_compressed(rgb, pixels=images)
        np.savez_compressed(observations, render=render, commands=commands)
        save(ready, dict(passed=True, id=ident, split='train', kitchen=pair,
            seed=seed, frames=17, transitions=16, original=False,
            rgb=str(rgb), rgb_sha256=sha(rgb), observations=str(observations),
            observations_sha256=sha(observations),
            local_observations=str(Path(plan['local_cache'])/'observations'/(ident+'.npz')),
            input_fields=['RGB', 'actually_issued_current_commands'],
            current_XYZ_inputs_zero=True, collection_controller_differs_from_native_Reach=True,
            normalized_angular_norm_range=[.01, 1.], angular_tolerance_rad=.001,
            no_model_instantiated=True, no_goal_reward_state_or_expert_in_experience=True,
            not_assumed_physically_closed=True, seconds=time.time()-began))
        completed += 1
        print(json.dumps(dict(id=ident, complete=True, seconds=time.time()-began)), flush=True)
        if limit is not None and completed >= limit:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--lane', type=int, required=True)
    parser.add_argument('--lanes', type=int, default=16)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    collect(args.root, args.lane, args.lanes, args.limit)
