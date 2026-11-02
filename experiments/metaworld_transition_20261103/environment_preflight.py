"""Bounded interface check, using unchanged pinned public wrappers; never trains a model."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS


def main(source, output, gpu):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip())
    assert used <= 1, 'The designated, previously released baseline slot is occupied; do not render.'
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu))
    sys.path.insert(0, str(source))
    import numpy as np
    from PIL import Image
    from metaworld import _encode_task
    from evals.simu_env_planning.envs.metaworld import make_env, task_name_to_policy
    rows = []
    for task in ['mw-reach', 'mw-push', 'mw-bin-picking']:
        cfg = NS(meta=NS(seed=17), task_specification=NS(task=task, img_size=224, max_episode_steps=100,
            env=NS(freeze_rand_vec=False)))
        env = make_env(cfg)
        raw = env.unwrapped
        task_data = _encode_task('mw-reach', dict(rand_vec=raw._last_rand_vec.copy(),
            env_cls=type(raw).__bases__[0], partially_observable=False))
        raw.set_task(task_data)
        obs, info = env.reset(seed=17)
        initial = obs.copy()
        assert obs.shape == (39,) and env.action_space.shape == (4,)
        policy = task_name_to_policy(task)()
        actions, states, successes, frames = [], [obs.copy()], [], []
        try:
            for step in range(100):
                if step in [0, 1, 5, 20, 50, 99]:
                    frame = env.render()
                    assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8
                    assert frame.max() > frame.min()
                    Image.fromarray(frame).save(output / f'{task}-{step:03d}.png')
                    frames.append(dict(step=step, sha256=hashlib.sha256(frame.tobytes()).hexdigest()))
                action = np.clip(policy.get_action(obs), env.action_space.low, env.action_space.high).astype(np.float32)
                assert action.shape == (4,) and np.isfinite(action).all()
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(obs).all() and np.isfinite(reward)
                actions.append(action); states.append(obs.copy()); successes.append(bool(info['success']))
                if terminated or truncated:
                    assert step == 99
                    break
            same_reset, _ = env.reset(seed=17)
        finally:env.close()
        replay = make_env(cfg)
        replay.unwrapped.set_task(task_data)
        try:recreated, _ = replay.reset(seed=17)
        finally:replay.close()
        max_diff = float(np.max(np.abs(initial-recreated)))
        np.savez_compressed(output / (task + '-sanity.npz'), states=np.stack(states), actions=np.stack(actions),
            success=np.array(successes), initial=initial, recreated=recreated, same_instance_reset=same_reset)
        assert np.allclose(initial, recreated, atol=1.e-6, rtol=0), ('Goal/start reconstruction differs even with the public task binding.', max_diff)
        rows.append(dict(task=task, state_shape=[39], action_shape=[4], frames=frames, steps=len(actions),
            expert_any_success=any(successes), expert_last_success=successes[-1],
            same_instance_reset_exact=bool(np.array_equal(initial, same_reset)),
            recreated_environment_reset_exact=bool(np.array_equal(initial,recreated)), recreated_max_abs_diff=max_diff,
            public_task_rand_vector_shared=True,
            scope='Scripted-policy interface sanity only; not learned-world-model performance or official training data.'))
    names = ['metaworld', 'mujoco', 'gym', 'gymnasium', 'numpy', 'pillow']
    files = ['evals/simu_env_planning/envs/metaworld.py', 'evals/simu_env_planning/envs/wrappers/time_limit.py']
    result = dict(passed=True, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version, packages={name: importlib.metadata.version(name) for name in names},
        source=str(source), wrapper_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
        gpu_for_rendering_only=gpu, rows=rows, model_training_updates=0, model_forwards=0,
        official_dataset_downloaded=False,
        scope='Nativev3.1.1+MuJoCo3.3.0 isolatedPython3.12 sanity; official fullJEPA stack requiresPython3.10 and is not claimed installed. No running experiment changed.')
    (output/'proof.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, output=str(output), tasks=[dict(task=r['task'], reset_exact=r['recreated_environment_reset_exact'], expert_success=r['expert_any_success']) for r in rows])))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source', required=True);p.add_argument('--output', required=True);p.add_argument('--gpu', type=int, default=1)
    a=p.parse_args();main(a.source, a.output, a.gpu)
