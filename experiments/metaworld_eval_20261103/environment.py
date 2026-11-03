"""LeRobot's pinned camera convention with explicit, replayable task bindings.

The current public LeRobot reference supplies camera/flip conventions. Its
revision is not asserted to be the historical generator of the offline data.
"""
import json
from pathlib import Path
import numpy as np


def configuration(source):
    return json.loads((Path(source) / 'src/lerobot/envs/metaworld_config.json').read_text())


def make_environment(task, seed, rand_vec=None):
    import metaworld
    from metaworld import _encode_task
    np.random.seed(seed)
    benchmark = metaworld.MT1(task, seed=42)
    native_class = benchmark.train_classes[task]
    env = native_class(render_mode='rgb_array', camera_name='corner2')
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    env.set_task(benchmark.train_tasks[0])
    if rand_vec is None:
        env._freeze_rand_vec = False
        env.seeded_rand_vec = True
        state, info = env.reset(seed=seed)
        rand_vec = env._last_rand_vec.copy()
    binding = _encode_task(task, dict(rand_vec=np.asarray(rand_vec).copy(), env_cls=native_class,
                                      partially_observable=False))
    env.set_task(binding)
    state, info = env.reset(seed=seed)
    return env, state.copy(), np.asarray(rand_vec).copy()


def pixels(env):
    frame = np.flip(env.render(), (0, 1)).copy()
    assert frame.shape == (480, 480, 3) and frame.dtype == np.uint8
    return frame


def expert(source, task):
    import metaworld.policies as policies
    return getattr(policies, configuration(source)['TASK_POLICY_MAPPING'][task])()
