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
    # MetaWorld3.1.1 explicitly ignores reset(seed=...). Its native seed()
    # controls the Generator used for task-vector discovery.
    env.seed(seed)
    # Keep reset/settling history identical for discovery and bound replay.
    # The 39D observation alone does not describe every rendered robot joint.
    env._freeze_rand_vec = False
    env.seeded_rand_vec = True
    state, info = env.reset(seed=seed)
    if rand_vec is None:
        rand_vec = env._last_rand_vec.copy()
    binding = _encode_task(task, dict(rand_vec=np.asarray(rand_vec).copy(), env_cls=native_class,
                                      partially_observable=False))
    env.set_task(binding)
    state, info = env.reset(seed=seed)
    return env, state.copy(), np.asarray(rand_vec).copy()


def pixels(env):
    # CUDA model initialization can expose a transient first OpenGL frame on
    # some slots. One uniform no-action warmup matches the cached reference;
    # retain the strict replay gate rather than accepting a different image.
    if not getattr(env, '_spt_render_primed', False):
        env.render()
        env._spt_render_primed = True
    frame = np.flip(env.render(), (0, 1)).copy()
    assert frame.shape == (480, 480, 3) and frame.dtype == np.uint8
    return frame


def close_environment(env):
    # Gymnasium1.4's OffScreenViewer.free releases the GL context but leaves
    # MjrContext alive. Release its GPU resources in its OWN current context,
    # before a later viewer can reuse those GL resource IDs.
    renderer = env.mujoco_renderer
    for viewer in list(renderer._viewers.values()):
        viewer.opengl_context.make_current()
        viewer.con.free()
    env.close()
    renderer._viewers.clear()
    renderer.viewer = None


def physics_state(env):
    return {name: np.asarray(getattr(env.data, name)).copy()
            for name in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat', 'ctrl']}


def rgb_replay_check(expected, actual):
    # Native replays had identical physical/model arrays and a handful of
    # RGB pixels differing by one or two 8-bit levels. Record renderer precision
    # separately from physical equality; use the cached initial image in MPC.
    difference = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
    changed = np.any(difference != 0, axis=-1)
    return dict(exact=bool(np.array_equal(expected, actual)),
                max_abs=int(difference.max()), changed_pixels=int(changed.sum()),
                changed_fraction=float(changed.mean()),
                mse=float(np.square(difference.astype(np.float64)).mean()),
                passed=bool(np.square(difference.astype(np.float64)).mean() <= .001
                            and changed.mean() <= .001))


def expert(source, task):
    import metaworld.policies as policies
    return getattr(policies, configuration(source)['TASK_POLICY_MAPPING'][task])()
