"""Frozen, real-physics attribution of MetaWorld manipulation failures.

Experts and physical states are DIAGNOSTIC controls, never candidate inputs or
training data. Original standard-CEM actions are replayed unchanged. Every
prediction is compared with the actual image reached by its recorded actions.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.metaworld_eval_20261103.environment import (
    close_environment, expert, make_environment, physics_state, pixels, rgb_replay_check,
)


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def render(env):
    before = physics_state(env)
    clock = env.data.time
    rng = json.dumps(env.np_random.bit_generator.state, sort_keys=True)
    frame = pixels(env)
    assert clock == env.data.time
    assert rng == json.dumps(env.np_random.bit_generator.state, sort_keys=True)
    assert all(np.array_equal(v, physics_state(env)[k]) for k, v in before.items())
    return frame


def diagnostic_state(env):
    obj = np.asarray(env._get_pos_objects()).reshape(-1, 3)
    tcp = np.asarray(env.tcp_center)
    return dict(object_xyz=obj.tolist(), tcp_xyz=tcp.tolist(),
                gripper_object_distance=float(np.linalg.norm(obj-tcp, axis=-1).min()),
                object_target_distance=float(np.linalg.norm(obj-np.asarray(env._target_pos), axis=-1).min()))


def restore(env, state):
    import mujoco
    for name, value in state.items():
        getattr(env.data, name)[:] = value
    mujoco.mj_forward(env.model, env.data)


def factorial(env, start, goal):
    """Render robot/object joint interventions. No simulated policy uses these.

    The first nine coordinates are the native Sawyer arm/finger joints. Assert
    their names rather than assuming every model has a particular object joint.
    Joint interventions may be physically unreachable; they identify visual
    sensitivity only, never successful planning or dynamics performance.
    """
    names = [env.model.joint(i).name for i in range(env.model.njnt)]
    robot = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4',
             'right_j5', 'right_j6', 'r_close', 'l_close']
    assert names[:9] == robot, names
    assert list(env.model.jnt_qposadr[:9]) == list(range(9))
    original = physics_state(env)
    frames = []
    try:
        for arm_goal, object_goal in [(False, False), (True, False), (False, True), (True, True)]:
            state = {k:v.copy() for k, v in (goal if arm_goal else start).items()}
            state['qpos'][9:] = (goal if object_goal else start)['qpos'][9:]
            state['qvel'][:] = 0
            restore(env, state)
            frames.append(render(env))
    finally:
        restore(env, original)
    return frames, names


def realize(case, cached, policy_source, actions=None, with_factorial=False):
    env, state, _ = make_environment(case['task'], case['seed'], cached['rand_vec'])
    assert np.array_equal(state, cached['initial_state'])
    assert all(np.array_equal(v, cached['initial_physics_'+k]) for k, v in physics_state(env).items())
    frames, controls, success, diagnostics = [cached['initial'].copy()], [], [], [diagnostic_state(env)]
    gate = rgb_replay_check(cached['initial'], render(env))
    assert gate['passed'], gate
    initial = physics_state(env)
    factorial_frames, joint_names = None, None
    policy = expert(policy_source, case['task']) if actions is None else None
    try:
        limit = 500 if actions is None else len(actions)
        for step in range(limit):
            action = np.clip(policy.get_action(state) if policy is not None else actions[step], -1, 1).astype(np.float32)
            state, _, terminated, truncated, info = env.step(action)
            controls.append(action.copy()); success.append(bool(info['success']))
            diagnostics.append(diagnostic_state(env))
            stop = bool(terminated or truncated or (policy is not None and success[-1]))
            if (step+1) % 5 == 0:
                frames.append(render(env))
            if stop:
                break
        final_frame = render(env)
        if with_factorial:
            factorial_frames, joint_names = factorial(env, initial, physics_state(env))
    finally:
        close_environment(env)
    return dict(frames=np.stack(frames), actions=np.asarray(controls), success=np.asarray(success),
                diagnostics=diagnostics, initial_rgb=gate, final_frame=final_frame,
                factorial_frames=factorial_frames, joint_names=joint_names)


def encode(model, frames):
    import torch
    from experiments.metaworld_eval_20261103.evaluate import resize_tensor
    return torch.cat([model.encode(dict(pixels=model.normalize(resize_tensor(frames[i:i+24])[:, None].cuda())))['emb'][:, 0].float()
                      for i in range(0, len(frames), 24)])


def forecast(model, frames, actions, goal, mean, std):
    import torch
    from experiments.metaworld_eval_20261103.evaluate import resize_tensor
    rows = []
    count = len(actions)//5
    normalized = (actions[:count*5].reshape(count, 20)-mean)/std
    horizons = [h for h in [6, 12, 24] if h <= count]
    if count and count not in horizons:
        horizons.append(count)
    # Initial prefixes isolate the existing30-step horizon from longer plans.
    for h in sorted(set(horizons)):
        info = dict(pixels=model.normalize(resize_tensor(frames[:1])[None, None].cuda()),
                    goal=goal[None, None, None], action=torch.zeros(1, 1, 1, 20, device='cuda'),
                    action_history=torch.empty(1, 1, 0, 20, device='cuda'))
        plan = torch.from_numpy(normalized[:h]).cuda()[None, None]
        predicted = model.rollout(info, plan)['predicted_emb'][0, 0, -h:].float()
        observed = encode(model, frames[1:h+1])
        rows.append(dict(horizon_macro=h, native_steps=5*h,
                         predicted_goal_mse=float((predicted[-1]-goal).square().mean()),
                         realized_goal_mse=float((observed[-1]-goal).square().mean()),
                         forecast_mse=float((predicted-observed).square().mean()),
                         copy_mse=float((observed-encode(model, frames[:1])).square().mean())))
    # Contact/interaction segments use the same six-step prediction context.
    segment_rows = []
    z = encode(model, frames)
    for start in range(0, max(0, count-5), 3):
        context_start = max(0, start-1)
        context = frames[context_start:start+1]
        info = dict(pixels=model.normalize(resize_tensor(context)[None, None].cuda()),
                    goal=goal[None, None, None], action=torch.zeros(1, 1, len(context), 20, device='cuda'),
                    action_history=torch.from_numpy(normalized[context_start:start])[None, None].cuda())
        predicted = model.rollout(info, torch.from_numpy(normalized[start:start+6])[None, None].cuda())['predicted_emb'][0, 0, -6:].float()
        target = z[start+1:start+7]
        segment_rows.append(dict(start_native=5*start, predicted_goal_mse=float((predicted[-1]-goal).square().mean()),
                                 realized_goal_mse=float((target[-1]-goal).square().mean()),
                                 forecast_mse=float((predicted-target).square().mean()),
                                 copy_mse=float((target-z[start]).square().mean())))
    return dict(prefixes=rows, segments=segment_rows)


def run(args):
    import torch
    import stable_worldmodel as swm
    from experiments.metaworld_eval_20261103.evaluate import model_sha
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False),
                   device=torch.device('cuda'), parallel_state=None)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.plan).read_text())
    recipe = next(r for r in plan['recipes'] if r['name'] == args.name)
    model = swm.wm.utils.load_pretrained(recipe['checkpoint']).cuda().eval().requires_grad_(False)
    before = model_sha(model)
    card = json.loads((Path(plan['data'])/'dataset_card.json').read_text())
    mean, std = np.asarray(card['action_mean'], np.float32), np.asarray(card['action_std'], np.float32)
    cases = json.loads(Path(plan['manifest']).read_text())['rows']
    summary = json.loads(Path(plan['old_summary']).read_text())
    rows = []
    for case in cases:
        if case['task'] != recipe['task'] or case['seed'] not in plan['case_seeds']:
            continue
        assert digest(case['path']) == case['sha256']
        cached = np.load(case['path'])
        obs = [obs for group in summary['groups'] if group['step'] == 16480 and group['arm'] == recipe['arm']
               and group['task'] == case['task'] for obs in group['observations']
               if obs['train_seed'] == recipe['parent_seed'] and obs['seed'] == case['seed']]
        assert len(obs) == 1, (recipe, case, len(obs))
        obs = obs[0]
        assert digest(obs['trace']) == obs['trace_sha256']
        original = np.load(obs['trace'])
        paths = {'expert_diagnostic':realize(case, cached, plan['policy_source'], with_factorial=True),
                 'recorded_cem':realize(case, cached, plan['policy_source'], original['actions'])}
        assert np.array_equal(paths['recorded_cem']['success'], original['successes'])
        record = dict(task=case['task'], seed=case['seed'], case_sha256=case['sha256'],
                      original_trace_sha256=obs['trace_sha256'], paths={})
        with torch.inference_mode():
            goal = encode(model, [cached['goal']])[0]
            for name, value in paths.items():
                directory = output/f"{case['seed']}-{name}"; directory.mkdir()
                z = encode(model, value['frames'])
                costs = (z-goal).square().mean(-1).cpu().numpy()
                d = value['diagnostics']
                close = int(np.argmin([x['gripper_object_distance'] for x in d]))
                metrics = dict(native_steps=len(value['actions']), any_success=bool(value['success'].any()),
                               macro_goal_mse=costs.tolist(), min_gripper_object_distance=d[close]['gripper_object_distance'],
                               closest_native_step=close, physical_initial=d[0], physical_final=d[-1],
                               initial_rgb=value['initial_rgb'], forecast=forecast(model, value['frames'], value['actions'], goal, mean, std))
                if name == 'expert_diagnostic':
                    metrics['final_rgb_vs_cached_goal'] = rgb_replay_check(cached['goal'], value['final_frame'])
                    assert metrics['final_rgb_vs_cached_goal']['passed']
                    fact = encode(model, value['factorial_frames'])
                    metrics['factorial'] = dict(order=['initial_arm_initial_object', 'goal_arm_initial_object',
                                                       'initial_arm_goal_object', 'goal_arm_goal_object'],
                                                goal_mse=(fact-goal).square().mean(-1).tolist(), joint_names=value['joint_names'],
                                                scope='Rendered joint intervention,not guaranteed physically reachable;diagnostic only.')
                np.savez_compressed(directory/'trajectory.npz', frames=value['frames'], actions=value['actions'],
                                    successes=value['success'], goal=cached['goal'])
                save(directory/'physical.json', d)
                save(directory/'metrics.json', metrics)
                record['paths'][name] = metrics
        rows.append(record)
        save(output/'progress.json', dict(rows=rows))
        print(json.dumps(dict(name=args.name, task=case['task'], seed=case['seed'], completed=len(rows),
                              expert_success=record['paths']['expert_diagnostic']['any_success'])), flush=True)
    assert len(rows) == len(plan['case_seeds'])
    after = model_sha(model); assert after == before
    save(output/'summary.json', dict(passed=True, name=args.name, rows=rows, model_sha256=before,
                                    model_unchanged=True, model_updates=0,
                                    finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    scope='Known cases,diagnostic expert controls and privileged posthoc factors;not a deployable policy or new benchmark score.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['plan', 'name', 'output']:
        parser.add_argument('--'+key, required=True)
    run(parser.parse_args())
