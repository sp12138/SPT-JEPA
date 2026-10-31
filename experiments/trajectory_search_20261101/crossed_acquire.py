"""Train-only crossed initial-state x open-loop-action observations, not labels of optimality."""
import argparse
from pathlib import Path
import sys
import numpy as np
from common import B, array_sha, read, sha, write


class Complete(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank-report', required=True)
    p.add_argument('--native-evaluator', required=True)
    p.add_argument('--reference-manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--first-quad', type=int, default=0)
    p.add_argument('--quads', type=int, default=16)
    a = p.parse_args()
    import torch
    from stable_worldmodel.policy import RandomPolicy
    from stable_worldmodel.world.env_pool import EnvPool
    from as_jepa_tools import benchmark_evaluate
    bank = read(a.bank_report)
    assert bank['passed'] and bank['no_validation_test_overlap'] and sha(bank['bank']) == bank['bank_sha256']
    assert Path(benchmark_evaluate.__file__).resolve() == Path(a.native_evaluator).resolve()
    assert 0 <= a.first_quad and a.first_quad + a.quads <= 512
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    entries = bank['episodes'][2*a.first_quad:2*(a.first_quad+a.quads)]
    with np.load(bank['bank']) as data:
        library_actions = data['raw_actions'][2*a.first_quad:2*(a.first_quad+a.quads)].copy()
    # Within each quad: s0,a0; s0,a1; s1,a0; s1,a1. All actions are fixed
    # before executing any outcome. Nothing is labeled optimal/successful.
    pairs, commands = [], []
    for q in range(a.quads):
        for state in (0, 1):
            for action in (0, 1):
                e = entries[2*q+state]
                pairs.append(dict(episode_id=e['episode'], start_step=e['start'], goal_step=e['start']+25))
                commands.append(library_actions[2*q+action])
    commands = np.stack(commands).transpose(1, 0, 2).copy()  # T,N,D
    manifest = read(a.reference_manifest)
    assert manifest['task'] == 'cube'
    manifest.update(campaign_split='train', pairs=pairs, acquisition='Crossed train states/actions; actual observations only; no optimal-action labels.')
    mp = out / 'train-reset-manifest.json'
    write(mp, manifest)
    old_action, old_step = RandomPolicy.get_action, EnvPool.step
    starts, endpoints, times, masks, source_commands = [], [], [], [], []
    counter = 0

    def action(policy, obs, **kwargs):
        nonlocal counter
        if counter == 25:
            raise Complete()
        if not starts:
            pixels = np.asarray(obs['pixels']).copy()
            assert pixels.shape == (4*a.quads, 224, 224, 3)
            starts.append(pixels)
        value = commands[counter].copy()
        assert policy.env.action_space.shape == value.shape
        counter += 1
        return value

    def step(pool, actions, mask=None):
        np.testing.assert_array_equal(actions, commands[counter-1])
        result = old_step(pool, actions, mask=mask)
        source_commands.append(array_sha(actions))
        if counter % 5 == 0:
            endpoints.append(np.asarray(result[-1]['pixels']).copy())
            times.append(counter)
            masks.append(np.ones(4*a.quads, bool) if mask is None else np.asarray(mask, dtype=bool).copy())
        return result

    argv = [benchmark_evaluate.__file__, '--manifest', str(mp), '--checkpoint', 'random',
            '--data', str(B/'data/hardtasks-v1/cube-metadata'), '--output', str(out/'native-unused')]
    saved = sys.argv
    try:
        RandomPolicy.get_action, EnvPool.step = action, step
        sys.argv = argv
        try:
            benchmark_evaluate.main()
        except Complete:
            pass
    finally:
        RandomPolicy.get_action, EnvPool.step = old_action, old_step
        sys.argv = saved
    assert counter == 25 and len(starts) == 1 and times == [5, 10, 15, 20, 25]
    images = np.stack(starts + endpoints, axis=1).reshape(a.quads, 4, 6, 224, 224, 3)
    valid = np.stack(masks, axis=1).reshape(a.quads, 4, 5)
    executed = commands.transpose(1, 0, 2).reshape(a.quads, 4, 25, 5)
    assert np.array_equal(images[:, 0, 0], images[:, 1, 0])
    assert np.array_equal(images[:, 2, 0], images[:, 3, 0])
    assert np.array_equal(executed[:, 0], executed[:, 2])
    assert np.array_equal(executed[:, 1], executed[:, 3])
    # Keep every predetermined quad, including incomplete/degenerate cases.
    files = []
    for q in range(a.quads):
        path = out / f'quad{a.first_quad+q:04d}.npz'
        np.savez_compressed(path, pixels=images[q], raw_actions=executed[q], observed_step_mask=valid[q])
        files.append(dict(quad=a.first_quad+q, path=str(path), sha256=sha(path),
              states=entries[2*q:2*q+2], all_steps_active=bool(valid[q].all()),
              actions_distinct=not np.array_equal(executed[q, 0], executed[q, 1])))
    write(out / 'acquisition.json', dict(passed=True, first_quad=a.first_quad, quads=a.quads, files=files,
          bank_report_sha256=sha(a.bank_report), bank_sha256=bank['bank_sha256'], manifest_sha256=sha(mp),
          actual_env_step_calls=25, branch_steps=25*4*a.quads, recorded_rgb_frames=6*4*a.quads,
          same_start_RGB_within_state=True, same_actual_commands_across_states=True,
          command_step_sha256=source_commands, native_evaluator=str(Path(benchmark_evaluate.__file__).resolve()),
          native_evaluator_sha256=sha(benchmark_evaluate.__file__), native_random_policy_sha256=sha(sys.modules[RandomPolicy.__module__].__file__),
          fields_for_learning=['pixels', 'raw_actions', 'observed_step_mask'], optimal_action_labels=0,
          world_model_calls=0, all_training_episodes=True, all_prespecified_quads_retained=True,
          limitation='Additional simulator interactions from train reset states. Shared queries/data must be given to all learning controls. Incomplete quads cannot establish full crossed-outcome support. No novelty or task-performance claim.'))
    print(f'Crossed observation acquisition complete:{a.quads} quads,{25*4*a.quads} branch steps.', flush=True)


if __name__ == '__main__':
    main()
