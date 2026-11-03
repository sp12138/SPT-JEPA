"""Cache shared image goals and exact initial task vectors before model outcomes."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np

from experiments.metaworld_eval_20261103.environment import make_environment, pixels, expert

TASKS = ['reach-v3', 'reach-wall-v3', 'push-v3', 'pick-place-v3', 'drawer-open-v3',
         'door-open-v3', 'peg-insert-side-v3', 'bin-picking-v3']


def main(source, output, seeds, tasks):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in tasks:
        for seed in seeds:
            env, initial_state, vector = make_environment(task, seed)
            initial = pixels(env); policy = expert(source, task)
            state = initial_state.copy(); success = False; any_success = False
            try:
                for step in range(500):
                    action = np.clip(policy.get_action(state), -1, 1).astype(np.float32)
                    state, reward, terminated, truncated, info = env.step(action)
                    success = bool(info['success']); any_success |= success
                    if success or terminated or truncated:break
                goal = pixels(env)
            finally:env.close()
            replay, reconstructed, _ = make_environment(task, seed, vector)
            try:reconstructed_pixels = pixels(replay)
            finally:replay.close()
            assert np.array_equal(initial_state, reconstructed), (task, seed, np.max(np.abs(initial_state - reconstructed)))
            assert np.array_equal(initial, reconstructed_pixels), (task, seed, 'RGB reset differs')
            path = output / f'{task}-seed{seed}.npz'
            np.savez_compressed(path, initial=initial, goal=goal, rand_vec=vector, initial_state=initial_state)
            rows.append(dict(task=task, seed=seed, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             initial_rgb_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                             expert_any_success=any_success, expert_final_success=success, expert_steps=step + 1,
                             exact_state_and_rgb_reconstruction=True))
            print(json.dumps({k:v for k,v in rows[-1].items() if k not in ['path','sha256','initial_rgb_sha256']}), flush=True)
    result = dict(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
                  packages={p:importlib.metadata.version(p) for p in ['metaworld','mujoco','gymnasium','numpy']},
                  primary_seeds=[s for s in seeds if s < 1000], fresh_seeds=[s for s in seeds if s >= 1000],
                  scope='Shared cached image-goal extension on the independent LeRobot camera/data. '
                        'Expert only constructs evaluation goals,never supplies training/online optimal actions. '
                        'Goal uses first native success within500 steps,or final frame on failure; all cases/failures retained. '
                        'Not the published JEPA-WMs100-step goal protocol or standard task-ID-conditionedMT50. '
                        'Expert failure is not proof of task impossibility; report goal-valid and all-case rates separately.')
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42,52))+list(range(1042,1052)))
    parser.add_argument('--tasks', nargs='+', default=TASKS)
    args = parser.parse_args()
    main(args.source, args.output, args.seeds, args.tasks)
