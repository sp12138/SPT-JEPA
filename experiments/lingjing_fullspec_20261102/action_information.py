"""CPU-only action/state decodability control; no RGB, model update or optimal-action labels."""
import argparse
from pathlib import Path
import numpy as np
from common import B, write, sha


def arrays(task, split, count, seed):
    rng = np.random.default_rng(seed)
    if task == 'pusht':
        from as_jepa_tools.data import PushTWindows
        root = B / 'data/pusht-lance-full-v1/phase0'
        ds = PushTWindows(root, window=4, split='test' if split == 'validation' else 'train'); ds.open()
        picks = rng.choice(len(ds), min(count, len(ds)), replace=False)
        located = [ds.locate(int(i)) for i in picks]
        rows = np.array([int(ds.offsets[ep]) + start + 2 for ep, start in located])
        current = np.asarray(ds.actions[rows], dtype=np.float64)
        history = np.asarray(ds.actions[rows[:, None] + np.arange(-2, 1)], dtype=np.float64).reshape(len(rows), -1)
        previous, future = np.asarray(ds.states[rows, :4], dtype=np.float64), np.asarray(ds.states[rows + 1, :4], dtype=np.float64)
        names = ['agent_x', 'agent_y', 'block_x', 'block_y']
    else:
        from as_jepa_tools.benchmark_data import BenchmarkWindows
        root = B / 'data/hardtasks-v1' / (task + '-metadata')
        ds = BenchmarkWindows(root, window=4, split=split); ds.open()
        picks = rng.choice(len(ds), min(count, len(ds)), replace=False)
        located = [ds.locate(int(i)) for i in picks]
        rows = np.array([int(ds.offsets[ep]) + start + 2 * ds.stride for ep, start in located])
        current = np.asarray(ds.actions[rows[:, None] + np.arange(ds.stride)], dtype=np.float64)
        current = ((current - ds.mean) / ds.std).reshape(len(rows), -1)
        history = np.asarray(ds.actions[rows[:, None] + np.arange(-2 * ds.stride, ds.stride)], dtype=np.float64)
        history = ((history - ds.mean) / ds.std).reshape(len(rows), -1)
        keys = ['pos_agent', 'pos_target'] if task == 'tworoom' else ['privileged_block_0_pos', 'proprio_effector_pos']
        values = {key: np.asarray(ds.file[key], dtype=np.float64) for key in keys}
        previous = np.concatenate([value[rows] for value in values.values()], -1)
        future = np.concatenate([value[rows + ds.stride] for value in values.values()], -1)
        names = [key + '_' + str(i) for key, value in values.items() for i in range(value.shape[-1])]
        ds.file.close()
    episodes = [int(ds.episode_ids[ep]) if hasattr(ds, 'episode_ids') else int(ep) for ep, start in located]
    assert all(np.isfinite(value).all() for value in [current, history, previous, future])
    return current, history, previous, future, episodes, names, sha(root / 'dataset_card.json')


def main(output):
    from as_jepa_tools.benchmark_information_audit import ridge_probe
    results = {}
    for task in ['pusht', 'tworoom', 'cube']:
        current, history, previous, future, eps, names, card = arrays(task, 'train', 20000, 17)
        tc, th, tp, tf, teps, test_names, test_card = arrays(task, 'validation', 5000, 18)
        assert names == test_names and card == test_card and not set(eps).intersection(teps)
        rng = np.random.default_rng(1042)
        modes = dict(current_action=(current, tc), action_history3=(history, th),
                     shuffled_action=(current[rng.permutation(len(current))], tc[rng.permutation(len(tc))]),
                     oracle_previous_position_plus_action=(np.concatenate([previous, current], -1), np.concatenate([tp, tc], -1)))
        scores = {mode: dict(absolute_next=ridge_probe(x, future, tx, tf),
                             displacement=ridge_probe(x, future - previous, tx, tf - tp)) for mode, (x, tx) in modes.items()}
        results[task] = dict(targets=names, scores=scores, train_count=len(current), validation_count=len(tc),
                             train_episode_count=len(set(eps)), validation_episode_count=len(set(teps)), dataset_card_sha256=card)
        print(task, {mode: value['absolute_next']['r2_against_train_mean'] for mode, value in scores.items()}, flush=True)
    write(output, dict(tasks=results, ridge=.001, GPU_calls=0, RGB_reads=0, model_updates=0,
          scope='Action-only observational decodability, not causal action effect. Canonical episode-disjoint splits, fixed ridge and train-only standardization. Privileged positions are offline diagnostic targets, never policy/world-model inputs or optimal-action training labels. Oracle-state control is explicitly diagnostic.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    main(p.parse_args().output)
