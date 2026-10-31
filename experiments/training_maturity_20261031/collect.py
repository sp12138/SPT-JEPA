"""Read completed Cube logs/reports only. No models, fitting, simulation or GPU."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

B = Path('/efs/prescottsun/SPT-JEPA')
RUNS = {
    'local': ('local_isotropic6000_20261031', 'cube-spatial-local-seed{}', 6000),
    'native': ('native_cube12000_20261031', 'cube-native-lewm-seed{}', 12000),
    'tf': ('cube_tf12000_20261031', 'cube-temporal-forecast-seed{}', 12000),
}


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def trend(rows, cap):
    # The last logged minibatch at a duplicate boundary is kept once. These are
    # correlated stochastic traces, not IID validation observations or p-values.
    by_step = {r['step']: r for r in rows}
    window = [by_step[s] for s in sorted(by_step) if .75 * cap <= s <= cap]
    early = [r for r in window if r['step'] < .875 * cap]
    late = [r for r in window if r['step'] >= .875 * cap]
    assert early and late and window[-1]['step'] == cap
    keys = [k for k in window[-1] if k.startswith('grad_norm_')]
    keys += [k for k in ['loss', 'fm', 'rate', 'pred', 'sigreg', 'feature_std'] if k in window[-1]]
    result = {}
    for key in keys:
        y = np.asarray([r[key] for r in window], dtype=float)
        x = np.asarray([r['step'] for r in window], dtype=float)
        assert np.isfinite(y).all()
        a, z = float(np.median([r[key] for r in early])), float(np.median([r[key] for r in late]))
        slope = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
        result[key] = dict(early_median=a, late_median=z,
                           median_change_percent=100 * (z / a - 1) if a else None,
                           slope_per_1000_updates=1000 * slope,
                           quarter_slope_change_percent=100 * slope * (.25 * cap) / a if a else None,
                           minimum=float(y.min()), maximum=float(y.max()))
    return dict(interval=[.75 * cap, cap], early_steps=[r['step'] for r in early],
                late_steps=[r['step'] for r in late], unique_records=len(window),
                duplicates_removed=len(rows) - len(by_step), metrics=result,
                qualification='Descriptive medians/OLS only: correlated sampled training minibatches; no statistical-significance claim.')


def state_summary(s):
    if not s:
        return None
    c = s['conditions']
    clean = c['clean_rollout']['mse']
    out = dict(manifest_sha256=s.get('manifest_sha256'),
               observed_std=s.get('observed_std', s.get('observed_state_std_mean')),
               observed_rms=s.get('observed_rms', s.get('observed_state_rms')),
               initial_copy_mse=s['actual_temporal_variation']['initial_mean'],
               local_copy_mse=s['actual_temporal_variation']['local_mean'],
               conditions={})
    for k, v in c.items():
        out['conditions'][k] = {q: v[q] for q in ['mse', 'normalized_by_initial_variation', 'normalized_by_local_variation'] if q in v}
        out['conditions'][k]['over_clean'] = v['mse'] / clean
    return out


def health_summary(h):
    if not h:
        return None
    return dict(feature_std=h.get('feature_std'), summary=h.get('summary'), audit=h.get('audit'))


def small_planning(obj):
    if isinstance(obj, dict):
        return {k: small_planning(v) for k, v in obj.items()
                if k not in ['signature', 'planning_signature', 'outcomes', 'old_outcomes', 'new_outcomes', 'episode_successes']}
    return obj


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--output', type=Path, default=B/'reports/training-maturity-20261031')
    args = args.parse_args()
    root = B/'data/hardtasks-v1/cube-metadata'
    card = read(root/'dataset_card.json')
    lengths, ids, train = [np.load(root/n) for n in ['lengths.npy', 'episode_ids.npy', 'train_episodes.npy']]
    selected = np.isin(ids, train) & (lengths > 7 * card['frameskip'])
    windows = int((lengths[selected] - 7 * card['frameskip']).sum())
    updates = windows // 128
    assert windows == 1494000 and updates == 11671 and int(selected.sum()) == 9000
    total, warmup = 124910, 1249
    budgets = {}
    for step in [2000, 6000, 12000, total]:
        scale = .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
        budgets[str(step)] = dict(windows_seen=128 * step, loader_epochs=step/updates,
                                 natural_window_epochs=128*step/windows,
                                 schedule_percent=100*step/total,
                                 lr_fraction_of_peak=scale, base_lr=5e-5*scale)
    runs = []
    for family, (run_root, name, cap) in RUNS.items():
        for seed in [1, 2]:
            run = B/'runs'/run_root/name.format(seed)
            cfg = read(run/'config.json')
            assert cfg['max_steps'] == total and cfg['batch_size'] == 128 and cfg['model']['window'] == 8
            records = [json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines() if l]
            assert records[-1]['state'] == 'finished' and records[-1]['step'] == cap
            assert abs(records[-1]['learning_rates'][0] - budgets[str(cap)]['base_lr']) < 1e-15
            source = read(run.parent/'source-record.json')
            source_dir = Path(source['source'])
            frozen_paths = ['as_jepa_tools/benchmark_data.py', 'as_jepa_lewm/train.py']
            if family == 'native': frozen_paths += ['as_jepa_lewm/native_lewm.py']
            runs.append(dict(family=family, seed=seed, cap=cap,
                             config=artifact(run/'config.json'), log=artifact(run/'metrics.jsonl'),
                             source_record=artifact(run.parent/'source-record.json'),
                             model_source_commit=source['source_commit'],
                             source_files=[artifact(source_dir/p) for p in frozen_paths],
                             last_quarter=trend(records, cap), final_record=records[-1]))

    reports = {}
    for name in ['local-isotropic6000-20261031', 'local-isotropic2000-20261031',
                 'native-lewm2000-20261031', 'native-cube12000-20261031',
                 'carrier6000-20261031', 'cube-tf12000-20261031']:
        path = B/'reports'/name/'results.json'
        d = read(path)
        rows = []
        for r in d['rows']:
            id_ = r['id']
            if not id_.startswith('cube-') or not any(s in id_ for s in ['spatial-local', 'native-lewm', 'temporal-forecast']):
                continue
            row = dict(id=id_, seed=r.get('seed', r.get('training_seed')))
            for key in ['state', 'state6000', 'state600', 'parent2000_state']:
                if key in r: row[key] = state_summary(r[key])
            for key in ['health', 'health_old', 'health_new']:
                if key in r: row[key] = health_summary(r[key])
            for key in ['old_state', 'state_summary', 'parent2000_state_summary', 'health_summary',
                        'native_success', 'calibrated_success', 'success_rate', 'parent2000',
                        'planning', 'planning6000', 'fullstate_metrics', 'fullstate_metrics600']:
                if key in r: row[key] = small_planning(r[key])
            rows.append(row)
        comparisons = d.get('comparison_summaries', d.get('joint_case_comparisons'))
        if comparisons:
            comparisons = [c for c in comparisons if c.get('task', 'cube') == 'cube']
        reports[name] = dict(artifact=artifact(path), rows=rows, comparison_summaries=comparisons)

    tf_early = []
    for seed in [1, 2]:
        path = B/'runs/hardtasks_learn2000_20261030'/f'health-cube-temporal-forecast-seed{seed}-learn2000/health.json'
        tf_early.append(dict(seed=seed, step=2000, artifact=artifact(path), health=health_summary(read(path))))

    official = B/'sources/stable-worldmodel'
    recipe = official/'scripts/train/config/lewm.yaml'
    text = recipe.read_text()
    assert 'max_epochs: 100' in text and 'history_size: 3' in text
    source = official/'scripts/train/lewm.py'
    commit = '21446f1ede6d5284e981bd7b47f432b994e6d812'
    for p in [recipe, source]:
        blob = subprocess.check_output(['git', '-C', str(official), 'show', f'{commit}:{p.relative_to(official)}'])
        assert blob == p.read_bytes()
    out = dict(scope='CPU/read-only completed Cube logs and existing reports; no new forward/probe/evaluation or training.',
               collector=artifact(__file__), data=dict(train_episodes=int(selected.sum()),
               train_rows=int(lengths[selected].sum()), window=8, frameskip=card['frameskip'],
               valid_window_starts=windows, loader_windows_per_epoch=128*updates,
               updates_per_loader_epoch=updates, metadata=[artifact(root/n) for n in
                   ['dataset_card.json', 'lengths.npy', 'episode_ids.npy', 'train_episodes.npy']]),
               schedule=dict(max_steps=total, warmup_updates=warmup, budgets=budgets), runs=runs,
               existing_reports=reports, tf_2000_health=tf_early, official_recipe=dict(commit=commit,
               yaml=artifact(recipe), script=artifact(source), default_epochs=100, default_batch=128,
               history=3, window=4,
               qualification='Repository default, not proof of the released checkpoints actual training duration. Original script interval=epoch versus adapted per-update scheduler; defaults are not an executed matching-budget run.'))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output), runs=len(runs), updates_per_epoch=updates)))


if __name__ == '__main__':
    main()
