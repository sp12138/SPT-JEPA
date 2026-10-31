#!/usr/bin/env python3
"""CPU-only figure from completed, audited summaries; no model/data calls."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    paths = {
        'span': args.repo / 'experiments/predictor_span_20261031/results/summary.json',
        'local6000': args.repo / 'experiments/local_carrier_20261031/results/gate6000/summary.json',
        'fresh2000': args.repo / 'experiments/local_carrier_20261031/results/fresh2000/summary.json',
        'basis2000': args.repo / 'experiments/local_carrier_20261031/results/basis2000/summary.json',
    }
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    assert data['span']['passed'] and data['span']['model_calls'] == 0
    checks = data['span']['observation_checks']
    assert len(checks) == 4 and all(r['local2000_equals6000'] for r in checks)
    assert max(r['basis_rotation_max_abs'] for r in checks) < 3e-7
    span = {(r['task'], r['family'], r['step'], r['seed']): r
            for r in data['span']['rows'] if r['family'] in ('local', 'basis')}
    planning = {}
    for r in data['fresh2000']['rows']:
        if r['arm'] == 'local':
            planning[(r['task'], 'local', 2000, r['seed'])] = dict(primary=r['primary'], fresh=r['fresh'])
    for r in data['local6000']['rows']:
        if r['arm'] == 'local':
            planning[(r['task'], 'local', 6000, r['seed'])] = r['planning']
            assert r['parent2000'] == planning[(r['task'], 'local', 2000, r['seed'])]
    for r in data['basis2000']['rows']:
        vals = {k: v['success_rate'] for k, v in r['values'].items()}
        if r['arm'] == 'identity':
            assert vals == planning[(r['task'], 'local', 2000, r['seed'])]
        else:
            assert r['arm'] == 'orthogonal'
            planning[(r['task'], 'basis', 2000, r['seed'])] = vals
    groups = [('local', 2000), ('basis', 2000), ('local', 6000)]
    order = [(task, family, step, seed) for task in ('tworoom', 'cube')
             for family, step in groups for seed in (1, 2)]
    assert len(span) == len(planning) == len(order) == 12
    rows = []
    for key in order:
        r, p = span[key], planning[key]
        c = r['clean']
        assert r['rank_LN_effective'] == 191 and r['cache_rows'] == 448
        assert c['pythagorean_relative_gap'] < 3e-9
        assert abs(c['prediction_mse'] - c['oracle_mse'] - c['reachable_error_mse']) < 2e-9
        rows.append(dict(task=key[0], family=key[1], step=key[2], seed=key[3],
                         mse=c['prediction_mse'], outside=c['oracle_mse'],
                         inside=c['reachable_error_mse'], planning=p))
    # This exact descriptive contrast must hold in every task and seed.
    for task in ('tworoom', 'cube'):
        for seed in (1, 2):
            a, b = span[(task, 'local', 2000, seed)]['clean'], span[(task, 'basis', 2000, seed)]['clean']
            assert b['prediction_mse'] < a['prediction_mse'] and b['oracle_mse'] > a['oracle_mse']

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(16.6, 7.7), facecolor='white')
    grid = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.12, 1.1],
                           left=.075, right=.955, top=.78, bottom=.21, wspace=.57)
    outside_color, inside_color = '#d46950', '#6a91aa'
    y = np.array([0, 1, 2.6, 3.6, 5.2, 6.2])
    labels = [f'{family.capitalize()} {step // 1000}k · s{seed}'
              for family, step in groups for seed in (1, 2)]
    for col, task in enumerate(('tworoom', 'cube')):
        ax = fig.add_subplot(grid[0, col])
        subset = [r for r in rows if r['task'] == task]
        outside, inside = [r['outside'] for r in subset], [r['inside'] for r in subset]
        ax.barh(y, outside, height=.72, color=outside_color, label='Outside-space oracle')
        ax.barh(y, inside, left=outside, height=.72, color=inside_color, label='Within-space residual')
        for yy, r in zip(y, subset):
            ax.text(r['mse'] + .009, yy, f"{r['mse']:.3f}\n{100*r['outside']/r['mse']:.0f}% outside",
                    va='center', fontsize=9.4, color='#334155', linespacing=1.25)
        ax.set_yticks(y, labels)
        ax.set_ylim(6.85, -.65)
        ax.set_xlim(0, .56 if task == 'tworoom' else .64)
        ax.set_xticks([0, .2, .4] if task == 'tworoom' else [0, .2, .4, .6])
        ax.set_xlabel('Latent MSE per coordinate', labelpad=9)
        ax.set_title(f"{'A' if col == 0 else 'B'}  {'TwoRoom' if task == 'tworoom' else 'Cube'}", loc='left',
                     fontsize=15, fontweight='bold', pad=15)
        ax.xaxis.grid(True, alpha=.17)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=8)
        for yy in (1.8, 4.4): ax.axhline(yy, color='#e5e7eb', lw=.9)

    ax = fig.add_subplot(grid[0, 2])
    values = np.array([[r['planning'][k] for k in ('primary', 'fresh')] for r in rows])
    image = ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    ax.set_yticks(range(12), [f"{'TR' if r['task']=='tworoom' else 'Cube'}  {r['family']} {r['step']//1000}k · s{r['seed']}" for r in rows])
    ax.set_xticks([0, 1], ['Primary', 'Fresh'])
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', length=0, pad=7, labelsize=10.5)
    for i in range(12):
        for j in range(2):
            ax.text(j, i, f'{values[i,j]:.0f}%', ha='center', va='center',
                    color='white' if values[i,j] >= 60 else '#172535', fontsize=11.5, fontweight='medium')
    for yy in [1.5, 3.5, 5.5, 7.5, 9.5]: ax.axhline(yy, color='white', lw=3 if yy == 5.5 else 1.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title('C  Planning outcomes', loc='left', fontsize=15, fontweight='bold', pad=40)
    ax.set_xlabel('50 cases per set; both seeds shown', fontsize=10.4, labelpad=12)

    fig.suptitle('Lower prediction MSE can conceal a larger fixed-output-space error floor',
                 x=.075, y=.97, ha='left', fontsize=20, fontweight='bold', color='#172535')
    fig.text(.075, .918, 'Completed checkpoints · identical or orthogonally rotated 1024D observation codes within each task',
             fontsize=12.1, color='#475569')
    fig.legend(handles=[Patch(color=outside_color, label='Outside current affine output space: oracle lower bound'),
                        Patch(color=inside_color, label='Residual inside that output space')],
               loc='upper left', bbox_to_anchor=(.07, .887), ncol=2, frameon=False, fontsize=11)
    fig.text(.075, .118, 'Basis 2k: lower total MSE, higher outside-space floor in both tasks and both seeds.',
             fontsize=12.2, fontweight='bold', color='#172535')
    fig.text(.075, .073, 'Current-weight bound (rank 191/1024); not a causal explanation of planning. Categorical snapshots, not a training-trajectory estimate.',
             fontsize=10.8, color='#475569')
    fig.text(.075, .037, 'MSE: 64 fixed validation windows, 7 forecast horizons. Planning: original-command controller; primary and fresh are separate case sets.',
             fontsize=10.8, color='#475569')
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(args.output / f'mechanism.{ext}', dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    provenance = {'sources': {k: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},
                  'span_full_results': data['span']['full_results'],
                  'span_full_results_sha256': data['span']['full_results_sha256'],
                  'observation_checks': checks, 'rows': rows, 'model_calls': 0, 'cuda_initialized': False,
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'inputs-and-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'passed': True, 'rows': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
