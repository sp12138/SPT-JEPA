"""Plot completed gate only, preserving both seeds and both controllers."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    assert data['passed'] and data['all48_exit0'] and len(data['runs']) == 8
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = {'fixed': '#3776ab', 'adaptive': '#c66a23'}
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6))
    for col, (task, title) in enumerate([('tworoom', 'TwoRoom'), ('cube', 'Cube')]):
        rows = {(r['arm'], r['seed']): r for r in data['runs'] if r['task'] == task}
        for arm, offset in [('fixed', -.16), ('adaptive', .16)]:
            for seed, marker, jitter in [(1, 'o', -.035), (2, '^', .035)]:
                r = rows[arm, seed]
                x = np.arange(2) + offset + jitter
                y = [r['planning'][c]['success_rate'] for c in ['original', 'correlated']]
                for value, c in zip(y, ['original', 'correlated']):
                    assert abs(value - 100*np.mean(r['planning'][c]['successes'])) < 1e-8
                axes[0, col].plot(x, y, marker=marker, color=colors[arm],
                                  linestyle=':', alpha=.9, label=f'{arm} Q, seed {seed}')
                y2 = [r['branches'][b]['clean_over_initial_copy'] for b in ['F', 'Q']]
                axes[1, col].plot(x, y2, marker=marker, color=colors[arm], linestyle=':', alpha=.9)
        axes[0, col].set(title=title, xticks=[0, 1], xticklabels=['Original CEM', 'Correlated CEM'],
                         ylabel='Success (%)', ylim=(-2, 42), xlim=(-.5, 1.5))
        axes[1, col].set(xticks=[0, 1], xticklabels=['Protected F', 'Fixed / adaptive Q'],
                         ylabel='Forecast MSE / initial-copy MSE', ylim=(.45, 1.08), xlim=(-.5, 1.5))
        axes[1, col].axhline(1, color='.45', linestyle='--', linewidth=1)
        for ax in axes[:, col]:
            ax.grid(axis='y', alpha=.2)
    axes[0, 1].legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
    fig.suptitle('Protected observation information does not protect its prediction', fontsize=15, y=.985)
    fig.text(.5, .935, 'Hybrid128 at 600 updates: all 48 endpoints complete; each marker is one training seed.',
             ha='center', fontsize=10)
    fig.text(.07, .055, 'F observations are identical between arms. Lower prediction ratios are better; 1 = copying the initial state.\n'
             'Planning uses the same 50 cases. Two seeds and a short warmup-stage gate do not establish mature superiority.',
             fontsize=9, color='.3')
    fig.subplots_adjust(top=.88, bottom=.16, hspace=.32, wspace=.30)
    for suffix in ['png', 'pdf']:
        fig.savefig(args.output / f'hybrid600.{suffix}', dpi=160, facecolor='white')
    provenance = {'summary_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  'plot_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'runs': 8, 'all_seeds_and_controllers': True,
                  'new_model_calls': 0, 'new_fits': 0}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
