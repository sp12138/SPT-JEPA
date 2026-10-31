"""Plot all fixed-bank models/seeds; no fitting or selection of outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p = argparse.ArgumentParser()
p.add_argument('--summary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
d = json.loads(a.summary.read_text())
assert d['all12_pass'] and len(d['rows']) == 12
a.out.mkdir(parents=True, exist_ok=True)
kinds = ['identity', 'static', 'state', 'local6000', 'native12000', 'tf12000']
labels = ['Identity 2k', 'STATIC 2k', 'STATE 2k', 'Local 6k', 'Native 12k', 'TF 12k']
colors = ['#17659b', '#cc6527']
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
for ri, group in enumerate(['nonoracle', 'all']):
    for ci, keys in enumerate([
        ['encoded_predicted_rank', 'physical_encoded_rank'],
        ['predicted_pose_cost', 'encoded_oracle_pose_cost'],
    ]):
        ax = axes[ri, ci]
        for row in d['rows']:
            y = kinds.index(row['kind']) + (-.11 if row['seed'] == 1 else .11)
            stats = row['summary'][group]
            for j, key in enumerate(keys):
                v = stats[key]
                lo, hi = v['ci95']
                ax.plot([lo, hi], [y, y], color=colors[j], alpha=.55, lw=1.4)
                ax.plot(v['mean'], y, 'o' if row['seed'] == 1 else 's',
                        color=colors[j], ms=5, markeredgecolor='white', markeredgewidth=.4)
        ax.set_yticks(range(6), labels)
        ax.set_ylim(6.1, -.5)
        ax.grid(axis='x', color='#e3e5e8', lw=.7)
        ax.spines[['top', 'right']].set_visible(False)
        if ci == 0:
            ax.set_xlim(0, 1.03)
            ax.set_xlabel('Mean within-case rank correlation (higher is better)')
        else:
            oracle = d['rows'][0]['summary'][group]['physical_oracle_pose_cost']['mean']
            ax.axvline(oracle, color='#30343b', ls='--', lw=1.2)
            ax.text(oracle + .055, 5.85, f'Physical selector: {oracle:.3f}', fontsize=9)
            ax.set_xlim(-.12, 5.6)
            ax.set_xlabel('Selected physical pose cost (lower is better)')
        bank = '22 nonoracle candidates' if ri == 0 else 'All 39, including 17 recorded-action neighbors'
        ax.set_title(bank, fontsize=11, loc='left', pad=10)
fig.legend(handles=[Line2D([], [], color=colors[0], label='Predicted vs true encoded cost'),
                          Line2D([], [], color=colors[1], label='True encoded vs physical cost')],
                  loc='upper left', bbox_to_anchor=(.12, .934), frameon=False, fontsize=9)
fig.legend(handles=[Line2D([], [], color=colors[0], label='Predicted-cost selector'),
                          Line2D([], [], color=colors[1], label='True-encoded-cost selector')],
                  loc='upper left', bbox_to_anchor=(.62, .934), frameon=False, fontsize=9)
fig.suptitle('Accurate latent-cost ranking does not ensure a good physical choice',
             fontsize=16, x=.06, ha='left', y=.99)
fig.text(.06, .947, 'Cube | 50 fixed cases | every model and both seeds | no new simulation or training', fontsize=11)
fig.text(.06, .035, 'Circles: seed 1; squares: seed 2. Bars: case-bootstrap 95% intervals, conditional on these trained models.\n'
                   'Pose cost = max(position error / 3 cm, symmetry-aware angle error / 15 degrees).\n'
                   'An old 25-step candidate bank, not current-model CEM or a 50-step closed-loop benchmark.', fontsize=10)
fig.subplots_adjust(left=.12, right=.985, top=.83, bottom=.15, hspace=.43, wspace=.31)
for ext in ['png', 'pdf']:
    fig.savefig(a.out / f'ranking-mechanism.{ext}', dpi=170, facecolor='white')
(a.out / 'provenance.json').write_text(json.dumps({
    'summary_sha256': hashlib.sha256(a.summary.read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows': len(d['rows']), 'groups': ['nonoracle', 'all'],
    'all_original_groups_available': 'summary.json also contains recorded-only results',
    'new_model_calls': 0,
}, indent=2) + '\n')
