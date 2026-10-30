"""Plot verified planning outcomes; points retain both training seeds."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source, out = Path(a.summary), Path(a.output)
    data = json.loads(source.read_text())
    assert data['completed_endpoint_count'] == 68
    out.mkdir(parents=True, exist_ok=True)
    styles = {
        'identity': ('#4263a5', 'Fixed carrier'),
        'unconstrained': ('#b64e35', 'Carrier + free chart'),
        'bounded': ('#229075', 'Carrier + bounded chart'),
        'temporal-forecast': ('#6c6876', 'Temporal-forecast'),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
    for row, task in enumerate(('tworoom', 'cube')):
        for col, cases in enumerate(('primary', 'fresh')):
            ax = axes[row, col]
            for arm, (color, label) in styles.items():
                runs = [r for r in data['rows'] if r['task'] == task and r['arm'] == arm]
                if not runs:
                    continue
                assert sorted(r['seed'] for r in runs) == [1, 2]
                means = []
                for step in (2000, 6000):
                    values = [r[f'success{step}'][cases] for r in runs]
                    means.append(sum(values) / 2)
                    for r, value in zip(runs, values):
                        ax.scatter(step, value, s=39, marker='o' if r['seed'] == 1 else '^',
                                   color=color, edgecolors='white', linewidth=.5, zorder=3)
                ax.plot([2000, 6000], means, color=color, lw=1.8, label=label)
            ax.set_title(f'{"TwoRoom" if task == "tworoom" else "Cube"} | {cases} cases', loc='left')
            ax.set_xlim(1600, 6400)
            ax.set_ylim(-3, 103)
            ax.set_xticks([2000, 6000], ['2,000', '6,000'])
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.grid(axis='y', alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            if col == 0:
                ax.set_ylabel('Planning success (%)')
            if row == 1:
                ax.set_xlabel('Training updates')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, .94))
    fig.suptitle('TwoRoom improves strongly; the fixed Cube carrier does not', x=.08, ha='left', fontsize=15)
    fig.text(.08, .035, 'Points: training seeds 1 (circle), 2 (triangle). Lines: seed means. 50 common cases per set.\n'
             'Fresh cases share the validation pool. Two seeds do not establish broad stability; native LeWM is evaluated separately.',
             fontsize=9, color='#444444')
    fig.subplots_adjust(top=.80, bottom=.16, left=.08, right=.98, hspace=.28, wspace=.14)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'planning6000.{ext}', dpi=180)
    (out / 'figure-provenance.json').write_text(json.dumps(dict(
        summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        completed_endpoints=68, source=str(source), scope='All primary/fresh outcomes; no selected training seed.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
