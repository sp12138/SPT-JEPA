"""Plot every seed from the completed short gate; no winner selection or task-score claim."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary, output = Path(summary), Path(output)
    rows = json.loads(summary.read_text())['rows']
    output.mkdir(parents=True, exist_ok=False)
    table = []
    for row in rows:
        job, state = row['job'], row['state']
        table.append(dict(id=job['id'], task=job['task'], model=job['model_class'], seed=job['seed'],
                          snr=state['feature_std_over_sigma0'],
                          swap_percent=100 * state['state_swap_relative_error_increase'] if 'state_swap_relative_error_increase' in state else None,
                          noisy_agent_r2=state.get('noisy_agent_r2'), gate_passed=row['gate_passed']))
    with (output / 'gate-values.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(table)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    labels = [['T1', 'T2', 'T5', 'T6', 'T7', 'T8', 'T9', 'T4 beta=0'],
              ['K1', 'K2', 'K3', 'K4', 'K5', 'K6'], ['beta=.1', 'no init', 'lambda=0', 'init=.3']]
    prefixes = [['round2-T-' + name for name in ['T1', 'T2', 'T5', 'T6', 'T7', 'T8', 'T9', 'T4-beta0']],
                ['round2-K-K' + str(i) for i in range(1, 7)],
                ['decoder_free-T-df-' + name for name in ['b0.1', 'noinit', 'l0', 'init3']]]
    for index, ax in enumerate(axes):
        metric = 'noisy_agent_r2' if index == 2 else 'swap_percent'
        for seed, colour, marker in [(1, '#2369a0', 'o'), (2, '#dc7834', 's')]:
            values = [next(r[metric] for r in table if r['id'] == prefix + '-s' + str(seed)) for prefix in prefixes[index]]
            ax.scatter([i + (-.1 if seed == 1 else .1) for i in range(len(values))], values,
                       color=colour, marker=marker, label='seed ' + str(seed), s=36, zorder=3)
        ax.set_xticks(range(len(labels[index])), labels[index], rotation=35, ha='right')
        ax.axhline(0, color='#a0a0a0', linewidth=.7)
        ax.grid(axis='y', alpha=.18)
    axes[0].set(title='TwoRoom: renderer ignores state', ylabel='Shuffling state: FM error increase (%)', ylim=(-.02, .02))
    axes[0].text(.02, .94, 'Required +5% is above this scale', transform=axes[0].transAxes, va='top', fontsize=9)
    axes[1].set(title='Cube: all six pairs use state', ylabel='Shuffling state: FM error increase (%)', ylim=(-2, 92))
    axes[1].axhline(5, linestyle='--', color='#555555', linewidth=1, label='use gate +5%')
    axes[2].set(title='TwoRoom: decoder-free state is weak', ylabel='Noisy-code agent-position R2', ylim=(-.05, .065))
    axes[2].axhline(.05, linestyle='--', color='#555555', linewidth=1, label='position gate .05')
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle('Completed 2000-update gates | FP32 | both training seeds retained', fontsize=13)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(output / ('insights-gates.' + suffix), dpi=180)
    svg = output / 'insights-gates.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    (output / 'provenance.json').write_text(json.dumps(dict(summary=str(summary),
        summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=len(table), scope='All36 new short gates. Plot shows16+12 renderer and8 decoder-free gates; no P0 success scores or confidence interval across training seeds.'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--summary', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.summary, a.output)
