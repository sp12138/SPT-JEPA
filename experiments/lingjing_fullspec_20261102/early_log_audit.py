"""Read all reference learning traces; do not infer FP32 information from BF16 training spread."""
import argparse
from pathlib import Path

from common import read, write, sha


def main(inventory, output):
    inventory = Path(inventory)
    inv = read(inventory)
    rows = []
    seen = set()
    for ref in inv['rows']:
        key = ref['family'], ref['seed'], ref['run']
        if key in seen or ref['family'] not in ['original', 'lower_beta', 'TF', 'FO', 'TN']:
            continue
        seen.add(key)
        run = Path(ref['run'])
        path = run / 'metrics.jsonl'
        if not path.exists():
            continue
        trace = [__import__('json').loads(x) for x in path.read_text().splitlines()]
        cfg = read(run / 'config.json')
        sigma = cfg['model'].get('sigma0', .1)
        chosen = {}
        for step in [100, 1000, 2000, 4000, 6000, 12491, 24982, 37473]:
            candidates = [r for r in trace if abs(r['step'] - step) <= max(100, step * .01)]
            if candidates:
                row = min(candidates, key=lambda r: abs(r['step'] - step))
                chosen[str(step)] = {k: v for k, v in row.items()
                                     if k in ['step', 'feature_std', 'rate', 'fm', 'entry', 'surprise', 'calib_slope', 'loss', 'learning_rates']}
                if 'feature_std' in row:
                    chosen[str(step)]['training_spread_over_sigma0'] = row['feature_std'] / sigma
        rows.append(dict(family=ref['family'], seed=ref['seed'], run=str(run), metrics_sha256=sha(path),
                         config_sha256=sha(run / 'config.json'), points=chosen, first_step=trace[0]['step'], last_step=trace[-1]['step']))
    write(output, dict(rows=rows, inventory_sha256=sha(inventory),
                      scope='Descriptive historical trajectories, all available seeds. BF16 logged feature spread is not an FP32 information measurement. No training/simulation or causal attribution.'))
    for row in rows:
        if row['family'] == 'lower_beta':
            print(row['seed'], {k: round(v.get('training_spread_over_sigma0', -1), 5) for k, v in row['points'].items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    main(a.inventory, a.output)
