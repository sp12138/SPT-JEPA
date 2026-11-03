"""Arithmetic on archived final endpoints; no models or new evaluations."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def mean(values):
    return statistics.mean(values)


def main(performance, historical, output):
    performance, historical, output = map(Path, [performance, historical, output])
    rows = json.loads(performance.read_text())['rows']
    old = json.loads(historical.read_text())
    groups = defaultdict(list)
    for row in rows:
        for arm in row['arms']:
            groups[row['track'], row['block'], row['arm'], row['steps'], arm['variant']].append((row, arm))
    grouped = []
    for key, members in groups.items():
        grouped.append(dict(track=key[0], block=key[1], arm=key[2], steps=key[3], variant=key[4],
                            primary=mean(a['primary'] for _, a in members), fresh=mean(a['fresh'] for _, a in members),
                            seeds=[dict(seed=r['seed'], primary=a['primary'], fresh=a['fresh']) for r, a in members]))
    p2 = {r['arm']: [] for r in rows if r['track'] == 'round2' and r['block'] == 'P2'}
    references = {r['training_seed']: {m['seed']: m['value'] for m in r['metrics']} for r in old['rows']}
    for row in rows:
        if row['track'] != 'round2' or row['block'] != 'P2':
            continue
        arm = row['arms'][0]
        new = {m['value']['seed']: m['value'] for m in arm['metrics']}
        reference = references[row['seed']]
        for seed, value in reference.items():
            assert new[seed]['episodes'] == value['episodes']
            assert new[seed]['start_steps'] == value['start_steps']
        values = {}
        for name, seeds in [('matched_primary42_43', [42, 43]), ('fresh', [1042, 1043, 1044])]:
            values[name] = mean(new[s]['metrics']['success_rate'] for s in seeds)
            values[name + '_historical_tf'] = mean(reference[s]['metrics']['success_rate'] for s in seeds)
            values[name + '_delta_pp'] = values[name] - values[name + '_historical_tf']
        p2[row['arm']].append(dict(seed=row['seed'], **values))
    comparisons = []
    for name, members in p2.items():
        assert {r['seed'] for r in members} == {1, 2}
        comparisons.append(dict(arm=name, seeds=members, **{k: mean(r[k] for r in members) for k in members[0] if k != 'seed'}))
    contrasts = []
    indexed = {r['id']: r for r in rows}
    for a, b in [('F2', 'F1'), ('F4', 'F3'), ('F6', 'F5'), ('F8', 'F7'), ('F10', 'F9')]:
        per_seed = []
        for seed in [1, 2]:
            aa = indexed[f'round2-P2-{a}-s{seed}']['arms'][0]
            bb = indexed[f'round2-P2-{b}-s{seed}']['arms'][0]
            for am, bm in zip(aa['metrics'], bb['metrics']):
                assert am['value']['seed'] == bm['value']['seed']
                assert am['value']['episodes'] == bm['value']['episodes']
                assert am['value']['start_steps'] == bm['value']['start_steps']
            per_seed.append(dict(seed=seed, primary_delta_pp=aa['primary'] - bb['primary'], fresh_delta_pp=aa['fresh'] - bb['fresh']))
        contrasts.append(dict(a=a, b=b, seeds=per_seed, primary_delta_pp=mean(r['primary_delta_pp'] for r in per_seed),
                              fresh_delta_pp=mean(r['fresh_delta_pp'] for r in per_seed)))
    report = dict(groups=grouped, p2_vs_historical_tf24982=comparisons, rho_contrasts=contrasts,
                  inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [performance, historical]},
                  scope='All final nominal models retained. No best-of checkpoints. Historical TF is not a contemporary rerun; '
                        'matched24982 updates/cases do not imply equal FLOPs or identical initialization. '
                        'The historical two-epoch primary bank contains42/43 only; new three-seed primary averages are not substituted. '
                        'These are exploratory means, not selection-corrected superiority or training-seed uncertainty estimates.')
    with output.open('x') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    for row in sorted(grouped, key=lambda r: (r['track'], r['block'], r['arm'], r['variant'])):
        print(f"{row['track']}/{row['block']}/{row['arm']} {row['steps']} {row['variant']}: {row['primary']:.2f}/{row['fresh']:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['performance', 'historical', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    main(args.performance, args.historical, args.output)
