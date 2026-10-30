"""Aggregate CPU query support; no dataset reads, training or GPU use."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ordered_query_checks import transfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.input).read_text())
    representatives = {'I': (), 'A': (0,), 'B': (1,), 'AB': (0, 1), 'BA': (1, 0)}
    lookup = {tuple(transfer(word).ravel()): name for name, word in representatives.items()}
    names = list(representatives)
    # The five-element set is closed under every pair of compositions.
    table = {left:{right:lookup[tuple((transfer(lw)@transfer(rw)).ravel())]
                   for right,rw in representatives.items()} for left,lw in representatives.items()}
    results = {}
    for task, row in data['real_data'].items():
        counts = {part:Counter() for part in ['whole', 'first_half', 'second_half']}
        for record in row['records']:
            word = record['word']
            for part, segment in [('whole', word), ('first_half', word[:4]), ('second_half', word[4:])]:
                counts[part][lookup[tuple(transfer(segment).ravel())]] += 1
        halves = counts['first_half']+counts['second_half']
        results[task] = {key:value for key,value in row.items() if key!='records'}
        results[task]['monoid_counts'] = {part:{name:counter[name] for name in names} for part,counter in counts.items()}
        results[task]['half_monoid_counts'] = {name:halves[name] for name in names}
        # All of these are negative under scalar completion, but represent distinct transformations.
        results[task]['half_incomplete_but_contains_B_fraction'] = (halves['B']+halves['BA'])/sum(halves.values())
        results[task]['half_incomplete_without_B_fraction'] = (halves['I']+halves['A'])/sum(halves.values())
    summary = dict(constructions=data['constructions'], monoid_multiplication=table, real_data=results)
    (target/'ordered-query-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Ordered query: CPU evidence', '',
             'No model was trained. The labels use only existing RGB frames. Full implementation contract and prior-art limits: `S3_REFINED.md`.', '',
             'The exhaustive oracle checks all 3,280 words over {A,B,O} through length7: oracle, composition and hard-symbol stuttering errors are all0. The five matrices are closed under all25 compositions. A correlated AA/BB mixture gives true ordered-hit0; independently multiplying its marginals gives0.25. This rejects a naive marginal-hazard implementation.', '',
             '| Task | Queries | Interior shuffle changes result | Interior reversal changes result | Duplicate prototypes |',
             '|---|---:|---:|---:|---:|']
    for task, row in results.items():
        lines.append(f"| {task} | {row['queries']} | {100*row['endpoint_preserving_shuffle_transfer_change_fraction']:.2f}% | {100*row['endpoint_preserving_reverse_transfer_change_fraction']:.2f}% | {100*row['degenerate_prototype_fraction']:.2f}% |")
    lines += ['', 'Each task uses128 existing training windows,8 random prototype queries per window. Initial/final symbols are held fixed during the diagnostic shuffle/reversal. The perturbed symbol paths are never treated as action-conditioned training outcomes.', '',
              '**Necessary correction:** selecting all3 prototype frames from a complete window guarantees their occurrence, so its full-window target has only AB/BA classes. There the full transfer adds no information over scalar completion. The proposed objective therefore includes its2 factual half-segments under the same query. Their targets do contain transformations collapsed by one scalar.', '',
              '| Task / segment | I | A | B | AB | BA |', '|---|---:|---:|---:|---:|---:|']
    for task,row in results.items():
        for segment in ['whole', 'first_half', 'second_half']:
            values = row['monoid_counts'][segment]
            lines.append('| '+task+' / '+segment+' | '+' | '.join(str(values[name]) for name in names)+' |')
    lines += ['', 'Under a scalar completion target I,A,B,BA are all0. Under an already-A prefix, B and BA complete the query while I and A do not. Across half-segments, both kinds actually occur:', '']
    for task,row in results.items():
        lines.append(f"- {task}: incomplete but contains B = {100*row['half_incomplete_but_contains_B_fraction']:.2f}% of halves; incomplete without B = {100*row['half_incomplete_without_B_fraction']:.2f}%.")
    lines += ['', 'This verifies an available compositional target distinction, not physical event semantics, a causal action effect, learning superiority or absolute novelty. Episode-level dependence remains; query counts are not independent experimental seeds. All targets and controls still need a native-planner gate before a performance claim.']
    (target/'ORDERED_QUERY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, (task,row) in zip(axes,results.items()):
        x = np.arange(len(names))
        whole = row['monoid_counts']['whole']
        halves = row['half_monoid_counts']
        ax.bar(x-.18, [100*whole[n]/row['queries'] for n in names], .36, label='Whole window')
        ax.bar(x+.18, [100*halves[n]/(2*row['queries']) for n in names], .36, label='Factual half-segments')
        ax.set_xticks(x,names)
        ax.set_title(task)
        ax.set_ylim(0, 80)
        ax.set_ylabel('Fraction of targets (%)')
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle('Full-window targets lose the compositional distinction; half-segments recover it')
    fig.savefig(target/'ordered-query-support.png',dpi=160)
    fig.savefig(target/'ordered-query-support.svg')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
