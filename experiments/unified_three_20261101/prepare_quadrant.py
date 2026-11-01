"""One common local-render query control, only six native2-step gates."""
import argparse
import copy
from pathlib import Path

from common import read, sha, source_check, write


def main():
    parser = argparse.ArgumentParser()
    for key in ['parent', 'root', 'source', 'cpu']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    parent, root = Path(args.parent), Path(args.root)
    assert not root.exists()
    old, source, cpu = read(parent / 'protocol.json'), read(args.source), read(args.cpu)
    source_check(source)
    assert cpu['passed'] and len(cpu['full_actual_model_regressions']) == 3
    assert cpu['all_original_pixels_covered_once']
    extra = {r['extra_parameters'] for r in cpu['full_actual_model_regressions']}
    assert len(extra) == 1
    extra = extra.pop()
    runs = []
    for row in old['runs']:
        if row['arm'] != 'tf':
            continue
        cfg = copy.deepcopy(read(Path(row['run']) / 'config.json'))
        cfg['output'] = str(root / row['id'])
        cfg['model_class'] = 'quadrant_render'
        new = dict(copy.deepcopy(row), run=cfg['output'], gpu=2 if row['seed'] == 1 else 5,
                   parent_config_sha256=sha(Path(row['run']) / 'config.json'))
        path = Path(cfg['output']) / 'config.json'
        write(path, cfg); new['config_sha256'] = sha(path); runs.append(new)
    assert len(runs) == 6
    for task in ['pusht', 'tworoom', 'cube']:
        write(root / 'cohorts' / (task + '.json'), read(parent / 'cohorts' / (task + '.json')))
    write(root / 'source-record.json', source)
    write(root / 'protocol.json', dict(old, source=source, runs=runs, parent=str(parent),
          cpu_contract_sha256=sha(args.cpu), preflight_stop=2, short_gate_stop=600,
          review_required_before600=True, automatic_long_training=False,
          note='Same global64D TF encoder/predictor;sample-level FM renders one uniform quadrant at64px '
               'with its matching local anchor and known2D address. Full224px RGB already existed in both arms. '
               f'{extra} extra renderer-only query parameters;all550 original initial tensors and outer RNG exact. '
               'No physical labels/action targets/prior-sampling/high-noise change or task-specific setting. '
               'No novel-method or expected task-success claim.'))
    print('Prepared6 local-query technical gates', root)


if __name__ == '__main__':
    main()
