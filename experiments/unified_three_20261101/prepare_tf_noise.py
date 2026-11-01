"""The single fixed noise setting on learned TF, with/without prior draws."""
import argparse
import copy
from pathlib import Path

from common import read, sha, source_check, write


def main():
    parser = argparse.ArgumentParser()
    for key in ['parent', 'root', 'source', 'cpu']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--sampled', action='store_true')
    args = parser.parse_args()
    parent, root = Path(args.parent), Path(args.root)
    assert not root.exists()
    old, source, cpu = read(parent / 'protocol.json'), read(args.source), read(args.cpu)
    source_check(source)
    assert cpu['passed'] and len(cpu['full_actual_model_regressions']) == 3
    runs = []
    for row in old['runs']:
        if row['arm'] != 'tf':
            continue
        cfg = copy.deepcopy(read(Path(row['run']) / 'config.json'))
        assert cfg['model_class'] == 'robust'
        assert cfg['model'].get('flow_logit_shift', 0.) == 0.
        assert not cfg['model'].get('sample_forecast', False)
        cfg['output'] = str(root / row['id'])
        cfg['model']['flow_logit_shift'] = .8
        if args.sampled:
            cfg['model']['sample_forecast'] = True
        new = dict(copy.deepcopy(row), run=cfg['output'],
                   gpu=(row['seed'] - 1) * 3 + int(args.sampled),
                   parent_config_sha256=sha(Path(row['run']) / 'config.json'))
        path = Path(cfg['output']) / 'config.json'
        write(path, cfg)
        new['config_sha256'] = sha(path)
        runs.append(new)
    assert len(runs) == 6
    for task in ['pusht', 'tworoom', 'cube']:
        write(root / 'cohorts' / (task + '.json'), read(parent / 'cohorts' / (task + '.json')))
    write(root / 'source-record.json', source)
    write(root / 'protocol.json', dict(old, source=source, runs=runs, parent=str(parent),
          cpu_contract_sha256=sha(args.cpu), preflight_stop=2, short_gate_stop=600,
          review_required_before600=True, automatic_long_training=False,
          note='Same fixed training-logit+.8 already used on frozen carriers, now on learned64D TF. '
               'Both prior means/draws are retained, completing a prespecified2x2 with ordinary TF controls. '
               'Frozen-carrier noise controls cannot test representation learning. No time sweep, '
               'inference schedule change, new parameter, loss, label, or task-specific setting.'))
    print('Prepared6 learned-TF high-noise technical gates', root, 'sampled=', args.sampled)


if __name__ == '__main__':
    main()
