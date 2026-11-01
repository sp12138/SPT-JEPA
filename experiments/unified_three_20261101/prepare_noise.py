"""One fixed high-noise control crossed with flat/full-capacity spatial bodies."""
import argparse
import copy
from pathlib import Path

from common import read, sha, source_check, write


def main():
    p = argparse.ArgumentParser()
    for key in ['parent', 'root', 'source', 'cpu']:p.add_argument('--' + key, required=True)
    a = p.parse_args(); parent, root = Path(a.parent), Path(a.root)
    assert not root.exists()
    original, source, cpu = read(parent / 'protocol.json'), read(a.source), read(a.cpu)
    source_check(source)
    assert cpu['passed'] and cpu['new_predictor_parameters'] == 11001104
    runs = []
    for row in original['runs']:
        if row['arm'] != 'flat':continue
        base = read(Path(row['run']) / 'config.json')
        assert base['model'].get('flow_logit_shift', 0.) == 0.
        for arm in ['flat', 'field']:
            cfg = copy.deepcopy(base);ident = f'{row["task"]}-{arm}-seed{row["seed"]}'
            cfg['output'] = str(root / ident)
            cfg['model']['flow_logit_shift'] = .8
            if arm == 'field':
                cfg['model_class'] = 'spatial_native'
                cfg['model']['spatial_predictor_seed'] = row['seed']
                cfg.setdefault('audit_expectations', {}).setdefault('gradients', {})['action_encoder'] = 'active'
            new = dict(copy.deepcopy(row), id=ident, run=cfg['output'], arm=arm,
                       gpu=(row['seed']-1)*3+(0 if arm == 'flat' else 1),
                       parent_flat_id=row['id'], parent_config_sha256=sha(Path(row['run']) / 'config.json'))
            path = Path(cfg['output']) / 'config.json';write(path, cfg);new['config_sha256'] = sha(path);runs.append(new)
    assert len(runs) == 12
    for task in ['pusht', 'tworoom', 'cube']:write(root / 'cohorts' / (task + '.json'), read(parent / 'cohorts' / (task + '.json')))
    write(root / 'source-record.json', source)
    write(root / 'protocol.json', dict(original, source=source, runs=runs, parent=str(parent),
          cpu_contract_sha256=sha(a.cpu), preflight_stop=2, short_gate_stop=600,
          review_required_before600=True, automatic_long_training=False,
          note='Single prior high-noise setting:training time logit+0.8,unchanged inference schedule. Cross flat/local1024 and full native spatial bodies on ALL3 tasks/seeds. No new parameter/loss/action label,K/time/feature sweep or automatic extension. Complements ordinary-noise counterparts;not an original-idea claim.'))
    print('Prepared12 high-noise two-update checks;no600 authorization', root)


if __name__ == '__main__':main()
