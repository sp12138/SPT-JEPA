"""One common full-capacity spatial predictor;six native2-update gates."""
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
    assert cpu['passed'] and cpu['all_six_native_blocks_and_condition_tensors_exact'] and cpu['no_CUDA_reseed']
    assert cpu['new_predictor_parameters'] == 11001104
    runs = []
    for row in original['runs']:
        if row['arm'] != 'flat':continue
        assert read(parent / 'worker-records' / (row['id'] + '-gate.json'))['passed']
        cfg = copy.deepcopy(read(Path(row['run']) / 'config.json'))
        ident = f'{row["task"]}-field-seed{row["seed"]}'
        cfg['output'], cfg['model_class'] = str(root / ident), 'spatial_native'
        cfg['model']['spatial_predictor_seed'] = row['seed']
        cfg.setdefault('audit_expectations', {}).setdefault('gradients', {})['action_encoder'] = 'active'
        new = dict(copy.deepcopy(row), id=ident, run=cfg['output'], arm='field', gpu=2 if row['seed'] == 1 else 5,
                   parent_flat_id=row['id'], parent_config_sha256=sha(Path(row['run']) / 'config.json'))
        path = Path(cfg['output']) / 'config.json';write(path, cfg);new['config_sha256'] = sha(path);runs.append(new)
    assert len(runs) == 6
    for task in ['pusht', 'tworoom', 'cube']:write(root / 'cohorts' / (task + '.json'), read(parent / 'cohorts' / (task + '.json')))
    write(root / 'source-record.json', source)
    write(root / 'protocol.json', dict(original, source=source, runs=runs, parent=str(parent),
          cpu_contract_sha256=sha(a.cpu), preflight_stop=2, short_gate_stop=600,
          review_required_before600=True, automatic_long_training=False,
          note='Full native192hidden/6layer/16head/2048MLP body applied to64 spatial tokens;only input/output projections replaced. No new loss/data/goal metric. All native body initial tensors match flat;compute/dropout streams differ. Not the held523536-parameter h64 patch branch.'))
    print('Prepared6 full-capacity spatial native2 gates', root)


if __name__ == '__main__':main()
