"""Six matched temporal-forecast controls using actual Gaussian prior draws."""
import argparse
import copy
from pathlib import Path

from common import read, sha, source_check, write


def main():
    p = argparse.ArgumentParser()
    for key in ['parent', 'root', 'source', 'cpu']:p.add_argument('--' + key, required=True)
    a = p.parse_args();parent, root = Path(a.parent), Path(a.root);assert not root.exists()
    old, source, cpu = read(parent / 'protocol.json'), read(a.source), read(a.cpu);source_check(source)
    assert cpu['passed'] and len(cpu['full_actual_model_regressions']) == 3 and not cpu['CUDA_initialized']
    runs = []
    for row in old['runs']:
        if row['arm'] != 'tf':continue
        cfg = copy.deepcopy(read(Path(row['run']) / 'config.json'))
        assert read(parent / 'worker-records' / (row['id'] + '-gate.json'))['passed']
        assert cfg['model_class'] == 'robust' and cfg['model']['forecast_fraction'] == .5
        assert cfg['model'].get('rate_prior', 'one-step') == 'one-step' and cfg['model'].get('posterior_noise_scale', 1.) == 1.
        cfg['output'] = str(root / row['id']);cfg['model']['sample_forecast'] = True
        new = dict(copy.deepcopy(row), run=cfg['output'], gpu=5+row['seed'],
                   parent_config_sha256=sha(Path(row['run']) / 'config.json'))
        path = Path(cfg['output']) / 'config.json';write(path, cfg);new['config_sha256'] = sha(path);runs.append(new)
    assert len(runs) == 6
    for task in ['pusht', 'tworoom', 'cube']:write(root / 'cohorts' / (task + '.json'), read(parent / 'cohorts' / (task + '.json')))
    write(root / 'source-record.json', source)
    write(root / 'protocol.json', dict(old, source=source, runs=runs, parent=str(parent),
          cpu_contract_sha256=sha(a.cpu), preflight_stop=2, short_gate_stop=600,
          review_required_before600=True, automatic_long_training=False,
          note='Only forecast-render branch draws fresh independent sigma0 innovations from the existing Gaussian transition prior. '
               'Same51M architecture,posterior AR noise,beta,forecast_fraction,data,CEM/inference means. '
               'Independent private innovation stream preserves legacy outer RNG/dropout/mask/image noise. '
               'Not a total-ELBO,new theorem or novelty claim;must test all3 tasks/two seeds.'))
    print('Prepared6 Gaussian-prior-sampled native2-update gates', root)


if __name__ == '__main__':main()
