"""Eight frozen-model rescoring jobs on the same old Cube executed pool."""
import argparse
import copy
import importlib.util
from pathlib import Path

import torch

from common import B, PY, read, sha, source_check, write


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True);a = p.parse_args()
    root = Path(a.root);assert not root.exists();torch.set_num_threads(2)
    old = B / 'runs/cube_pool_reuse2000_20261031'
    parent = B / 'runs/unified_three_20261101'
    direct = B / 'runs/unified_direct600_20261101'
    template = read(old / 'specs/local6000-seed1.json')
    helper = template['helper_source']
    runner = Path(helper['source']) / 'experiments/cube_pool_reuse_20261031/assay.py'
    assert sha(runner) == template['helper_sha256']
    spec = importlib.util.spec_from_file_location('unchanged_pool_assay', runner)
    assay = importlib.util.module_from_spec(spec);spec.loader.exec_module(assay)
    inventory = read(template['inventory'])
    assert sha(template['inventory']) == template['inventory_sha256']
    for case in inventory['cases_provenance']:assert sha(case['path']) == case['sha256']
    current = read(parent / 'evaluations/cube-flat-seed1/metrics.json')
    assert current['pairs'] == [x['pair'] for x in inventory['cases_provenance']]
    rows, jobs, verified = [], [], set()
    for label, family, budget, base, host in [
        ('flat2000', 'flat', 2000, parent, 'dev2'),
        ('field2000', 'field', 2000, parent, 'dev13'),
        ('field600', 'field', 600, parent, 'dev15'),
        ('direct600', 'field', 600, direct, 'dev15')]:
        source = read(base / 'source-record.json')
        for src in [source, helper]:
            if src['source'] not in verified:source_check(src);verified.add(src['source'])
        for seed in [1, 2]:
            ident = f'{label}-seed{seed}';runid = f'cube-{family}-seed{seed}'
            cfgpath = base / runid / 'config.json';cfg = read(cfgpath)
            weights = base / runid / f'model-step{budget:07d}/weights.pt'
            auditpath = base / 'audits' / f'{runid}-step{budget}/audit.json';audit = read(auditpath)
            assert audit['passed'] and sha(weights) == audit['portable_sha256']
            s = copy.deepcopy(read(old / f'specs/local6000-seed{seed}.json'))
            raw = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
            assay.geometry_contract(cfg, raw, s['encoder_geometry_sha256']);del raw
            changes = dict(id=ident, kind=label, model_source=source,
                           config=str(cfgpath), config_sha256=sha(cfgpath), weights=str(weights),
                           weights_sha256=sha(weights), output=str(root / ident))
            s.update(changes)
            assert not s['encoder_owner'] and s['budget'] == dict(rgb_frames=0, encoder=0, predict=250, action_encoder=50, renderer=0)
            path = root / 'specs' / (ident + '.json');write(path, s)
            jobs.append(dict(id=ident, host=host, gpu=5+seed, spec=str(path), spec_sha256=sha(path),
                             command=[PY, '-B', '-u', str(runner), 'run', '--spec', str(path)],
                             output=s['output'], depends_on=[]))
            rows.append(dict(id=ident, budget=budget, model_audit=str(auditpath), model_audit_sha256=sha(auditpath),
                             matched_geometry_tensors=201, source=source['source'], weights_sha256=s['weights_sha256']))
    write(root / 'protocol.json', dict(passed=True, jobs=jobs, bindings=rows, helper_source=helper,
          model_calls_during_preparation=0, inventory_sha256=sha(template['inventory']),
          total_calls=dict(predict=2000, action_encoder=400, encoder=0, renderer=0, rgb_frames=0),
          matched_contrasts=[['field2000', 'flat2000'], ['direct600', 'field600']],
          scope='Same old50x39 raw action bank and frozen encodings;nonoracle22 primary,recorded17 secondary. '
                'Zero new simulation/encoding/training/physical-label fitting. ColdH1,T5 diagnostic,not currentCEM or50-step MPC coverage. '
                'Both seeds/allcases retained;do not mix600 and2000 as a matched exposure comparison.'))
    print('Prepared8 zero-encoder/zero-simulation frozen counterfactual rescoring jobs', root)


if __name__ == '__main__':main()
