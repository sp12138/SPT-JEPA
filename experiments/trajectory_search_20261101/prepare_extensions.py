"""Freeze a new-case Cube confirmation or the identical TwoRoom controller test."""
import argparse
import copy
from pathlib import Path
from common import B, PY, read, sha, source_check, write


def new_cube_manifest(root):
    import numpy as np
    import clear_lewm.manifests as manifests
    import clear_lewm.tasks as tasks
    from clear_lewm.protocols import protocol_from_dict
    base_root = B / 'runs/trajectory_search_20261101'
    old = read(base_root / 'candidate-plan.json')
    seen = set()
    for j in old['jobs']:
        seen.update(x['episode_id'] for x in read(read(j['spec'])['manifest'])['pairs'])
    data = B / 'data/hardtasks-v1/cube-metadata'
    train = np.load(data / 'train_episodes.npy')
    allowed = np.setdiff1d(np.load(data / 'validation_episodes.npy'), list(seen))
    assert len(allowed) >= 300
    original = read(read(old['jobs'][0]['spec'])['manifest'])
    protocol = protocol_from_dict(original['protocol'])
    # Preserve the already frozen public geometry/validity sampler. Only the
    # allowed validation episodes and predetermined new sampling seed differ.
    def read_rows(dataset, key, rows):
        unique, inverse = np.unique(rows, return_inverse=True)
        return np.asarray(dataset[key][unique])[inverse]
    tasks._read_pair = lambda ds, key, starts, goals: (read_rows(ds, key, starts), read_rows(ds, key, goals))
    manifests.split_episode_ids = lambda episodes, fraction, seed: (train, allowed)
    card = read(data / 'dataset_card.json')
    result = manifests.generate_manifest(card['h5_path'], 'cube', protocol, 50, 2045, split='heldout')
    assert set(x['episode_id'] for x in result['pairs']).issubset(set(allowed.tolist()))
    assert len(result['pairs']) == 50 and result['protocol'] == original['protocol']
    result.update(campaign_split='validation', policy_seed=42, case_sampling_seed=2045,
          campaign_scope='Predetermined new validation cases after trajectory-search screen;all source episodes disjoint from primary/fresh. No tuning/test episodes.',
          metadata_card_sha256=sha(data / 'dataset_card.json'))
    path = root / 'newcases.json'; write(path, result)
    write(root / 'newcase-provenance.json', dict(seed=2045, cases=50, distinct_episodes=len({x['episode_id'] for x in result['pairs']}),
          excluded_previous_episodes=sorted(seen), eligible_remaining_episodes=len(allowed),
          no_train_test_overlap=True, disjoint_from_previous_primary_fresh=True,
          protocol_exact=True, manifest_sha256=sha(path)))
    return str(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['cube-newcases', 'tworoom'], required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--source-record', required=True)
    p.add_argument('--bank-report', required=True)
    p.add_argument('--contracts', required=True)
    a = p.parse_args()
    root = Path(a.root); assert not root.exists(); root.mkdir()
    helper = read(a.source_record); source_check(helper)
    bank = read(a.bank_report)
    assert bank['passed'] and bank['no_validation_test_overlap'] and sha(bank['bank']) == bank['bank_sha256']
    contracts = read(a.contracts)
    assert contracts['passed'] and contracts['validated_action_widths'] == [10, 25]
    task = 'cube' if a.task == 'cube-newcases' else 'tworoom'
    assert bank['task'] == task
    template = read(B / 'runs/cube_maturity23342_20261031/plan.json')
    if task == 'cube':
        manifest = new_cube_manifest(root)
        old = read(B / 'runs/trajectory_search_20261101/candidate-plan.json')
        bases = [read(j['spec']) for j in old['jobs'] if j['id'].endswith('-primary-original')]
        assert len(bases) == 4
    else:
        old = read(B / 'runs/public_ngc_stable_evaluation_v2_20261031/plan.json')
        bases = []
        for j in old['jobs']:
            s = read(j['spec'])
            if s['task'] != task or s['mode'] != 'l2' or s['dimension'] not in (64, 192):
                continue
            s['family'] = 'tf' if s['dimension'] == 64 else 'native'
            s['reference_metrics'] = str(Path(j['output']) / 'metrics.json')
            bases.append(s)
        assert len(bases) == 8
    slots = [('dev2', g) for g in range(2, 8)] + [('dev13', g) for g in range(2, 8)] + [('dev15', g) for g in range(8)]
    jobs = []
    for base in bases:
        case = 'newcases' if task == 'cube' else base['case_set']
        for mode in ('original', 'random', 'retrieval'):
            s = copy.deepcopy(base)
            name = f'{s["family"]}-seed{s["seed"]}-{case}-{mode}'
            s.update(id=name, task=task, helper_source=helper, mode=mode, case_set=case,
                     output=str(root / name), bank_report=a.bank_report)
            if task == 'cube':
                s['manifest'] = manifest
            inputs = [Path(s['checkpoint']) / 'weights.pt', Path(s['checkpoint']) / 'config.json',
                      Path(s['manifest']), Path(s['data']) / 'dataset_card.json', Path(s['reference_metrics']),
                      Path(s['native_evaluator']), Path(a.bank_report), Path(bank['bank']), Path(a.contracts)]
            s['immutable_inputs'] = {str(path): sha(path) for path in inputs}
            path = root / 'specs' / (name + '.json'); write(path, s)
            host, gpu = slots[len(jobs) % len(slots)]
            jobs.append(dict(id=name, host=host, gpu=gpu, hold=True, output=s['output'],
                  spec=str(path), spec_sha256=sha(path), depends_on=[],
                  command=[PY, '-B', '-u', str(Path(helper['source']) / 'experiments/trajectory_search_20261101/evaluate.py'), '--spec', str(path)],
                  environment=dict(PYTHONPATH=s['model_source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1',
                                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')))
    assert len(jobs) == (12 if task == 'cube' else 24)
    write(root / 'candidate-plan.json', dict(root=str(root), environment=template['environment'], jobs=jobs))
    write(root / 'protocol.json', dict(task=task, endpoints=len(jobs), helper_source=helper,
          bank_report=a.bank_report, bank_sha256=bank['bank_sha256'], fixed_replacements=64,
          no_tuning_or_extra_optimal_action_labels=True, training_steps=0,
          scope='Exact same train-only RGB retrieval/random-library/native controller and planning budgets. Both seeds and all models retained;newcases or cross-task confirmation,not a new learned objective.',
          reserved_slots=[['dev2', 0], ['dev2', 1], ['dev13', 0], ['dev13', 1]],
          activation='Root review,actual-free check and one-case TwoRoom runtime contract before dispatch. No continuous monitor.'))
    write(root / 'preparation.json', dict(passed=True, endpoints=len(jobs), plan_sha256=sha(root / 'candidate-plan.json'),
          protocol_sha256=sha(root / 'protocol.json'), helper_commit=helper['source_commit']))
    (root / 'HOLD_DISPATCH').write_text('Prepared only;root concrete review and actual-free check required.\n')
    print(f'Prepared{len(jobs)} held {task} confirmation endpoints.', flush=True)


if __name__ == '__main__':
    main()
