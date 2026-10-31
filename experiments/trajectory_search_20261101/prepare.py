"""Prepare a held 24-endpoint controller comparison on four mature Cube models."""
import argparse
from pathlib import Path
from common import B, PY, read, sha, source_check, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--source-record', required=True)
    p.add_argument('--bank-report', required=True)
    p.add_argument('--contracts', required=True)
    a = p.parse_args()
    root = Path(a.root)
    assert not root.exists()
    helper = read(a.source_record)
    source_check(helper)
    bank = read(a.bank_report)
    assert bank['passed'] and bank['no_validation_test_overlap'] and bank['count'] == 1024
    assert sha(bank['bank']) == bank['bank_sha256']
    contract = read(a.contracts)
    assert contract['passed'] and contract['injected_candidates_drive_actual_native_elite_update']
    oldroot = B / 'runs/cube_maturity23342_20261031'
    old = read(oldroot / 'plan.json')
    jobs, specs = [], []
    checked = set()
    for family, slug in [('native', 'native-lewm'), ('tf', 'temporal-forecast')]:
        for seed in (1, 2):
            for case_set in ('primary', 'fresh'):
                oldid = f'original-{case_set}-cube-{slug}-seed{seed}'
                prior = next(j for j in old['jobs'] if j['id'] == oldid)
                assert read(oldroot / 'exits' / (oldid + '.json'))['returncode'] == 0
                source = prior['source']
                if source['source_commit'] not in checked:
                    source_check(source); checked.add(source['source_commit'])
                args = prior['command']
                checkpoint = args[args.index('--checkpoint') + 1]
                manifest = args[args.index('--manifest') + 1]
                data = args[args.index('--data') + 1]
                assert read(manifest)['task'] == 'cube' and len(read(manifest)['pairs']) == 50
                assert bank['dataset_card_sha256'] == sha(Path(data) / 'dataset_card.json')
                reference = Path(prior['output']) / 'metrics.json'
                evaluator = Path(source['source']) / 'as_jepa_tools/benchmark_evaluate.py'
                for mode in ('original', 'random', 'retrieval'):
                    name = f'{family}-seed{seed}-{case_set}-{mode}'
                    inputs = [Path(checkpoint) / 'weights.pt', Path(checkpoint) / 'config.json',
                              Path(manifest), Path(data) / 'dataset_card.json', reference, evaluator,
                              Path(a.bank_report), Path(bank['bank']), Path(a.contracts)]
                    immutable = {str(path): sha(path) for path in inputs}
                    s = dict(id=name, task='cube', family=family, seed=seed, case_set=case_set, mode=mode,
                        helper_source=helper, model_source=source, checkpoint=checkpoint, manifest=manifest,
                        data=data, output=str(root / name), bank_report=a.bank_report, immutable_inputs=immutable,
                        native_evaluator=str(evaluator), reference_metrics=str(reference))
                    path = root / 'specs' / (name + '.json'); write(path, s); specs.append(s)
                    i = len(jobs)
                    jobs.append(dict(id=name, host=['dev2', 'dev13', 'dev15'][i // 8], gpu=i % 8, hold=True,
                        output=s['output'], spec=str(path), spec_sha256=sha(path), depends_on=[],
                        command=[PY, '-B', '-u', str(Path(helper['source']) / 'experiments/trajectory_search_20261101/evaluate.py'), '--spec', str(path)],
                        environment=dict(PYTHONPATH=source['pythonpath'], PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                                         MKL_NUM_THREADS='2', MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0')))
    assert len(jobs) == 24
    write(root / 'candidate-plan.json', dict(root=str(root), jobs=jobs))
    write(root / 'protocol.json', dict(helper_source=helper, bank_report=a.bank_report, bank_sha256=bank['bank_sha256'],
          contracts=a.contracts, specs=specs, jobs=24, cases_per_endpoint=50, models=4,
          per_host_wall_seconds=1200, training_steps=0, new_model_parameters=0,
          comparison='Original vs64 random training segments vs64 RGB-nearest start/goal training segments.',
          budgets='Same native300 candidates x30 iterations, top30, history3/H5/block5/receding5/50 raw steps; replace first-iteration indices1..64 only.',
          controls='All methods retain current mean candidate0 and235 other native first-bank candidates. RNG draws, all later solver code, action scalers, goal cost, returned mean and execution are native.',
          bank='1024 distinct train episodes, one uniform valid window/episode, nativeRGB224 pooled8x8; recorded25-step sequence and its true start/end. No optimality labels or validation/test/library tuning.',
          analysis='Complete all24 actual exits before outcome analysis. Retain both seeds, original primary/fresh50, all modes; report retrieval-original and retrieval-random case-paired effects, no best-of controller selection.',
          preflight='CPU native elite/RNG contracts plus a single first-case retrieval run for each model family; inspect execution only before all24 task results.',
          continuation='No automatic training, library/feature/K sweep or further planner change. Gains must survive both case sets and compare against both models with the same controller.'))
    (root / 'HOLD_DISPATCH').write_text('Prepared only; root source/runtime review and actual GPU-free checks required.\n')
    write(root / 'preparation.json', dict(passed=True, root=str(root), jobs=len(jobs),
          protocol_sha256=sha(root / 'protocol.json'), plan_sha256=sha(root / 'candidate-plan.json'),
          bank_sha256=bank['bank_sha256'], source_commits=sorted(checked), helper_commit=helper['source_commit']))
    print('Prepared24 held matched endpoints.', flush=True)


if __name__ == '__main__':
    main()
