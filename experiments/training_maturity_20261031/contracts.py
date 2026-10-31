"""Small CPU orchestration contracts; actual12 native restores are separate."""
import copy
from pathlib import Path
from common import CAP, FAMILIES, PREFIX, allocations, stages, semantic, sha, write
from prepare import configure, jobs_for, validate_jobs


def main():
    directory = Path(__file__).resolve().parent
    jobs = []
    for r in allocations():
        family = r['family']
        old = dict(output='/old', batch_size=128, max_steps=124910, seed=r['training_seed'],
                   model=dict(window=8, action_dim=25, marker='all-original-options-retained'),
                   checkpoint_every=2000, retain_steps=[r['initial_step']])
        new = configure(old, '/new', family)
        assert semantic(new) == semantic(old) and old['retain_steps'] == [r['initial_step']]
        r.update(config_sha256='expected', source=dict(source='/immutable-model', source_commit=FAMILIES[family]['commit']),
                 parent='/old', initial='/initial/weights.pt', model_record='/source.json', data='/old-data', windows='/same64.json',
                 references={k: dict(manifest='/'+k+'.json', manifest_sha256=k) for k in ['primary', 'fresh']})
        jobs.extend(jobs_for(Path('/candidate'), dict(source='/immutable-helper'), r))
    result = validate_jobs(jobs)
    assert CAP == 2*11671 and 128*11671 == 1493888
    assert set((r['host'], r['gpu']) for r in allocations()) == {(h, g) for h in ['dev2', 'dev13', 'dev15'] for g in [0, 1]}
    assert all('as_jepa_tools.train_stage' in j['command'] and '--resume' in j['command'] for j in jobs if j['phase'].startswith('learn'))
    assert not any('calibrat' in j['phase'] or 'ngc' in j['phase'] for j in jobs)
    assert all(j['command'][j['command'].index('--expected-step')+1] == str(CAP) for j in jobs if j['phase'] == 'fullstate')
    assert all(j['command'][j['command'].index('--cases')+1] == '64' for j in jobs if j['phase'] == 'transition')
    assert all(j['command'][j['command'].index('--cases')+1] == '16' for j in jobs if j['phase'] == 'health')
    bad = copy.deepcopy(jobs); bad[-1]['depends_on'] = []
    try: validate_jobs(bad)
    except AssertionError: pass
    else: raise AssertionError('Broken dependency chain accepted')
    bad = copy.deepcopy(jobs); bad[0]['source']['source_commit'] = 'wrong'
    try: validate_jobs(bad)
    except AssertionError: pass
    else: raise AssertionError('Different model source accepted')
    result.update(helper_sha256={n: sha(directory/n) for n in ['common.py', 'prepare.py', 'audit.py', 'contracts.py']},
                  no_models_imported=True, no_GPU=True, held_only=True,
                  actual_restore='Separate12 real pre/post CPU model/AdamW/scheduler audits are mandatory before root review; this is a DAG/config contract only.')
    write(directory/'cpu-contracts.json', result)
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__': main()
