"""Create all primary recipes; activate only explicit finite stage lists."""
import argparse
import copy
from pathlib import Path

from common import B, read, write, sha, source_check
from catalog import catalog


SMOKES = ['round2-P2-F7-s1', 'round2-P2-F2-s1', 'round2-P2-F3-s1', 'round2-P1-L0-s1',
          'round2-T-T4-s1', 'round2-K-K1-s1', 'round2-K-K3-s1', 'round2-K-K5-s1', 'round2-P4-I2-s1',
          'decoder_free-A-b0.3-l1-s1', 'decoder_free-B-l1-om-s1', 'decoder_free-D-win-l1-s1', 'decoder_free-T-df-s1',
          'round2-P2-F7-s2']


def main(root, source):
    root = Path(root)
    assert not root.exists()
    source = read(source)
    source_check(source)
    prior = read(B / 'runs/branch_trials_20261102/protocol.json')
    inv = read(B / 'reports/lingjing_fullspec_20261102/reference-inventory-v1.json')
    library = catalog()
    root.mkdir()
    references = {}
    for seed in range(1, 9):
        paths = sorted({str(Path(r['run']) / 'initial-model') for r in inv['rows']
                        if r['family'] == 'lower_beta' and r['seed'] == seed and r.get('initial')})
        assert paths, seed
        references[seed] = paths[0]
    rows = []
    for recipe in library['recipes']:
        task, seed = recipe['task'], recipe['seed']
        run = root / recipe['id']
        # Keep the established data/scalers, trainer and task environment.
        template = next(r for r in prior['runs'] if r['task'] == task and r['seed'] == (seed if seed in [1, 2] else 1))
        cfg = copy.deepcopy(read(Path(template['run']) / 'config.json'))
        for key in ['initial_encoder_checkpoint', 'initial_encoder_scope', 'frozen_encoder_sha256', 'audit_expectations',
                    'calibrate_state_std', 'benchmark_anchor_rule', 'gap', 'goal_query_reference', 'goal_query_reference_mode']:
            cfg.pop(key, None)
        cfg.update(output=str(run), seed=seed, model_class=recipe['model_class'], model=recipe['model'],
                   max_steps=152650 if recipe['model_class'] == 'base' else 124910,
                   max_seconds=31536000, checkpoint_every=1000,
                   record_initial_weights=True, export_initial=True, trace_batches=True, record_runtime_source=True,
                   retain_steps=sorted(set([200, 1000, 2000, 6000, 12000, 11671, 23342, 24982, 30530, 37473, 45795, 124910, 152650])),
                   all_phase_data=True, ramp_steps=0, stop_after=None)
        if isinstance(recipe['calibration'], (int, float)):
            cfg['calibrate_state_std'] = recipe['calibration']
        if task == 'pusht':
            cfg.pop('dataset_kind', None)
            cfg['data'] = str(B / 'data/pusht-lance-full-v1')
            cfg['evaluation_data'] = str(B / 'data/pusht-lance-full-v1/phase0')
            cfg['gap'] = cfg['model']['anchor_gap']
            if seed in references:
                cfg['initial_encoder_checkpoint'] = references[seed]
                cfg['initial_encoder_scope'] = 'Exact lower-beta RANDOM initial encoder; no trained/pretrained weights.'
                cfg['frozen_encoder_sha256'] = sha(Path(references[seed]) / 'weights.pt')
        else:
            gap = prior['gap_selection']['gaps'][task] if recipe['anchor'] == 'past' else 0
            cfg.update(benchmark_anchor_rule='past_gap', gap=gap)
            cfg['model']['anchor_gap'] = gap
        if recipe['model_class'] == 'base':
            cfg['audit_expectations'] = dict(gradients=dict(encoder='active', predictor='active', renderer='optional'))
        write(run / 'config.json', cfg)
        rows.append(dict(recipe, run=str(run), config_sha256=sha(run / 'config.json'),
                         dependencies=['P3a original_std'] if recipe['calibration'] == 'P3a_original_std' else []))
    assert len(rows) == 135 and all(any(r['id'] == key for r in rows) for key in SMOKES)
    jobs = [dict(next(r for r in rows if r['id'] == key), host=['dev2', 'dev13', 'dev15'][i % 3], target_step=200)
            for i, key in enumerate(SMOKES)]
    protocol = dict(source=source, runs=rows, jobs=jobs, environment=prior['environment'],
                    manifests=prior['manifests'], manifest_sha256=prior['manifest_sha256'],
                    gap_selection=prior['gap_selection'], reference_initial_encoders=references,
                    primary_recipes=135, activated_smokes=len(jobs), active_stage='technical smoke200 only',
                    automatic_long_training=False, continuous_monitor=False, original_goal='paused',
                    gate_policy=library['gate_policy'], conditional=library['conditional'])
    write(root / 'protocol.json', protocol)
    write(root / 'coverage.json', library)
    print(dict(primary_recipes=len(rows), smoke_jobs=len(jobs)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    main(a.root, a.source)
