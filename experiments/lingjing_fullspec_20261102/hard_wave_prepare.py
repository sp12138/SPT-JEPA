"""Finish the untested hard-task hypotheses; reuse completed canonical state gates."""
import argparse
import copy
from pathlib import Path
from common import B, read, write, sha, source_check


def main(root, helper):
    root = Path(root); assert not root.exists(); root.mkdir()
    parent = B / 'runs/lingjing_fullspec_20261102'
    p = read(parent / 'protocol.json'); helper = read(helper); source_check(p['source']); source_check(helper)
    gates = read(B / 'reports/lingjing_fullspec_20261102/gates-summary-v1.json')
    assert gates['passed'] and gates['technical_smokes'] == 14 and gates['frozen_diagnostics'] == 26
    anchors = read(B / 'reports/lingjing_fullspec_20261102/anchor-audit-v1.json'); assert anchors['passed']
    old = read(B / 'runs/branch_trials_20261102/protocol.json')
    references = {}
    for task in ['tworoom', 'cube']:
        for seed in [1, 2]:
            ref = next(r for r in old['runs'] if r['task'] == task and r['seed'] == seed and
                       ((r['family'] == 'round2' and r['arm'] == 'calibrated-past') if task == 'tworoom' else r['family'] == 'decoder'))
            cfg = read(Path(ref['run']) / 'config.json')
            references[task, seed] = dict(initial_encoder_checkpoint=cfg['initial_encoder_checkpoint'],
                                          initial_model=str(Path(ref['run']) / 'initial-model'))
    chosen = [r for r in p['runs'] if r['task'] == 'cube' or
              (r['track'] == 'round2' and r['task'] == 'tworoom' and r['arm'] not in ['T3', 'T4']) or
              (r['track'] == 'decoder_free' and r['task'] == 'tworoom' and r['arm'] != 'df')]
    assert len(chosen) == 32
    jobs = []
    def add(recipe, ident=None, changes=None):
        cfg = copy.deepcopy(read(Path(recipe['run']) / 'config.json'))
        ident = ident or recipe['id']; run = root / ident
        cfg.update(output=str(run), checkpoint_every=1000, retain_steps=[2000, 6000, 12000],
                   record_initial_weights=True, export_initial=True, record_runtime_source=True, trace_batches=False)
        reference = references[recipe['task'], recipe['seed']]
        cfg['initial_encoder_checkpoint'] = reference['initial_encoder_checkpoint']
        cfg['initial_encoder_scope'] = 'Exact original RANDOM encoder shared with archived controls; no trained/pretrained encoder.'
        cfg['frozen_encoder_sha256'] = sha(Path(cfg['initial_encoder_checkpoint']) / 'weights.pt')
        if changes:
            if 'model' in changes: cfg['model'].update(changes['model'])
            cfg.update({k: v for k, v in changes.items() if k != 'model'})
        write(run / 'config.json', cfg)
        jobs.append(dict(recipe, id=ident, run=str(run), config_sha256=sha(run / 'config.json'),
                         reference_initial=reference['initial_model'], target_step=2000,
                         conditional_diagnostic=bool(changes), host=['dev2', 'dev13', 'dev15'][len(jobs) % 3]))
    for recipe in chosen: add(recipe)
    for seed in [1, 2]:
        price = next(r for r in gates['frozen'] if r['job']['id'] == f'original2000-round2-tworoom-calibrated-past-seed{seed}-fp32')
        assert price['result']['feature_std_over_sigma0'] < .3
        recipe = next(r for r in p['runs'] if r['id'] == f'round2-T-T4-s{seed}')
        add(recipe, f'round2-T-T4-beta0-s{seed}', dict(model=dict(beta=0.)))
        trapped = next(r for r in gates['frozen'] if r['job']['id'] == f'original2000-decoder-tworoom-pooled-seed{seed}-fp32')
        assert not trapped['result']['state_gate']
        recipe = next(r for r in p['runs'] if r['id'] == f'decoder_free-T-df-s{seed}')
        add(recipe, f'decoder_free-T-df-init3-s{seed}', dict(calibrate_state_std=.3))
    assert len(jobs) == 36
    write(root / 'protocol.json', dict(source=p['source'], helper=helper, jobs=jobs, environment=p['environment'],
          manifests=p['manifests'], manifest_sha256=p['manifest_sha256'], stop_step=2000,
          reused=['Round2 T3/T4 both seeds at2000', 'DF canonical TwoRoom both seeds at2000'],
          trigger_summary_sha256=sha(B / 'reports/lingjing_fullspec_20261102/gates-summary-v1.json'),
          continuous_monitor=False, automatic_long_training=False,
          scope='32 untested primary hard-task recipes plus exactly the two predeclared paired diagnostics: T4beta0 and DFinit.3. All fresh2000 only. Full states/probes/rendered use are collected, then HOLD; no automatic6000/12000 or irreversible failure claim. Old canonical models are reused rather than retrained.'))
    print(dict(jobs=len(jobs), counts={h: sum(j['host'] == h for j in jobs) for h in ['dev2', 'dev13', 'dev15']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--helper', required=True)
    a = p.parse_args(); main(a.root, a.helper)
