"""All remaining PushT primary hypotheses, with prespecified maturity and insight-first gates."""
import argparse
import copy
from pathlib import Path
from common import B, read, write, sha, source_check


def main(root, helper, cube_review=None):
    root = Path(root); assert not root.exists(); root.mkdir()
    prior = read(B / 'runs/lingjing_fullspec_20261102/protocol.json')
    helper = read(helper); source_check(prior['source']); source_check(helper)
    source = prior['source']; jobs = []
    chosen = [r for r in prior['runs'] if r['task'] == 'pusht' and not (r['track'] == 'round2' and r['block'] == 'P1')]
    assert len(chosen) == 94
    def add(recipe, cfg, stages, resume_parent=None):
        ident = recipe['id']; run = root / ident
        cfg.update(output=str(run), record_initial_weights=True, export_initial=True, record_runtime_source=True,
                   trace_batches=False, checkpoint_every=12491 if recipe['model_class'] != 'base' else 15265,
                   retain_steps=stages, stop_after=None, max_seconds=31536000)
        prefixes = ['encoder', 'predictor', 'action_encoder'] + (['projector', 'pred_proj'] if recipe['model_class'] == 'base' else ['pool'])
        gradients = {key: 'active' for key in prefixes}; gradients['renderer'] = 'active' if recipe['model_class'] == 'forecast_scored' else 'optional'
        cfg['audit_expectations'] = dict(gradients=gradients)
        write(run / 'template-config.json', cfg)
        sham = recipe['track'] == 'round2' and recipe['block'] == 'P3' and recipe['arm'] == 'sham'
        dependency = str(root / f'round2-P3-a-s{recipe["seed"]}' / 'initial_scale.json') if sham else None
        if not sham:write(run / 'config.json', cfg)
        jobs.append(dict(recipe, run=str(run), stages=stages, resume_parent=resume_parent,
                         source_config_sha256=sha(run / 'template-config.json'), config_deferred=sham, calibration_dependency=dependency,
                         reference_initial=recipe.get('reference_initial') if resume_parent else cfg.get('initial_encoder_checkpoint'),
                         initial_step=2000 if resume_parent else 0))
    for recipe in chosen:
        cfg = copy.deepcopy(read(Path(recipe['run']) / 'config.json'))
        if recipe['model_class'] == 'base':
            cfg = read(B / 'reports/lingjing_fullspec_20261102/P1-templates/full-L0-10epoch-seed1.json')
            cfg['model'] = dict(cfg['model'], domain='color'); cfg['model_class'] = 'base'
            stages = [15265, 30530, 45795]
        else:
            stages = sorted(set([12491, 24982, recipe['nominal_target_step']]))
        add(recipe, cfg, stages)
    if cube_review:
        decisions = read(cube_review); parent = read(B / 'runs/lingjing_fullspec_hard2000_20261102/protocol.json')
        assert decisions['task'] == 'cube' and decisions['reviewed_all_seeds']
        for ident in decisions['continue_ids']:
            recipe = next(r for r in parent['jobs'] if r['id'] == ident); assert recipe['task'] == 'cube'
            terminal = read(B / 'runs/lingjing_fullspec_hard2000_20261102/terminals' / (ident + '.json')); assert terminal['passed']
            cfg = read(Path(recipe['run']) / 'config.json')
            add(recipe, cfg, [12000], resume_parent=recipe['run'])
        write(root / 'cube-review.json', decisions)
    # Weighted static host assignment, leaving dev2g0/1/7 for the three matched L0 baselines.
    jobs.sort(key=lambda r: (0 if r['task'] == 'cube' else r['tier'], r['track'], r['block'], r['arm'], r['seed']))
    owners = ['dev2'] * 5 + ['dev13'] * 8 + ['dev15'] * 8
    for i, row in enumerate(jobs):row['host'] = owners[i % len(owners)]
    assets = dict(pose=str(B / 'runs/insights12h_20261027/pose-v2/weights.pt'), i3d=str(B / 'weights/i3d_torchscript.pt'))
    assert all(Path(path).exists() for path in assets.values())
    cache = root / 'evaluations/cache'; cache.mkdir(parents=True)
    (cache / 'torch').symlink_to(B / 'cache/torch', target_is_directory=True)
    write(root / 'protocol.json', dict(source=source, helper=helper, jobs=jobs, environment=prior['environment'],
          slots=dict(dev2=[2, 3, 4, 5, 6], dev13=list(range(8)), dev15=list(range(8))),
          assets=assets, manifests=prior['manifests'], manifest_sha256=prior['manifest_sha256'],
          primary_pusht_recipes=94, cube_continuations=len(jobs) - 94,
          early_stop=dict(first_check='one full loader epoch, never at2000 for PushT', spread=.3, noisy_agent_r2=.05, info_dirs=2,
             condition='All three strong FP32 failures must agree: spread below.3,noisyagentR2<=.05,and no usable information directions or a collapsed trained forecast channel. Save the full evidence and mark early-budget stopping,not nominal maturity.'),
          nominal_budget='Round2P2/P3=24982;NCE=37473;DF=37473;colourL0=45795;review-selectedCube=12000',
          continuous_monitor=False, automatic_next_research_round=False,
          scope='All remaining primary PushT recipes; representative technical smokes/native restore and tool interfaces already passed. One-epoch collapse decisions follow the latest user insight-first priority. Healthy arms finish their prescribed budget; no LR/batch/architecture/controller tuning. P2 selection/DFPhase2 are separately gated,not silently launched. Public pooled/NCE/L0 controls are not sample-level candidates.'))
    print(dict(pusht=94, cube=len(jobs) - 94, hosts={h: sum(r['host'] == h for r in jobs) for h in ['dev2', 'dev13', 'dev15']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--helper', required=True); p.add_argument('--cube-review')
    a = p.parse_args(); main(a.root, a.helper, a.cube_review)
