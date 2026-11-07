"""Complete the one interrupted original recipe; no new method/baseline/seed."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import traceback

from experiments.metaworld_input_inverse_20261107.campaign import (
    PYTHON, read, save, digest, environment, execute, evaluate, now)


def stage(helper, source):
    # Bind the same verified local shards; then the fixed operational save hook.
    import importlib.util
    old = Path('/efs/prescottsun/SPT-JEPA/reports/metaworld_input_inverse_20261107/helper-io-v2/io_stage.py')
    spec = importlib.util.spec_from_file_location('input_inverse_cache', old)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.install_cache()
    safe_spec = importlib.util.spec_from_file_location('input_inverse_safe_save', helper/'safe_train_stage.py')
    safe = importlib.util.module_from_spec(safe_spec); safe_spec.loader.exec_module(safe)
    import as_jepa_lewm.train as entry
    entry.PersistentProgress.save = safe.safe_save
    sys.argv = [str(helper/'train_stage.py'), *sys.argv[sys.argv.index('--config'):]]
    runpy.run_path(str(helper/'train_stage.py'), run_name='__main__')


def recover(root, helper):
    plan = read(root/'plan.json'); source = Path(plan['source'])
    name = 'predictive-s4-w10-seed1'; recipe = next(r for r in plan['recipes'] if r['name'] == name)
    run = Path(recipe['run']); out = root/'recovery-prune-v2'
    out.mkdir(exist_ok=False); original = out/'original'; original.mkdir()
    deadline = min(plan['deadline_unix'], time.time()+7200)
    env = environment(source, recipe['gpu']); env['SPT_INPUT_INVERSE_NN_SOURCE'] = str(source)
    try:
        from experiments.metaworld_eval_20261103.worker import free_gpu
        free_gpu(recipe['gpu'])
        assert digest(run/'config.json') == recipe['config_sha256']
        for p in [run/'status.json', run/'execution-error-io-v2.json', run/'error.txt',
                  run/'train-io-v2-5000.log', root/'selection.json', root/'results/screen-summary.json', root/'HOLD']:
            if p.exists():
                prefix = 'run-' if p.parent == run else 'root-'
                shutil.copyfile(p, original/(prefix+p.name))
        old_selection = read(root/'selection.json')
        assert old_selection['selected'] == []
        peer = next(s for s in old_selection['scores'][recipe['arm']] if s['seed'] == 2)
        assert peer['manipulation'] == 0  # This completion cannot satisfy BOTH-seed promotion.
        save(out/'START.json', dict(utc=now(), name=name, deadline_unix=deadline,
            original_error='Missing obsolete export; first repair also hit an existing runtime-source audit name at4001. Both retained.',
            scope='Only checkpoint cleanup idempotence and unique per-attempt provenance filenames. Same immutable learner/trainer/data/config/optimizer/LR/sampler; original5000budget only.',
            checkpointed_sgd_repeated=0, prior_unsaved_update_replayed=1,
            nn_source=str(source), helper=str(helper), source_sha256=digest(source/'as_jepa_lewm/executed_inverse.py')))
        old_stage = Path('/efs/prescottsun/SPT-JEPA/reports/metaworld_input_inverse_20261107/helper-io-v2/io_stage.py')
        before = out/'audit-native4000.json'
        execute([PYTHON, '-B', '-u', str(old_stage), 'audit', '--run', str(run), '--step', '4000', '--output', str(before)],
                out/'audit-native4000.log', env, source, deadline, cap=1800)
        audit = read(before); assert audit['passed'] and audit['checkpoint']['step'] == 4000
        os.link(run/'last.ckpt', out/'parent-native4000.ckpt')
        save(out/'parent.json', dict(step=4000, sha256=audit['checkpoint_sha256'],
            samples=audit['checkpoint']['campaign_samples'], complete_cpu_restore=audit['actual_cpu_restore']))
        execute([PYTHON, '-B', '-u', str(helper/'repair_prune.py'), '--mode', 'stage',
                 '--root', str(root), '--helper', str(helper), '--config', str(run/'config.json'), '--stop-step', '5000', '--resume'],
                out/'train4000-to5000.log', env, source, deadline)
        completion = read(run/'stage-complete.json')
        assert completion['initial_step'] == 4000 and completion['stop_step'] == 5000
        after = out/'audit-native5000.json'
        execute([PYTHON, '-B', '-u', str(old_stage), 'audit', '--run', str(run), '--step', '5000', '--output', str(after)],
                out/'audit-native5000.log', env, source, deadline, cap=1800)
        audit = read(after); assert audit['passed']
        archive = run/'archive-step0005000'; archive.mkdir(exist_ok=False)
        os.link(run/'last.ckpt', archive/'native.ckpt')
        save(archive/'manifest.json', dict(step=5000, sha256=audit['checkpoint_sha256']))
        evaluate(root, source, run, 5000, Path(plan['screen_manifest']), run/'screen5000', env, deadline)
        save(run/'screen-complete.json', dict(passed=True, step=5000, recovered_by=str(out), ended_utc=now()))
        save(run/'progress.json', dict(step=5000, state='stage_complete', recovery=str(out), updated_utc=now()))
        # All original error/log/HOLD evidence was preserved above; supersede
        # only the effective collector, reusing the other31 completed screens.
        spec_path = Path('/efs/prescottsun/SPT-JEPA/reports/metaworld_input_inverse_20261107/helper-io-v2/io_collect.py')
        import importlib.util
        spec = importlib.util.spec_from_file_location('input_inverse_collect', spec_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.collect(root)
        revised = read(root/'selection.json'); assert revised['selected'] == [] and not revised['execution_errors']
        assert all(old_selection['scores'][arm] == revised['scores'][arm]
                   for arm in old_selection['scores'] if arm != recipe['arm'])
        save(out/'CORRECTIONS.json', dict(passed=True, original_error_retained=str(run/'execution-error-io-v2.json'),
            original_summary_preserved=str(original), actual_resume_from=4000, actual_stop=5000,
            checkpointed_sgd_repeated=0, unsaved_update_replayed=1, other31_screens_reused=True, new_selected=[], utc=now()))
        save(out/'COMPLETE.json', dict(passed=True, original_recipes_complete=32, effective_errors=0, historical_errors=1, utc=now()))
    except BaseException as error:
        save(out/'execution-error.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=now()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['work', 'stage'], required=True)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--helper', type=Path, required=True)
    a, _ = p.parse_known_args()
    if a.mode == 'work': recover(a.root, a.helper)
    else: stage(a.helper, Path(read(a.root/'plan.json')['source']))
