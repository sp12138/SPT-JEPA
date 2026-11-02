"""Recover already-started proposal runs after an audit-only failure; defer untouched recipes."""
import argparse
import copy
from pathlib import Path
from common import B, read, write, sha, source_check


def main(root, helper):
    root = Path(root); root.mkdir(exist_ok=False)
    parent = B / 'runs/lingjing_fullspec_main_v3_20261102'; p = read(parent / 'protocol.json')
    helper = read(helper); source_check(p['source']); source_check(helper)
    jobs, fresh = [], []
    for original in p['jobs']:
        record = parent / 'processes' / (original['id'] + '.json')
        row = copy.deepcopy(original)
        if not record.exists():
            row['recovery_mode'] = 'fresh_after_technical_release'
            assert not (Path(row['run']) / 'last.ckpt').exists()
            fresh.append(row);jobs.append(row);continue
        row['old_process'] = read(record)
        row['repair_parent_terminal'] = str(parent / 'terminals' / (row['id'] + '.json'))
        if row['task'] == 'cube':row['recovery_mode'] = 'inherit_completed_parent'
        else:
            row['recovery_mode'] = 'reuse_finished_training'
            row['reuse_completed_training_step'] = row['stages'][0]
        jobs.append(row)
    assert len(jobs) == 106 and len(fresh) == 63
    owners = ['dev2'] * 5 + ['dev13'] * 8 + ['dev15'] * 8
    recover = sorted([r for r in jobs if r['recovery_mode'] == 'reuse_finished_training'], key=lambda r: (not Path(r['repair_parent_terminal']).exists(), r['tier'], r['id']))
    for index, row in enumerate(recover + fresh):row['host'] = owners[index % len(owners)]
    (root / 'evaluations/cache').mkdir(parents=True)
    (root / 'evaluations/cache/torch').symlink_to(B / 'cache/torch', target_is_directory=True)
    write(root / 'protocol.json', dict(p, helper=helper, jobs=jobs, repair_parent_root=str(parent),
        fresh_pending_release=[r['id'] for r in fresh], delegated_cube=12, recovered_pusht=31, automatic_fresh_launch=False,
        scope='Same106 approved jobs: inherit12 healthy Cube completions;31PushT reuse completed first-stage SGD and rerun the corrected audit/evaluation. Healthy codes may finish original budgets. Untouched63 wait for explicit technical-release evidence; no new grid,NN/loss/LR/batch/data change.'))
    print(dict(jobs=len(jobs), recovered=len(recover), fresh_pending_release=len(fresh)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--helper', required=True)
    a = p.parse_args(); main(a.root, a.helper)
