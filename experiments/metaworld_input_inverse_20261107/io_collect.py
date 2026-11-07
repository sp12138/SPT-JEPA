"""Finite JSON-only promotion collector; includes retained operational failures."""
import argparse
from pathlib import Path
import time

from experiments.metaworld_input_inverse_20261107.campaign import read, save, now


def failure(run):
    for name in ['execution-error-io-v2.json', 'execution-error.json']:
        if (run/name).exists():
            return name
    return None


def collect(root):
    plan = read(root/'plan.json'); recipes = plan['recipes']
    while time.time() < plan['deadline_unix']:
        if (root/'STOP').exists():
            return
        if all((Path(r['run'])/'screen-complete.json').exists() or failure(Path(r['run'])) for r in recipes):
            break
        time.sleep(30)
    else:
        return
    grouped, errors = {}, []
    for recipe in recipes:
        run = Path(recipe['run'])
        if not (run/'screen-complete.json').exists():
            errors.append(dict(name=recipe['name'], **read(run/failure(run))))
            continue
        rows = read(run/'screen5000/summary.json')['rows']
        manipulation = [r for r in rows if r['task'] not in ['reach-v3', 'reach-wall-v3']]
        hard = [r for r in manipulation if r['task'] not in ['door-open-v3', 'drawer-open-v3']]
        score = dict(seed=recipe['seed'], manipulation=100*sum(r['success'] for r in manipulation)/len(manipulation),
                     hard4=100*sum(r['success'] for r in hard)/len(hard), overall=100*sum(r['success'] for r in rows)/len(rows))
        grouped.setdefault(recipe['arm'], []).append(score)
    eligible = [(sum(r['manipulation'] for r in scores)/2, arm) for arm, scores in grouped.items()
                if len(scores) == 2 and all(r['manipulation'] >= 35 and r['hard4'] >= 20 for r in scores)]
    selected = [arm for _, arm in sorted(eligible, reverse=True)[:4]]
    selection = dict(selected=selected, scores=grouped, execution_errors=errors,
                     created_utc=now(), scope='Development-screen compute allocation only; not a public-baseline superiority claim.')
    save(root/'selection.json', selection)
    save(root/'results/screen-summary.json', dict(passed=not errors, **selection, baselines_trained=0, source=plan['source']))
    if not selected:
        save(root/'HOLD', dict(reason='No candidate passed long-training gate', utc=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    collect(p.parse_args().root)
