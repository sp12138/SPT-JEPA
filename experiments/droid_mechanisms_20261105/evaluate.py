"""Public planner/scorer for a frozen checkpoint of any new candidate arm."""
import argparse
import importlib.util
import json
from pathlib import Path

from common import OLD, arm_root, bootstrap


def run(args):
    root = arm_root(args.root, args.arm)
    dump = bootstrap(root)
    import torch
    spec = importlib.util.spec_from_file_location('droid_original_evaluate', OLD / 'helper-v8/evaluate.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    ready = json.loads(Path(args.ready).read_text())
    assert ready['passed'] and ready['arm'] == args.arm and ready['seed'] == args.seed
    checkpoint = ready['checkpoint']
    before = old.state_hash(torch.load(checkpoint, map_location='cpu', weights_only=False))
    assert before == ready['model_state_sha256']
    # The native evaluator recognizes our family as temporal-forecast and uses
    # its unchanged public-cost adapter. It receives no factual future actions.
    old.main(argparse.Namespace(root=str(root), arm='temporal-forecast', checkpoint=checkpoint,
             model_seed=args.seed, planner_seed=args.planner_seed, clip=args.clip, output=args.output))
    result = json.loads((Path(args.output) / 'result.json').read_text())
    result.update(arm=args.arm, stage=ready['stage'], step=ready['step'])
    dump(Path(args.output) / 'result.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['root', 'arm', 'ready', 'output']:
        p.add_argument('--'+key, required=True)
    for key in ['seed', 'planner-seed', 'clip']:
        p.add_argument('--'+key, type=int, required=True)
    run(p.parse_args())
