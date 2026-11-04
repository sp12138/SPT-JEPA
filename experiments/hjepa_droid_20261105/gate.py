"""Disposable actual dual-GPU train, full restore, and full public-planner calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import traceback

from campaign import ARMS, PYTHON, child, environment, write


def main(args):
    root = Path(args.root)
    helper = Path(__file__).resolve().parent
    arm = ARMS[args.pair]
    gpu = 2 * args.pair
    preflight_root = root / 'preflight-v8' / args.host
    run = preflight_root / f'{arm}-seed1'
    run.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 3600

    def execute(name, command, gpus):
        rc = child(root, command, environment(gpus), run / (name + '.log'),
                   run / (name + '-launch.json'), deadline)
        if rc:
            raise RuntimeError(f'{name} exited {rc}; original log retained')

    try:
        command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc-per-node=2', str(helper / 'train.py'), '--root', str(root),
                   '--host', args.host, '--arm', arm, '--seed', '1', '--preflight',
                   '--preflight-tag', 'preflight-v8']
        execute('training', command, [gpu, gpu + 1])
        command = [PYTHON, '-u', str(helper / 'audit.py'), '--root', str(root), '--host', args.host,
                   '--arm', arm, '--run', str(run)]
        execute('restore', command, [gpu])
        model_name = 'droid_hjepa_l2' if arm == 'h-jepa' else 'droid_lewm'
        own_output = run / 'planning-check'
        own = [PYTHON, '-u', str(helper / 'evaluate.py'), '--root', str(root), '--arm', arm,
               '--model-seed', '1', '--planner-seed', '1', '--clip', '0', '--output', str(own_output),
               '--checkpoint', str(run / f'{model_name}_object.ckpt')]
        ref_arm = ['lewm', 'h-jepa', 'hwm', 'h-jepa'][args.pair]
        ref_seed = 1 if args.host == 'dev13' else 10000
        ref = [PYTHON, '-u', str(helper / 'evaluate.py'), '--root', str(root), '--arm', ref_arm,
               '--model-seed', str(ref_seed), '--planner-seed', '1', '--clip', str(args.pair),
               '--output', str(run / 'reference-check')]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(execute, 'planning', own, [gpu]),
                       pool.submit(execute, 'reference', ref, [gpu + 1])]
            for future in futures:
                future.result()
        write(run / 'gate-complete.json', dict(passed=True, arm=arm, host=args.host,
              step=3, full_public_planner=True, full_cpu_restore=True,
              published_reference_loaded=True, preflight_is_not_a_mature_score=True))
    except Exception:
        write(run / 'gate-error.json', dict(passed=False, traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--pair', required=True, type=int)
    main(parser.parse_args())
