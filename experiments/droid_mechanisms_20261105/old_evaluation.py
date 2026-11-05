"""Run the old, fixed 48 missing endpoints on the old training pair after exit."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from common import OLD, PYTHON, environment


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    limit = time.time() + 6 * 3600
    while time.time() < limit:
        ps = subprocess.check_output(['ps', '-eo', 'args'], text=True)
        active = any('repair-20261105-v1.py train' in line and not 'bash -c' in line
                     for line in ps.splitlines())
        if not active:
            break
        time.sleep(20)
    else:
        raise TimeoutError('Original recovery still running; do not overlap GPU ownership')
    os.execve(PYTHON, [PYTHON, '-u', str(OLD/'recovery-finish-20261105-v1.py'), 'eval',
                     '--root', str(OLD), '--host', 'dev15', '--gpu', str(args.gpu)], environment([args.gpu]))
