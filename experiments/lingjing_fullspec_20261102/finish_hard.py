"""Finite completion dependency, no model/GPU reads while waiting and no research-agent wakes."""
import argparse
from pathlib import Path
import time
from common import read, write
from hard_collect import main as collect


def main(root):
    root = Path(root); started = time.monotonic()
    files = [root / 'dispatchers' / (host + '-finished.json') for host in ['dev2', 'dev13', 'dev15']]
    while not all(path.exists() for path in files):
        if time.monotonic() - started > 28800:raise TimeoutError('Finite8h completion window ended; no restart or extension.')
        if (root / 'STOP').exists():
            write(root / 'completion-stopped.json', dict(reason='Campaign STOP encountered; no continuation.'));return
        time.sleep(30)
    collect(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    main(p.parse_args().root)
