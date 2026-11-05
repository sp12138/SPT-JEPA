"""Verify node-local read-only copies to avoid EFS random-read bottlenecks."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.metaworld_contact_20261106.diagnose import digest, save


def run(a):
    target = Path('/tmp/prescottsun-mw-pairs-20261106'); target.mkdir(exist_ok=False)
    files = []
    try:
        for shard in range(4):
            source = a.root/'cache'/f'tf-control-seed1-shard{shard}'
            destination = target/str(shard); destination.mkdir()
            for name in ['pixels.npy', 'actions.npy', 'manifest.json']:
                shutil.copyfile(source/name, destination/name)
                expected, actual = digest(source/name), digest(destination/name)
                assert expected == actual
                files.append(dict(path=str(destination/name), source=str(source/name), sha256=actual))
                print(json.dumps(files[-1]), flush=True)
        save(target/'COPIED.json', dict(passed=True, files=files, source=str(a.root), scope='Exact node-local bytes;no data/labels/splits modified.'))
    except BaseException as error:
        save(target/'COPY_ERROR.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); run(p.parse_args())
