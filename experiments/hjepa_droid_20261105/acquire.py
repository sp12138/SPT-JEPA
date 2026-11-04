"""Finite anonymous download with immutable provenance and per-file checksums."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import time

REPOSITORY = 'jepa-world-models/h-jepa'
DATA_REVISION = 'cc05cb8d2aaf9aec0d5364d073c6369ff760b60b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def main(args):
    from huggingface_hub import HfApi, hf_hub_download
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    model = args.kind == 'model'
    info = (HfApi(token=False).model_info(REPOSITORY, files_metadata=True) if model else
            HfApi(token=False).dataset_info(REPOSITORY, revision=DATA_REVISION, files_metadata=True))
    revision = info.sha
    assert info.gated is False, 'Only the independently ungated public release is authorized'
    files = [f for f in info.siblings if f.rfilename.startswith('droid/')]
    assert files
    write(root / (args.kind + '-source.json'), dict(repository=REPOSITORY, revision=revision,
          repo_type='model' if model else 'dataset', gated=info.gated, token=False,
          files=[dict(path=f.rfilename, size=f.size,
                      lfs=asdict(f.lfs) if is_dataclass(f.lfs) else f.lfs) for f in files]))
    destination = Path(args.destination)

    def download(item):
        lfs = (asdict(item.lfs) if is_dataclass(item.lfs) else item.lfs) or {}
        expected = lfs.get('sha256') or lfs.get('oid')
        if expected and expected.startswith('sha256:'):
            expected = expected.split(':', 1)[1]
        last_error = None
        for attempt in range(3):
            try:
                path = Path(hf_hub_download(REPOSITORY, item.rfilename,
                    repo_type='model' if model else 'dataset', revision=revision,
                    local_dir=str(destination), token=False))
                assert path.stat().st_size == item.size, item.rfilename
                digest = sha(path)
                assert not expected or digest == expected, item.rfilename
                value = dict(path=item.rfilename, bytes=path.stat().st_size, sha256=digest,
                             published_lfs_sha256=expected, passed=True)
                write(root / 'files' / (args.kind + '-' + item.rfilename.replace('/', '--') + '.json'), value)
                return value
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        raise last_error

    # The evaluator and split lists become available before the large corpus.
    priority = [f for f in files if '/shard-' not in f.rfilename]
    remainder = [f for f in files if f not in priority]
    completed = []
    for phase in [priority, remainder]:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(download, f) for f in phase]):
                value = future.result()
                completed.append(value)
                write(root / (args.kind + '-progress.json'), dict(completed=len(completed), total=len(files),
                      bytes_verified=sum(x['bytes'] for x in completed), last_file=value['path']))
                print(json.dumps(dict(completed=len(completed), total=len(files), **value)), flush=True)
        if phase is priority:
            write(root / (args.kind + '-priority-complete.json'), dict(passed=True, revision=revision, files=completed))
    write(root / (args.kind + '-complete.json'), dict(passed=True, revision=revision,
          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=completed,
          bytes=sum(v['bytes'] for v in completed)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--kind', choices=['dataset', 'model'], required=True)
    p.add_argument('--workers', type=int, default=4)
    main(p.parse_args())
