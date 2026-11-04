"""Verified node-local extraction; no alteration of the released corpus/splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(args):
    root = Path(args.root)
    source = Path(args.source)
    destination = Path(args.destination)
    out = root / 'cache' / args.host
    deadline = time.monotonic() + 7200
    droid = destination / 'droid'
    droid.mkdir(parents=True, exist_ok=True)
    while not (root / 'acquisition/dataset-priority-complete.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Dataset priority download exceeded two hours')
        time.sleep(5)
    for name in ['droid_paths_minus16_256p.csv', 'droid_val_indist_256p.csv', 'SHA256SUMS']:
        shutil.copy2(source / 'droid' / name, droid / name)

    def extract(relative):
        archive = source / relative
        marker = out / ('extracted-' + relative.replace('/', '--') + '.json')
        proof = root / 'acquisition/files' / ('dataset-' + relative.replace('/', '--') + '.json')
        while not proof.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(relative)
            time.sleep(5)
        verified = json.loads(proof.read_text())
        assert verified['passed'] and archive.stat().st_size == verified['bytes']
        if marker.exists():
            assert json.loads(marker.read_text())['archive_sha256'] == verified['sha256']
            return json.loads(marker.read_text())
        # Verify the actual bytes reaching this host, independently of the downloader.
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        assert digest.hexdigest() == verified['sha256']
        members = []
        with tarfile.open(archive) as tar:
            for item in tar:
                path = Path(item.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert item.isfile() or item.isdir(), 'Links/devices are not expected in the public tar'
                tar.extract(item, droid, filter='data')
                if item.isfile():
                    members.append([item.name, item.size])
        value = dict(passed=True, archive=relative, archive_sha256=verified['sha256'],
                     files=len(members), bytes=sum(v[1] for v in members),
                     inventory_sha256=hashlib.sha256(json.dumps(members).encode()).hexdigest())
        write(marker, value)
        return value

    raw = extract('droid/droid_raw_eval16.tar')
    write(out / 'eval-ready.json', dict(passed=True, destination=str(destination), raw=raw))
    files = json.loads((root / 'acquisition/dataset-source.json').read_text())['files']
    shards = [v['path'] for v in files if '/shard-' in v['path']]
    proofs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(extract, p) for p in shards]):
            proofs.append(future.result())
            write(out / 'progress.json', dict(completed=len(proofs), total=len(shards)))
    rows = (droid / 'droid_paths_minus16_256p.csv').read_text().splitlines()
    missing = [row.split()[0] for row in rows if not (droid / row.split()[0]).is_dir()]
    # Preserve the native retry semantics, including missing paths, rather than filter the CSV.
    write(out / 'train-ready.json', dict(passed=True, destination=str(destination), shards=proofs,
          training_rows=len(rows), missing_episode_paths=missing,
          split_sha256=hashlib.sha256((droid / 'droid_paths_minus16_256p.csv').read_bytes()).hexdigest(),
          csv_filtered=False, native_retry_policy_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['root', 'source', 'destination', 'host']:
        parser.add_argument('--' + key, required=True)
    main(parser.parse_args())
