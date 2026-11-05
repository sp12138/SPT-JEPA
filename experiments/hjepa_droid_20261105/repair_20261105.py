"""Repair telemetry I/O and resume only the interrupted original H-JEPA seed."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def durable_json(path, value, *, replace=os.replace, delay=time.sleep):
    """Unique sibling temporary files; bounded retries for transient filesystem faults."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2) + '\n').encode()
    last = None
    for attempt in range(3):
        temporary = None
        try:
            fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
            temporary = Path(name)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            replace(temporary, path)
            return
        except OSError as error:
            last = error
            if error.errno not in [errno.ENOENT, errno.EIO, errno.ESTALE, errno.EAGAIN]:
                raise
            if attempt < 2:
                delay(.1 * (attempt + 1))
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    raise last


def training_entry(args):
    root = Path(args.root)
    recovery = root / 'recovery-20261105'
    run = root / 'training/h-jepa-seed10000'
    sys.path.insert(0, str(root / 'helper-v8'))
    import train as original
    original.install_paths(root)
    expected = json.loads((recovery / 'checkpoint-proof.json').read_text())
    old_entry = json.loads((recovery / 'original/entry.json').read_text())

    def repaired_dump(path, value):
        path = Path(path)
        if path == run / 'entry.json':
            assert value['initial_state_sha256'] == expected['model_state_sha256'], 'GPU restore differs from CPU checkpoint'
            value['restored_state_sha256'] = value['initial_state_sha256']
            value['initial_state_sha256'] = old_entry['initial_state_sha256']
            value['random_initialization'] = False
            value['same_data_random_origin'] = True
            value['resumed_step'] = expected['global_step']
            value['resume_checkpoint_sha256'] = expected['checkpoint_sha256']
            path = recovery / 'resumed-entry.json'
        try:
            durable_json(path, value)
        except OSError as error:
            if path.name not in ['health-rank0.json', 'health-rank1.json']:
                raise
            # A telemetry failure must not destroy otherwise healthy optimization.
            fallback = Path('/tmp/prescottsun-hjepa-health-20261105') / path.name
            durable_json(fallback, value)
            print(json.dumps(dict(telemetry_fallback=str(fallback), original=str(path),
                                  error=repr(error), step=value.get('step'))), flush=True)

    original.dump = repaired_dump
    # Keep the original Trainer/model/loss/optimizer; cap the original total updates
    # explicitly because a mid-batch Lightning loop checkpoint can replay a boundary.
    import types
    import lightning as pl
    import main_hjepa as main
    native_trainer = pl.Trainer

    def capped_trainer(*a, **kwargs):
        kwargs['max_steps'] = 29200
        return native_trainer(*a, **kwargs)

    main.pl = types.SimpleNamespace(Trainer=capped_trainer, pytorch=pl.pytorch)
    original.run(argparse.Namespace(root=str(root), host=args.host, arm='h-jepa', seed=10000,
                                    preflight=False, preflight_tag='unused'))


def test_writer():
    from concurrent.futures import ThreadPoolExecutor
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        attempts = []

        def transient(source, target):
            attempts.append(1)
            if len(attempts) == 1:
                Path(source).unlink()
                raise FileNotFoundError(errno.ENOENT, 'injected missing temporary')
            os.replace(source, target)

        durable_json(root / 'transient.json', {'value': 3}, replace=transient, delay=lambda _: None)
        assert json.loads((root / 'transient.json').read_text()) == {'value': 3}
        assert len(attempts) == 2
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda index: durable_json(root / 'concurrent.json', {'value': index}), range(64)))
        assert json.loads((root / 'concurrent.json').read_text())['value'] in range(64)
        assert not list(root.glob('*.tmp'))
        failures = []

        def persistent(source, target):
            failures.append(1)
            raise OSError(errno.EIO, 'injected persistent error')

        try:
            durable_json(root / 'critical.json', {}, replace=persistent, delay=lambda _: None)
        except OSError:
            pass
        else:
            raise AssertionError('Persistent critical I/O must fail explicitly')
        assert len(failures) == 3
    print('PASS: missing-temp retry, 64 concurrent atomic writes, bounded persistent-I/O failure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--root')
    parser.add_argument('--host')
    args = parser.parse_args()
    test_writer() if args.mode == 'test' else training_entry(args)
