"""Resilient control-plane writes around the immutable original dispatcher.

This repairs a demonstrated failure mode, not an established root cause. Model,
training, evaluator, job commands, source verification and exit semantics remain
owned by the original frozen workflow.
"""
import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def atomic_writer(root, host):
    worker_snapshot = (Path(root) / 'workers' / (host + '.json')).resolve()

    def atomic(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, indent=2)
        last_error = None
        for attempt in range(3):
            temporary = None
            try:
                fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
                temporary = Path(name)
                with os.fdopen(fd, 'w') as stream:
                    stream.write(payload)
                os.replace(temporary, path)
                return
            except OSError as error:
                if error.errno != errno.ENOENT:
                    raise
                last_error = error
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
        if path.resolve() == worker_snapshot:
            print('Non-authoritative worker snapshot write failed after3 ENOENT attempts; '
                  'original dispatcher continues to collect authoritative exits: ' + str(last_error),
                  file=sys.stderr, flush=True)
            return
        # An exit/dispatch write failure never becomes a successful completion.
        raise last_error

    return atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--source-record', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    source = json.loads(Path(args.source_record).read_text())
    manifest = json.loads(Path(source['manifest']).read_text())
    relative = 'as_jepa_tools/workflow.py'
    original = Path(source['source']) / relative
    assert hashlib.sha256(original.read_bytes()).hexdigest() == manifest['files'][relative]
    assert manifest['source_commit'] == source['source_commit']
    assert all(job['source'] == source for job in plan['jobs'])
    spec = importlib.util.spec_from_file_location('immutable_recovery_workflow', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.atomic = atomic_writer(plan['root'], args.host)
    sys.argv = [str(original), '--plan', args.plan, '--host', args.host]
    module.main()


if __name__ == '__main__':
    main()
