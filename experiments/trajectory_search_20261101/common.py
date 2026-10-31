"""Small immutable-input utilities for the bounded trajectory-search control."""
import hashlib
import json
from pathlib import Path

B = Path('/efs/prescottsun/SPT-JEPA')
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
SEED = 20261101
LIBRARY = 1024
REPLACE = 64


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode() + value.tobytes()).hexdigest()


def source_check(source):
    for key in ('manifest', 'runtime_manifest'):
        if key not in source:
            continue
        manifest = read(source[key])
        for relative, expected in manifest['files'].items():
            assert sha(Path(manifest['source']) / relative) == expected, relative


def state_hash(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(str((name, tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
