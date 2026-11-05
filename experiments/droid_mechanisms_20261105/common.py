"""Shared paths and durable operational metadata, not a second trainer."""
import os
from pathlib import Path
import sys

OLD = Path('/efs/prescottsun/SPT-JEPA/runs/hjepa_droid_20261105')
PYTHON = '/tmp/prescottsun-hjepa-env-97054ec/bin/python'


def digest_state(model):
    import hashlib
    import torch
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def bootstrap(root):
    root = Path(root)
    sys.path.insert(0, str(OLD / 'helper-v8'))
    from bridge import install_paths
    install_paths(root)
    import as_jepa_lewm
    extra = str(Path(__file__).resolve().parents[2] / 'as_jepa_lewm')
    if extra not in as_jepa_lewm.__path__:
        as_jepa_lewm.__path__.append(extra)
    sys.path.insert(0, str(OLD))
    from repair_20261105 import durable_json
    return durable_json


def environment(gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
               HJEPA_HOME='/tmp/prescottsun-hjepa-droid-cc05cb8d', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', HYDRA_FULL_ERROR='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key in list(env):
        if key.startswith('SLURM_'):
            del env[key]
    return env


def arm_root(root, arm):
    path = Path(root) / 'arms' / arm
    path.mkdir(parents=True, exist_ok=True)
    for key in ['public', 'candidate', 'compat_wm']:
        try:
            (path / key).symlink_to(OLD / key, target_is_directory=True)
        except FileExistsError:
            assert (path / key).resolve() == (OLD / key).resolve()
    return path


def run_dir(root, arm, seed):
    return arm_root(root, arm) / 'training' / f'temporal-forecast-seed{seed}'
