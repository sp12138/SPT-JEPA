"""Small, auditable I/O helpers for the factual-transition experiment."""
import datetime
import hashlib
import json
import os
from pathlib import Path

BASE = Path('/efs/prescottsun/SPT-JEPA')
ROOT = BASE/'runs/metaworld_transition_only_20261109'
PYTHON = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(model, parameters_only=False):
    h = hashlib.sha256()
    values = model.named_parameters() if parameters_only else model.state_dict().items()
    for name, value in sorted(values):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def environment(source, gpu):
    result = dict(os.environ)
    paths = [source, BASE/'environments/metaworld-sanity-20261103/lib/python3.12/site-packages',
             BASE/'environments/completion-overlay', BASE/'environments/overlay']
    result.update(PYTHONPATH=':'.join(map(str, paths)), CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
        TOKENIZERS_PARALLELISM='false')
    result['LD_LIBRARY_PATH']=str(BASE/'environments/egl-lib/usr/lib/x86_64-linux-gnu')+':'+result.get('LD_LIBRARY_PATH','')
    egl = BASE/'environments/egl-lib/10_nvidia.json'
    if egl.exists():result['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(egl)
    return result
