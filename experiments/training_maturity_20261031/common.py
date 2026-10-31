"""Small file/provenance utilities for the six held maturity controls."""
import hashlib
import json
from pathlib import Path

B = Path('/efs/prescottsun/SPT-JEPA')
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
ROOT = B/'runs/cube_maturity23342_20261031'
PREFIX = 'experiments/training_maturity_20261031/'
CAP = 23342
FAMILIES = {
    'local': dict(parent='local_isotropic6000_20261031', stem='spatial-local', step=6000, host='dev15',
                  commit='32fad0cc04ca46cb9e6f5516a9aa0525b65b4cea', initial='local_isotropic600_20261031'),
    'native': dict(parent='native_cube12000_20261031', stem='native-lewm', step=12000, host='dev2',
                   commit='7a1ab4feb1a44cb3a6a37f5f7f2ad6a0e110487f', initial='native_lewm600_20261031'),
    'tf': dict(parent='cube_tf12000_20261031', stem='temporal-forecast', step=12000, host='dev13',
               commit='8e0f821bba84e7ed8c0f16500d7c880fde56d4cf', initial='hardtasks_gate600_20261030'),
}
CORR = B/'runs/action_covariance_source_20261031/source-record.json'
CORR_COMMIT = 'f4fd69a5e80abc68d9ecf156175ea1fd54f1af37'


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: f.write(json.dumps(x, indent=2, allow_nan=False)+'\n')


def semantic(c):
    return {k: v for k, v in c.items() if k not in
            {'output', 'checkpoint_every', 'retain_steps', 'is_resuming', 'slurm.job_id', 'slurm.task_id'}}


def source_check(s):
    for key in ['manifest', 'runtime_manifest']:
        if key not in s: continue
        m = read(s[key])
        for path, h in m['files'].items(): assert sha(Path(m['source'])/path) == h, path
    return dict(source_commit=s['source_commit'], manifest_sha256=sha(s['manifest']))


def stages(family): return [11671, 18000, CAP] if family == 'local' else [18000, CAP]


def allocations():
    return [dict(family=f, training_seed=s, task='cube', host=v['host'], gpu=s-1,
                 id=f'cube-{v["stem"]}-seed{s}', initial_step=v['step'])
            for f, v in FAMILIES.items() for s in [1, 2]]
