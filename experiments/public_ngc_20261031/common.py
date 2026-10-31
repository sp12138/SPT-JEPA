"""Small provenance helpers for the frozen public NGC comparison."""
import hashlib
import json
import os
from pathlib import Path
import sys

B = Path('/efs/prescottsun/SPT-JEPA')
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
NODES, NEIGHBORS, ANCHORS, SEED, BATCH = 5000, 15, 4, 20261031, 32
FILES = ['common.py', 'prepare.py', 'build.py', 'objective.py', 'evaluate.py',
         'review_graphs.py', 'contracts.py', 'vendor/cost.py', 'vendor/graph.py',
         'vendor/LICENSE', 'vendor/NOTICE.md', 'vendor/__init__.py']


def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def source_check(source):
    for key in ['manifest', 'runtime_manifest']:
        manifest = read(source[key])
        for f, h in manifest['files'].items(): assert sha(Path(manifest['source'])/f) == h, f
def state_hash(model):
    import torch
    h = hashlib.sha256()
    for kind, it in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
        for name, t in it:
            t = t.detach().cpu().contiguous()
            h.update(str((kind, name, tuple(t.shape), t.dtype)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def rng_hash():
    import random, numpy as np, torch
    h = hashlib.sha256(repr(random.getstate()).encode())
    h.update(repr(np.random.get_state()).encode()); h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for s in torch.cuda.get_rng_state_all(): h.update(s.cpu().numpy().tobytes())
    return h.hexdigest()
def reexec(spec, script):
    env = {**os.environ, 'PYTHONPATH':spec['model_source']['pythonpath'], 'PYTHONDONTWRITEBYTECODE':'1'}
    os.chdir(spec['model_source']['source'])
    os.execve(PY, [PY, '-B', '-u', str(Path(script).resolve()), *sys.argv[1:], '--inner'], env)
def verify(spec):
    for p, h in spec['immutable_inputs'].items(): assert sha(p) == h, p
    for f, h in spec['helper_files'].items(): assert sha(Path(__file__).parent/f) == h, f
    source_check(spec['model_source'])


def select_rows(episode_ids, offsets, lengths, train, excluded, count=NODES):
    """Common 8-frame/stride5 training population: episode -> window -> frame."""
    import numpy as np
    assert len(set(episode_ids.tolist())) == len(episode_ids)
    indices = {int(e):i for i,e in enumerate(episode_ids)}
    allowed = sorted(int(e) for e in train if lengths[indices[int(e)]] >= 36)
    assert not set(allowed) & set(map(int, excluded)) and len(allowed) >= count
    rng = np.random.default_rng(SEED); rows = []
    for e in rng.choice(allowed, count, replace=False):
        i = indices[int(e)]; start = int(rng.integers(int(lengths[i])-35))
        frame = int(rng.integers(8)); row = int(offsets[i])+start+5*frame
        rows.append(dict(episode=int(e), episode_array_index=i, start=start, frame=frame, row=row))
    return sorted(rows, key=lambda x:x['row'])


def verify_cohort(path, data):
    import numpy as np
    m = read(path); data = Path(data)
    assert m['count'] == NODES and m['seed'] == SEED and m['split'] == 'train'
    assert sha(data/'dataset_card.json') == m['dataset_card_sha256']
    for f,h in m['metadata_sha256'].items(): assert sha(data/f) == h
    arrays = {f:np.load(data/(f+'.npy')) for f in ['episode_ids','offsets','lengths','train_episodes','validation_episodes','test_episodes']}
    expected = select_rows(arrays['episode_ids'], arrays['offsets'], arrays['lengths'], arrays['train_episodes'],
                           np.r_[arrays['validation_episodes'],arrays['test_episodes']])
    assert expected == m['nodes'] and len({x['episode'] for x in expected}) == NODES
    return m
