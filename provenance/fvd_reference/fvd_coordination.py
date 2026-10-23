"""Checkpoint leases only; the frozen Native evaluator remains byte-for-byte unchanged."""
from pathlib import Path
import fcntl, hashlib, json, os, time, uuid
J=Path(__file__).resolve().parent
S=json.loads((J/'state.json').read_text())
S=dict(S,**S.get('evaluation_profile',{}))
R=Path(S['report']); O=R/'fvd'

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2)
    for attempt in range(3):
        tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('x') as f:
                f.write(payload);f.flush();os.fsync(f.fileno())
            try:os.replace(tmp,path);return
            except FileNotFoundError:
                if path.exists() and path.read_text()==payload:return
                if attempt==2:raise
        finally:tmp.unlink(missing_ok=True)

def evaluation_hardware():
    hardware=os.environ.get('FVD_EVALUATION_HARDWARE','H200')
    if hardware not in {'H200','B300'}:raise ValueError(hardware)
    return hardware

def evaluation_body(hardware=None):
    hardware=hardware or evaluation_hardware()
    if hardware=='H200':return J/S.get('eval_entrypoint','eval_entry_frozen.py')
    registry=json.loads((O/'protocol_registry.json').read_text())
    return J/registry[hardware]['body']

def protocol_path(hardware):
    if hardware=='H200':return O/'protocol.json'
    return O/json.loads((O/'protocol_registry.json').read_text())[hardware]['protocol']

def allowed_protocol_hashes():
    registry=json.loads((O/'protocol_registry.json').read_text()) if (O/'protocol_registry.json').exists() else {'H200':{}}
    return {protocol_hash(hardware) for hardware in registry}

def reserved_targets():
    targets=set()
    for root in [O/'extras/receipts',O/'b300_extras/receipts']:
        for path in root.glob('*.json'):
            receipt=json.loads(path.read_text())
            if not receipt.get('deleted'):targets.add((receipt['name'],receipt['step']))
    return targets

def sampling_batch_size(name):
    mapping=S.get('sampling_batch_sizes')
    return int(mapping[name]) if mapping else 1

def protocol_hash(hardware=None):
    hardware=hardware or evaluation_hardware()
    body=evaluation_body(hardware)
    if not body.exists():body=J/'eval_entry.py'
    value=dict(protocol=json.loads(protocol_path(hardware).read_text()),
      evaluator_sha256=hashlib.sha256(body.read_bytes()).hexdigest(),
      detector_sha256=hashlib.sha256(Path('/efs/stormyxie/fvd/fvd.py').read_bytes()).hexdigest(),
      native_source_digest=S['source_digest'])
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def lease_paths(output):
    key=hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()
    root=O/'leases';root.mkdir(parents=True,exist_ok=True)
    return root/(key+'.lock'),root/(key+'.owner.json')

class Lease:
    def __init__(self,output,blocking=True):
        self.output=str(Path(output).resolve());self.blocking=blocking
        self.handle=None;self.token=uuid.uuid4().hex
    def __enter__(self):
        lock,owner=lease_paths(self.output)
        self.handle=lock.open('a+')
        try:fcntl.flock(self.handle,fcntl.LOCK_EX|(0 if self.blocking else fcntl.LOCK_NB))
        except BaseException:
            self.handle.close();self.handle=None;raise
        atomic(owner,dict(output=self.output,token=self.token,pid=os.getpid(),
          host=os.uname().nodename,time=time.time(),task=os.environ.get('FVD_TASK_NAME')))
        return self
    def __exit__(self,*args):
        if self.handle is not None:
            fcntl.flock(self.handle,fcntl.LOCK_UN);self.handle.close();self.handle=None

def valid_score(output,expected):
    try:
        out=Path(output);s=json.loads((out/'score.json').read_text())
        return (s.get('name')==out.parent.name and s.get('weights')==out.parent.name.rsplit('__',1)[-1] and s.get('ema_decay')==0.99 and s.get('cases')==2048 and s.get('smoke') is False and
          s.get('fvd_protocol_sha256') in ({expected} if isinstance(expected,str) else set(expected)) and
          s.get('eval_manifest_sha256')==hashlib.sha256((out/'eval_manifest.json').read_bytes()).hexdigest())
    except (OSError,ValueError):return False

def check_parent(output,token):
    lock,owner=lease_paths(output)
    data=json.loads(owner.read_text())
    assert data['token']==token and data['output']==str(Path(output).resolve()),'Parent lease identity mismatch'
    with lock.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        fcntl.flock(f,fcntl.LOCK_UN)
    raise RuntimeError('Parent lease no longer held')

def guarded_run(output,body,finalize,expected):
    rank=int(os.environ.get('RANK','0'))
    parent=os.environ.get('FVD_PARENT_LOCK_TOKEN')
    if parent:
        check_parent(output,parent)
        body()
        if rank==0:finalize()
        return
    run_id=os.environ.get('TORCHELASTIC_RUN_ID')
    if not run_id or run_id=='none':
        raise RuntimeError('Checkpoint coordination requires a unique torchrun run id')
    identity=str(Path(output).resolve())+'|'+run_id+'|'+os.environ.get('TORCHELASTIC_RESTART_COUNT','0')
    marker=O/'leases'/(hashlib.sha256(identity.encode()).hexdigest()+'.round.json')
    if rank==0:
        with Lease(output):
            done=valid_score(output,expected)
            atomic(marker,dict(action='skip' if done else 'run',time=time.time(),identity=identity))
            if not done:
                body()
                finalize()
    else:
        deadline=time.monotonic()+12*3600
        while not marker.exists():
            if time.monotonic()>deadline:raise TimeoutError('Waiting for checkpoint lease')
            time.sleep(0.5)
        data=json.loads(marker.read_text())
        assert data['identity']==identity
        if data['action']=='run':body()
        elif data['action']!='skip':raise ValueError('Invalid checkpoint gate action')

def commit_score(name,ckpt,output,expected):
    import sys
    sys.path[:0]=[S['runtime_snapshot'],'/efs/stormyxie/fvd']
    import numpy as np
    from fvd_pool import read_features
    from fvd import frechet_distance
    out=Path(output)
    rows=[json.loads(x) for x in (R/'data/eval_2048.jsonl').read_text().splitlines()]
    assert len(rows)==2048
    real=np.stack([np.load(O/'bank'/(x['sample_id']+'.feature.npy'),allow_pickle=False) for x in rows])
    manifest=json.loads((out/'eval_manifest.json').read_text())
    fake=read_features(out,rows,manifest,expected)
    identity=next(x for x in json.loads((J/'eval_runs.json').read_text()) if x['name']==name)
    assert manifest['evaluation_weights']==identity['weights']
    for sidecar in out.glob('fvd_*.npy.json'):
        meta=json.loads(sidecar.read_text())['metadata']
        assert meta['evaluation_weights']==identity['weights'] and meta['ema_decay']==0.99
        if identity['weights']=='ema':assert meta['ema_updates']==int(Path(ckpt).name.split('.')[0])
    score=dict(training_name=identity['training_name'],weights=identity['weights'],ema_decay=0.99,name=name,step=int(Path(ckpt).name.split('.')[0]),checkpoint=ckpt,
      smoke=False,cases=2048,metric=frechet_distance(real,fake,ddof=1),
      protocol=dict(cfg=1,steps=50,observed_frames=16,metric_frames=128,fps=16,seed=20260920,cov_ddof=1,sampling_batch_size=sampling_batch_size(name)),
      fvd_protocol_sha256=expected,eval_manifest_sha256=hashlib.sha256((out/'eval_manifest.json').read_bytes()).hexdigest(),
      evaluation_hardware=evaluation_hardware(),source_digest=S['source_digest'],scheduler='checkpoint_lease_v1',finished=time.time())
    atomic(out/'score.json',score)
