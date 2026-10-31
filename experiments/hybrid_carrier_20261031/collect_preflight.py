"""Completed-only CPU collection and native model/optimizer restore audit."""
import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from as_jepa_lewm.train import TrainModule, WindowSampler
from as_jepa_tools.benchmark_data import BenchmarkWindows
from vforge.ops import OpsConfig, initialize_ops


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def restore(run,cfg,datasets):
    checkpoint=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['global_step']==2 and checkpoint['campaign_samples']==256
    torch.manual_seed(cfg['seed']);module=TrainModule(cfg)
    module.load_state_dict(checkpoint['state_dict'],strict=True)
    assert same(module.state_dict(),checkpoint['state_dict'])
    configured=module.configure_optimizers();optimizer=configured['optimizer'];scheduler=configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer_states'][0]);scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    assert same(optimizer.state_dict(),checkpoint['optimizer_states'][0])
    assert same(scheduler.state_dict(),checkpoint['lr_schedulers'][0])
    qstates=sum(p in optimizer.state for p in module.model.pool.parameters())
    assert qstates==(0 if cfg['model']['hybrid_query']=='fixed' else 9)
    rng=checkpoint['campaign_rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    assert same(random.getstate(),rng['python']) and same(np.random.get_state(),rng['numpy'])
    assert torch.equal(torch.get_rng_state(),rng['torch'])
    assert rng['cuda'] and all(torch.is_tensor(s) and s.dtype==torch.uint8 and s.numel()>0 for s in rng['cuda'])
    key=(cfg['data'],cfg['seed'])
    if key not in datasets:datasets[key]=BenchmarkWindows(cfg['data'],window=cfg['model']['window'],split='train',seed=cfg['seed'])
    def upcoming():
        it=iter(WindowSampler(datasets[key],start=checkpoint['campaign_samples'],seed=cfg['seed'],batch_size=cfg['batch_size']))
        return [next(it) for _ in range(16)]
    indices=upcoming();assert upcoming()==indices
    result=dict(passed=True,strict_native_module_optimizer_scheduler_roundtrip=True,
        CPU_python_numpy_torch_rng_roundtrip=True,cuda_rng_states_verified_only=len(rng['cuda']),
        cuda_rng_applied=False,optimizer_updates=0,model_tensors=len(module.model.state_dict()),
        restored_Q_optimizer_states=qstates,cursor=256,native_next16_window_indices=indices,
        checkpoint_sha256=sha(run/'last.ckpt'))
    del checkpoint,module,configured,optimizer,scheduler;gc.collect()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plan=read(root/'plan.json');closed=read(root/'closed.json')
    assert closed['completed']==16 and closed['failed']==0 and (root/'HOLD').exists()
    exits=[read(root/'exits'/(j['id']+'.json')) for j in plan['jobs']]
    assert len(exits)==16 and all(e['returncode']==0 for e in exits)
    torch.set_num_threads(2);initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)
    modelsrc=read(root/'model-source-record.json');helpersrc=read(root/'helper-source-record.json')
    for src in [modelsrc,helpersrc]:
        m=read(src['manifest'])
        for n,digest in m['files'].items():assert sha(Path(src['source'])/n)==digest,n
    rows=[];datasets={}
    for job in plan['jobs']:
        if job['phase']!='train2':continue
        run=Path(job['output']);cfg=read(run/'config.json');assert sha(run/'config.json')==job['config_sha256']
        status=read(run/'status.json');assert status['state']=='finished' and status['step']==2
        native=read(root/('fullstate-'+run.name)/'preflight.json')
        audit=read(root/('fullstate-'+run.name)/'hybrid-query-audit.json')
        assert native['passed'] and audit['passed'] and native['paired_initial_equal']['compared_tensors']==555
        trace=[json.loads(x) for x in (run/'preflight-observations.jsonl').read_text().splitlines()]
        encoded=[x for x in trace if x['kind']=='encoded_branches'];clip=[x for x in trace if x['kind']=='native_gradient_clipping']
        assert len(encoded)==len(clip)==2 and all(x['original_calls']==1 and x['configured_threshold']==1. for x in clip)
        assert all(s['finite'] for x in encoded for s in x['statistics'].values())
        assert all(s['finite'] for x in clip for side in ['before','after'] for s in x[side].values())
        rows.append(dict(id=run.name,task=job['task'],seed=job['training_seed'],arm=job['arm'],host=job['host'],gpu=job['gpu'],
            status=status,paired_initial_tensors=555,frozen=native['frozen_tensors_unchanged'],
            native_checkpoint_checks=native['checkpoint'],query_audit=audit,
            encoded_branches=encoded,clipping=clip,native_cpu_restore=restore(run,cfg,datasets)))
    paired=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            left=next(r for r in rows if (r['task'],r['seed'],r['arm'])==(task,seed,'fixed'))
            right=next(r for r in rows if (r['task'],r['seed'],r['arm'])==(task,seed,'adaptive'))
            paired.append(dict(task=task,seed=seed,first_batch_branch_statistics_exact=left['encoded_branches'][0]['statistics']==right['encoded_branches'][0]['statistics']))
    summary=dict(passed=True,run=str(root),model_source=modelsrc,helper_source=helpersrc,exits=exits,closed=closed,
        runs=rows,paired=paired,collector_sha256=sha(__file__),device='cpu',extra_GPU_calls=0,extra_optimizer_updates=0,
        scope='Two-update technical gate only; recorded throughput includes instrumentation/startup/checkpoint overhead. No task performance evaluated or600 continuation authorized.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,runs=len(rows),endpoints=len(exits),native_CPU_restores=8,output=str(out))))


if __name__=='__main__':main()
