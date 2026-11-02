"""Run the unchanged uploaded rollout diagnostic with frozen-state provenance."""
import argparse
import inspect
import os
from pathlib import Path
import sys
import time

from common import read,write,sha


def main(root,ident):
    root=Path(root);p=read(root/'protocol.json');job=next(j for j in p['jobs'] if j['id']==ident)
    out=root/'jobs'/ident;out.mkdir(parents=True,exist_ok=False)
    import as_jepa_tools.evaluate as evaluate
    import as_jepa_tools.rollout_gain as rollout
    from experiments.trajectory_search_20261101.common import state_hash
    import torch
    tracked=[];original=evaluate.load_model
    def load(*args,**kwargs):
        model=original(*args,**kwargs);tracked.append((model,state_hash(model)))
        return model
    cp=Path(job['checkpoint']);before=sha(cp/'weights.pt');assert before==job['weights_sha256']
    evaluate.load_model=load
    command=[str(Path(rollout.__file__)),'--checkpoint',str(cp),'--data',p['data'],'--output',str(out),
             '--cases',str(p['cases']),'--window','16','--context','3']
    if job['fp32']:command.append('--fp32')
    if p.get('dataset') == 'benchmark':command += ['--dataset', 'benchmark']
    started=time.monotonic();sys.argv=command
    rollout.main();assert len(tracked)==1
    model,state=tracked[0];assert state==state_hash(model) and before==sha(cp/'weights.pt')
    paths=dict(model=inspect.getfile(type(model)),encode=inspect.getfile(model.encode),evaluate=evaluate.__file__,rollout=rollout.__file__)
    for path in paths.values():assert Path(path).resolve().is_relative_to(Path(p['source']['source']).resolve()),path
    write(out/'proof.json',dict(passed=True,job=job,seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30,
          state_before=state,state_after=state_hash(model),model_updates=0,simulation_calls=0,
          runtime_paths=paths,runtime_sha256={k:sha(v) for k,v in paths.items()},cwd=os.getcwd(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    print(dict(passed=True,id=ident))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True);a=p.parse_args();main(a.root,a.id)
