"""Frozen self-generated control-path integration gate,0updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def run(a):
    from as_jepa_lewm.factory import build_model
    from as_jepa_lewm.path_projection import path_cost,path_projection
    from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    cfg=json.loads((a.run/'config.json').read_text());model=build_model(cfg).eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cpu',weights_only=True));before=model_sha(model)
    case=next(r for r in json.loads(a.manifest.read_text())['rows']if r['task']=='bin-picking-v3'and r['seed']==42)
    raw=np.load(case['path']);current=model.encoder(resize_tensor([raw['initial']]));goal=model.encoder(resize_tensor([raw['goal']]))
    past=current.new_zeros(1,4);valid=current.new_zeros(1);generator=torch.Generator().manual_seed(731)
    controls=torch.rand(4,2,4,generator=generator)*2-1;seen=[]
    class Recorder:
        grid=model.grid
        warp=model.warp
        def fields(self,image,goal,role,previous,valid):
            seen.append(previous.clone());return model.fields(image,goal,role,previous,valid)
    batched=path_cost(Recorder(),current,goal,past,valid,controls,block_steps=2)
    assert len(seen)==8
    for index in range(4):
        expected=past.expand(4,-1)if index==0 else controls[:,(index-1)//2]
        assert torch.equal(seen[2*index],expected)and torch.equal(seen[2*index+1],expected)
    single=torch.stack([path_cost(model,current,goal,past,valid,controls[j:j+1],block_steps=2)[0]for j in range(4)])
    relative=float(((batched-single).abs()/(1+single.abs())).max());assert relative<2e-5,relative
    commands,word,report=path_projection(model,current,goal,valid,past,generator,particles=4,iterations=2,blocks=2,block_steps=2)
    assert len(commands)==4 and word.shape==(2,4)and word.abs().max()<=1
    assert report['cost_history'][-1]<=report['cost_history'][0]
    assert model_sha(model)==before
    a.output.write_text(json.dumps(dict(passed=True,model_updates=0,model_unchanged=True,model_sha256=before,
        actual_planned_native_history_passed=True,sample_cost_independence_relative=relative,self_generated_word=word.tolist(),search=report,
        scope='CPU inference integration only.Full128x8/H15GPU and actualphysics gate still required.No demonstrated action read.'),indent=2)+'\n')
    print(json.dumps(dict(passed=True,relative=relative)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['run','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);run(p.parse_args())
