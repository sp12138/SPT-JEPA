"""Read-only physical interventions with our own nine bounded commands.

No demonstration actions,inverse model,action targets,or optimizer are used.
These extra short diagnostics are not benchmark success episodes or training data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha
from experiments.metaworld_contact_20261106.bridge_campaign import TASKS


def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def run(a):
    from as_jepa_lewm.factory import build_model
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    a.output.mkdir(parents=True,exist_ok=False);cfg=json.loads((a.run/'config.json').read_text())
    model=build_model(cfg).cuda().eval().requires_grad_(False)
    weights=a.run/f'model-step{a.step:07d}/weights.pt'
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));before=model_sha(model)
    cases=[r for r in json.loads(a.manifest.read_text())['rows']if r['task']in TASKS and r['seed']in[42,43,1042,1043]]
    cases.sort(key=lambda r:(TASKS.index(r['task']),r['seed']));assert len(cases)==32
    commands=np.zeros((9,4),np.float32)
    for k in range(4):commands[1+2*k,k]=1;commands[2+2*k,k]=-1
    rows=[]
    with torch.inference_mode():
        for case in cases:
            assert hashlib.sha256(Path(case['path']).read_bytes()).hexdigest()==case['sha256']
            cached=np.load(case['path']);initial=model.encoder(resize_tensor([cached['initial']]).cuda())
            zero=initial.new_zeros(1,4);valid=initial.new_zeros(1)
            drift,effect,_=model.fields(initial,initial,torch.zeros_like(initial),zero,valid,valid,zero)
            predicted=drift.expand(9,-1,-1,-1)+torch.einsum('na,achw->nchw',torch.from_numpy(commands).cuda(),effect[0])
            actual,images,gates=[],[],[]
            for command in commands:
                env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
                try:
                    assert np.array_equal(state,cached['initial_state'])
                    assert all(np.array_equal(v,cached['initial_physics_'+k])for k,v in physics_state(env).items())
                    gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed'],gate;gates.append(gate)
                    for _ in range(5):env.step(command)
                    image=resize_tensor([pixels(env)]);images.append(image[0].permute(1,2,0).numpy())
                    actual.append((model.encoder(image.cuda())-initial)[0].cpu())
                finally:close_environment(env)
            actual=torch.stack(actual);predicted=predicted.cpu();axis=[]
            for k in range(4):
                odd=(actual[1+2*k]-actual[2+2*k])*.5
                even=(actual[1+2*k]+actual[2+2*k])*.5-actual[0]
                inferred=effect[0,k].cpu();den=float(odd.norm())
                cosine=float(torch.nn.functional.cosine_similarity(inferred.flatten(),odd.flatten(),dim=0))
                axis.append(dict(axis=k,odd_rms=float(odd.square().mean().sqrt()),even_rms=float(even.square().mean().sqrt()),
                    even_to_odd=float(even.norm())/max(den,1e-12),predicted_odd_relative_error=float((inferred-odd).norm())/max(den,1e-12),
                    predicted_odd_cosine=cosine))
            path=a.output/f'{case["task"]}-seed{case["seed"]}.npz'
            np.savez_compressed(path,commands=commands,predicted_delta=predicted.numpy(),actual_delta=actual.numpy(),
                initial=resize_tensor([cached['initial']])[0].permute(1,2,0).numpy(),future=np.stack(images))
            row=dict(task=case['task'],seed=case['seed'],case_sha256=case['sha256'],axes=axis,
                zero_actual_rms=float(actual[0].square().mean().sqrt()),zero_prediction_mse=float((predicted[0]-actual[0]).square().mean()),
                own_command_prediction_mse=float((predicted-actual).square().mean()),copy_mse=float(actual.square().mean()),
                initial_rgb_checks=gates,trace=str(path),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row);save(a.output/'progress.json',dict(rows=rows));print(json.dumps(dict(task=case['task'],seed=case['seed'],axes=axis)),flush=True)
    assert model_sha(model)==before
    save(a.output/'summary.json',dict(passed=True,model_unchanged=True,model_sha256=before,weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        rows=rows,physical_cases=32,own_interventions=288,native_steps=1440,model_updates=0,
        scope='Nine self-chosen commands per initial physical case.No expert actions/references/labels or inverse fitting.'
              'Read-only physics,not task-success evaluation or extra model training data.Shared cases across models.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in['run','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,default=2500);run(p.parse_args())
