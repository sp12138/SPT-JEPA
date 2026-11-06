"""Algebra,unchanged raw-controller,and frozen GPU planning contracts."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from as_jepa_lewm.reachable_projection import prepare_metric,squared_metric,project
from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha


def run(a):
    from as_jepa_lewm.factory import build_model
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(91)
    population=torch.randn(24,40,dtype=torch.float64);error=torch.randn(9,40,dtype=torch.float64)
    prepared=prepare_metric(population);center=population-population.mean(0);covariance=torch.eye(40,dtype=torch.float64)+center.T@center/len(center)
    explicit=(error*torch.linalg.solve(covariance,error.T).T).sum(1);actual=squared_metric(error,prepared)
    discrepancy=float((explicit-actual).abs().max());assert discrepancy<1e-10
    # In the noiseless linear full-rank case,the observation metric recovers
    # control-prior geometry,without reading any demonstrated controls.
    prior=torch.randn(32,8,dtype=torch.float64);prior-=prior.mean(0);prior=torch.linalg.qr(prior,mode='reduced')[0]*np.sqrt(32)
    jacobian=torch.randn(40,8,dtype=torch.float64)*1000;differences=torch.randn(9,8,dtype=torch.float64)
    recovered=squared_metric(differences@jacobian.T,prepare_metric(prior@jacobian.T));reference=differences.square().sum(1)
    relative=float(((recovered-reference).abs()/(1+reference)).max());assert relative<1e-6
    cfg=json.loads((a.run/'config.json').read_text());model=build_model(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.run/'model-step0002500/weights.pt',map_location='cuda',weights_only=True));before=model_sha(model)
    case=next(x for x in json.loads(a.manifest.read_text())['rows']if x['task']=='bin-picking-v3'and x['seed']==42);data=np.load(case['path'])
    current=model.encoder(resize_tensor([data['initial']]).cuda());goal=model.encoder(resize_tensor([data['goal']]).cuda());past=current.new_zeros(1,4);valid=current.new_zeros(1)
    with torch.inference_mode():
        def generator():return torch.Generator(device='cuda').manual_seed(9182)
        old=model.plan_controls(current,current,goal,valid,past,generator(),particles=4,iterations=2)
        raw=project(model,current,current,goal,valid,past,generator(),particles=4,iterations=2)
        assert np.array_equal(np.stack(old[0]),np.stack(raw[0]))and old[2]['cost_history']==raw[2]['cost_history']
        measurements={}
        for metric in['rgb','reachable']:
            project(model,current,current,goal,valid,past,generator(),particles=32,iterations=4,metric=metric,control_prior='uniform')
            torch.cuda.synchronize();started=time.time()
            result=project(model,current,current,goal,valid,past,generator(),particles=32,iterations=4,metric=metric,control_prior='uniform')
            torch.cuda.synchronize();elapsed=time.time()-started
            assert np.isfinite(np.stack(result[0])).all()and np.abs(result[0]).max()<=1
            assert all(x>=y for x,y in zip(result[2]['cost_history'],result[2]['cost_history'][1:]))
            measurements[metric]=dict(seconds=elapsed,detail=result[2])
    from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
    env,state,_=make_environment(case['task'],case['seed'],data['rand_vec'])
    try:
        assert np.array_equal(state,data['initial_state'])
        assert all(np.array_equal(v,data['initial_physics_'+k])for k,v in physics_state(env).items())
        gate=rgb_replay_check(data['initial'],pixels(env));assert gate['passed'],gate
        for index in range(20):env.step(np.asarray(raw[0][min(index,14)],dtype=np.float32))
        frame=pixels(env);assert frame.shape==(480,480,3)and frame.max()>0
    finally:close_environment(env)
    assert model_sha(model)==before
    a.output.write_text(json.dumps(dict(passed=True,woodbury_vs_explicit_max_abs=discrepancy,linear_control_geometry_relative_error=relative,
        raw_controller_actions_and_costs_exact=True,model_unchanged=True,model_sha256=before,new_updates=0,measurements=measurements,
        actual_model_loaded_RGB_gate=gate,actual_own_control_physics_steps=20,
        scope='Only own hypothetical controls.No inverse head/actionlabels/future-action references.ModelscoringcoupleseachsinglecaseownCEMpopulation;traininglossunchanged/samplelevel.'),indent=2)+'\n');print(json.dumps(dict(passed=True,measurements=measurements)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
