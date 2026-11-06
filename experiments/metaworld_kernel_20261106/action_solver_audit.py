"""Frozen ODE convergence audit, not physical control or a new training loss."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_lewm.factory import build_model
from as_jepa_tools.native_inverse import NativeInversePairs
from experiments.metaworld_eval_20261103.evaluate import model_sha


def solve(model,memory,horizon,noise,steps,kind='euler'):
    value=noise.clone()
    for i in range(steps):
        time=value.new_full((len(value),),1-i/steps)
        velocity=model.action_field(value,time,horizon,memory)
        if kind=='midpoint':
            velocity=model.action_field(value-velocity/(2*steps),time-1/(2*steps),horizon,memory)
        value=value-velocity/steps
    return value


def run(a):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    reports=[]
    with torch.inference_mode():
        for path in a.runs:
            cfg=json.loads((path/'config.json').read_text());assert cfg['model']['action_dim']==4 and cfg['model']['objective']=='flow'
            assert json.loads((path/'audit-40000.json').read_text())['passed']
            model=build_model(cfg).cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(path/'model-step0040000/weights.pt',map_location='cuda',weights_only=True));before=model_sha(model)
            data_cfg=dict(cfg,seed=20261107);data=NativeInversePairs(data_cfg,'test');indices=np.linspace(0,len(data)-1,64,dtype=int).tolist()
            card=json.loads((Path(cfg['data'])/'dataset_card.json').read_text());mean=np.asarray(card['action_mean'][:4]);std=np.asarray(card['action_std'][:4])
            outputs={};rows=[];parity=[]
            for offset in range(0,len(indices),16):
                batch=next(iter(DataLoader(Subset(data,indices[offset:offset+16]),batch_size=16,num_workers=0)))
                minimal=dict(pixels=batch['pixels'].cuda());memory=model.memory(minimal);horizon=model.duration(memory)
                seed=771000+offset;generator=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(len(memory),1,4,device='cuda',generator=generator)
                reference=model.propose(minimal,generator=torch.Generator(device='cuda').manual_seed(seed),steps=12)
                euler=solve(model,memory,horizon,noise,12);error=float((reference-euler).abs().max());assert error==0,error;parity.append(error)
                for kind,steps in [('euler',12),('euler',24),('euler',48),('euler',96),('euler',192),('euler',384),('euler',768),('midpoint',12),('midpoint',24),('midpoint',48)]:
                    value=euler if kind=='euler'and steps==12 else solve(model,memory,horizon,noise,steps,kind)
                    outputs.setdefault(f'{kind}{steps}',[]).append(value[:,0].cpu().numpy())
                rows.extend(dict(episode=int(batch['episode'][i]),start=int(batch['start'][i]),image_rows=batch['rows'][i].tolist())for i in range(len(memory)))
            outputs={k:np.concatenate(v)for k,v in outputs.items()};reference=outputs['euler768'];raw_reference=np.clip(reference*std+mean,-1,1);metrics={}
            for name,value in outputs.items():
                raw=np.clip(value*std+mean,-1,1);delta=raw-raw_reference
                metrics[name]=dict(nfe=int(name.replace('midpoint','').replace('euler',''))*(2 if name.startswith('midpoint')else 1),
                    normalized_rmse_vs_euler768=float(np.sqrt(np.mean((value-reference)**2))),mean_raw_arm_l2=float(np.linalg.norm(delta[:,:3],axis=1).mean()),
                    raw_grip_abs_quantiles=np.quantile(abs(delta[:,3]),[.5,.9,.99]).tolist(),raw_grip_sign_disagreement=float(np.mean((raw[:,3]>=0)!=(raw_reference[:,3]>=0))))
            assert before==model_sha(model)
            reports.append(dict(run=str(path),model_sha256=before,model_unchanged=True,original_propose_vs_manual_euler12_max=parity,
                metrics=metrics,rows=rows,outputs={k:v.tolist()for k,v in outputs.items()}))
            del model;torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(passed=True,reports=reports,new_updates=0,new_simulator_runs=0,
        scope='64COMMONheldoutcurrent/goalwindows/model,sameownGaussianlatent,FP32. Currenta_tdataonly;no futureactionlabels/teacher. '
              'Euler768isNUMERICALreference,NOTtruephysical/optimalactions. MaturitygroupalsochangestrainingLRschedule/seed. '
              'Midpointuses2fieldpasses/step/neverqueriesendpointt0. Solvererrorisnotphysicalsuccess;mustverifyphysicalcontrolbeforepromotion.'),indent=2)+'\n')
    print(json.dumps({r['run']:{k:v for k,v in r['metrics'].items()if k in ['euler12','euler24','midpoint12','euler384']}for r in reports}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
