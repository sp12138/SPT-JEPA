"""Action-following evidence from matched simulator resets and fixed candidate actions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from as_jepa_tools.evaluate import load_model,predict_states,pose_net
from as_jepa_tools.data import PushTWindows
from vforge.ops import initialize_ops,OpsConfig


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--data',required=True);parser.add_argument('--pose',required=True);parser.add_argument('--output',required=True);parser.add_argument('--cases',type=int,default=32);parser.add_argument('--steps',type=int,default=50);args=parser.parse_args()
    torch.set_num_threads(2);initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda',0),parallel_state=None)
    from stable_worldmodel.envs.pusht import PushT as PushTEnv
    import inspect
    print('PushTEnv signature',inspect.signature(PushTEnv),flush=True)
    data=PushTWindows(args.data,window=16,gap=8,split='test');data.open();model=load_model(args.checkpoint);pose=pose_net(args.pose);rng=np.random.default_rng(42)
    # The world-model frames and control labels stay from held-out episodes.
    candidate=rng.choice(len(data),min(5000,len(data)),replace=False);eligible=[int(index) for index in candidate if data.locate(index)[1]>=8][:args.cases]
    results=[];started=time.time()
    # The simulator constructor and state setter are checked against the pinned upstream source before launching.
    env=PushTEnv(render_mode='rgb_array')
    raw=np.load(Path(args.data)/'action_raw.npy',mmap_mode='r');raw_rows=np.load(Path(args.data)/'raw_rows.npy');state=np.load(Path(args.data)/'states.npy',mmap_mode='r')
    card=json.loads((Path(args.data)/'dataset_card.json').read_text());mean=np.asarray(card['action_mean']);std=np.asarray(card['action_std'])
    for case,index in enumerate(eligible):
        batch=data[index];ep,start=data.locate(index);rows=batch['rows'].numpy();origin=int(rows[2]);initial=np.array(state[origin]);origin_raw=int(raw_rows[origin])
        donors=[origin_raw]+[int(raw_rows[int(data[rng.integers(len(data))]['rows'][2])]) for _ in range(3)]
        truth=[];normalized=[]
        for donor in donors:
            sequence=np.array(raw[donor:donor+40]);sequence=np.nan_to_num(sequence,nan=0.)
            env.reset(seed=case);env._set_state(initial.copy());trajectory=[]
            for t,action in enumerate(sequence):
                env.step(action)
                if (t+1)%5==0:
                    simulation=np.r_[np.array(env.agent.position),np.array(env.block.position),float(env.block.angle)]
                    trajectory.append(simulation)
            truth.append(np.asarray(trajectory));normalized.append(((sequence-mean)/std).reshape(8,10))
        pixels=batch['pixels'][:3][None].cuda();frames=batch['render'][:3][None].cuda().float()/127.5-1.
        anchorrow=int(data.offsets[ep])+start-model.anchor_gap if model.mode not in {'S0','S1'} else int(rows[0]);anchor=torch.from_numpy(np.array(data.p64[anchorrow])).permute(2,0,1)[None].cuda().float()/127.5-1.
        past=batch['action'][:2].numpy();actions=torch.from_numpy(np.stack([np.concatenate([past,a],0) for a in normalized])).cuda().float()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            if model.mode in {'S0','S1'}:generated=model.rollout_frames(frames.expand(4,-1,-1,-1,-1),actions,anchor.expand(4,-1,-1,-1),horizon=8,steps=args.steps,seed=123+case)
            else:
                context=model.normalize(pixels).expand(4,-1,-1,-1,-1) if model.mode.startswith('L') else frames.expand(4,-1,-1,-1,-1)
                states=predict_states(model,context,actions,8)
                aa=anchor.expand(4,-1,-1,-1).repeat_interleave(8,0);generated=model.render_states(states.flatten(0,1),aa,steps=args.steps,seed=123+case).reshape(4,8,3,64,64)
            estimate=pose(generated.flatten(0,1).clamp(-1,1)).float().reshape(4,8,6).cpu().numpy();positions=estimate[:,:,:4]*512
        target=np.asarray(truth)[:,:,:4];distances=((positions[:,None]-target[None])**2).reshape(4,4,8,2,2).sum(-1).mean((2,3))
        assigned=distances.argmin(1);results.append(dict(case=case,episode=ep,start=start,assigned=assigned.tolist(),correct=(assigned==np.arange(4)).tolist(),cost_matrix=distances.tolist()))
    env.close();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);report=dict(cases=len(results),rollouts=4*len(results),action_identification_accuracy=float(np.mean([r['correct'] for r in results])),chance=.25,sampler_steps=args.steps,seconds=time.time()-started,scope='single-seed simulator-reset counterfactual actions from the same context',details=results);(out/'counterfactual.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))

if __name__=='__main__':main()
