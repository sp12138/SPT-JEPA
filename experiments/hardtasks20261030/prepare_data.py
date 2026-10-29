"""Freeze splits and train-only action statistics; never consume privileged states in training."""
import argparse,hashlib,json
from pathlib import Path
import h5py,hdf5plugin,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument("task",choices=["cube","tworoom"]);a=p.parse_args()
    b=Path("/efs/prescottsun/SPT-JEPA/data/hardtasks-v1");src=b/a.task
    acquisition=json.loads((src/"acquisition.json").read_text());files=acquisition["files"]
    if len(files)!=1:raise ValueError(files)
    h5=Path(files[0]);out=b/(a.task+"-metadata");out.mkdir(exist_ok=False)
    with h5py.File(h5,"r") as f:
        key="episode_idx" if "episode_idx" in f else "ep_idx"
        eps=np.asarray(f[key]).reshape(-1);steps=np.asarray(f["step_idx"]).reshape(-1)
        starts=np.r_[0,np.flatnonzero(eps[1:]!=eps[:-1])+1];ends=np.r_[starts[1:],len(eps)]
        ids=eps[starts];lengths=ends-starts
        assert len(np.unique(ids))==len(ids)
        for start,end in zip(starts,ends):assert np.array_equal(steps[start:end],np.arange(end-start))
        order=np.random.default_rng(20261030).permutation(ids);n=max(1,len(ids)//20)
        splits=dict(test=np.sort(order[:n]),validation=np.sort(order[n:2*n]),train=np.sort(order[2*n:]))
        actions=np.asarray(f["action"],dtype=np.float32)
        if actions.ndim!=2:raise ValueError(actions.shape)
        eligible=np.isin(eps,splits["train"])&np.isfinite(actions).all(axis=1)
        # Episode-terminal NaN actions are expected; internal ones are forbidden.
        terminal=np.zeros(len(eps),dtype=bool);terminal[ends-1]=True
        assert np.isfinite(actions[~terminal]).all()
        mean=actions[eligible].mean(axis=0,dtype=np.float64);std=actions[eligible].std(axis=0,dtype=np.float64);std=np.where(std<1e-8,1.,std)
        for name,value in dict(offsets=starts,lengths=lengths,episode_ids=ids,**{k+"_episodes":v for k,v in splits.items()}).items():np.save(out/(name+".npy"),value)
        card=dict(task=a.task,h5_path=str(h5),h5_bytes=h5.stat().st_size,acquisition=acquisition,frameskip=5,split_seed=20261030,split_counts={k:len(v) for k,v in splits.items()},raw_frames=len(eps),pixel_shape=list(f["pixels"].shape[1:]),action_dim=actions.shape[1],model_action_dim=actions.shape[1]*5,action_mean=mean.tolist(),action_std=std.tolist(),normalization="training episodes only, finite raw actions; population standard deviation",anchor="first observed frame; no future anchor",training_inputs=["pixels","action"],schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in f.items()})
    (out/"dataset_card.json").write_text(json.dumps(card,indent=2)+"\n");print(json.dumps(card),flush=True)
if __name__=="__main__":main()
