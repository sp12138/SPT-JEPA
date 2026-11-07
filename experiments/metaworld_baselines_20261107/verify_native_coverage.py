"""Real terminal/short-trajectory coverage and native sampler resume checks."""
import argparse
import pickle
import random
import numpy as np
import torch

from experiments.metaworld_baselines_20261107.train_native import Windows,Batches,save


def main(output):
    torch.set_num_threads(2);root='/efs/prescottsun/SPT-JEPA/data/metaworld-lerobot-b46a1c18-packed-v1'
    rows=[];rng=(torch.get_rng_state().clone(),pickle.dumps(np.random.get_state()),pickle.dumps(random.getstate()))
    for split in ['train','test']:
        long=Windows(root,8,1,split)
        for n in [2,4,8]:
            data=Windows(root,n,1,split);base=data.base;base.open()
            expected=int(np.maximum(base.lengths[np.isin(base.episode_ids,np.load(base.root/(split+'_episodes.npy')))]-n+1,0).sum())
            assert len(data)==expected
            chosen=np.linspace(0,len(base.episodes)-1,10,dtype=np.int64)
            checked=0
            for pos in chosen:
                ep=int(base.episodes[pos]);start=int(base.lengths[ep])-n
                index=(int(base.ends[pos-1]) if pos else 0)+start
                assert base.locate(index)==(ep,start)
                item=data[index];offset=int(base.offsets[ep]);raw=np.asarray(base.pixels[offset+start:offset+start+n])
                golden=torch.from_numpy(raw.copy()).permute(0,3,1,2)
                assert torch.equal(item['pixels'],golden)
                expected_action=(base.actions[offset+start:offset+start+n-1]-base.mean)/base.std
                assert torch.equal(item['action'][:-1],torch.from_numpy(expected_action))
                assert torch.count_nonzero(item['action'][-1])==0
                assert set(item)=={'pixels','action'};checked+=1
            short=set(base.episodes)-set(long.base.episodes)
            assert all(base.lengths[e]>=n and base.lengths[e]<8 for e in short)
            def combined(begin,end):
                r=[list(Batches(len(data),7,begin,end,rank,2,128)) for rank in [0,1]]
                return [r[0][i]+r[1][i] for i in range(end-begin)]
            assert combined(0,5)[3:]==combined(3,5)
            assert all(len(set(v))==256 for v in combined(0,5))
            per=len(data)//256
            expected0=torch.randperm(len(data),generator=torch.Generator().manual_seed(7))[(per-1)*256:per*256].tolist()
            expected1=torch.randperm(len(data),generator=torch.Generator().manual_seed(7+1000003))[:256].tolist()
            assert combined(per-1,per+1)==[expected0,expected1]
            rows.append(dict(split=split,frames=n,native_windows=len(data),terminal_windows_checked=checked,
                additional_short_phase_episodes=len(short),all_eligible_phase_frames_reachable=True,steps_per_epoch=per))
    assert torch.equal(rng[0],torch.get_rng_state()) and rng[1]==pickle.dumps(np.random.get_state()) and rng[2]==pickle.dumps(random.getstate())
    save(output,dict(passed=True,rows=rows,terminal_window_pixel_action_checks=60,
        causal_action_padding=True,short_trajectory_inclusion=True,native_step3_resume_and_epoch_boundary=True,
        global_rng_unchanged=True,model_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
