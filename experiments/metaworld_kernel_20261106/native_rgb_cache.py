"""Recover original native-frame order from all five lossless temporal phases."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(a):
    import h5py
    import hdf5plugin
    torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False)
    ids,offsets,lengths=[np.load(a.data/(name+'.npy'))for name in['episode_ids','offsets','lengths']]
    train,test=[set(np.load(a.data/(name+'_episodes.npy')).tolist())for name in['train','test']]
    raw_ids=sorted(set((ids//5).tolist()))[a.shard::a.shards]
    groups=[np.flatnonzero(ids//5==episode)for episode in raw_ids]
    total=sum(int(lengths[g].sum())for g in groups)
    images=np.lib.format.open_memmap(a.output/'pixels.npy',mode='w+',dtype=np.uint8,shape=(total,128,128,3))
    inputs=np.lib.format.open_memmap(a.output/'known_input.npy',mode='w+',dtype=np.float32,shape=(total,4))
    card=json.loads((a.data/'dataset_card.json').read_text());records=[];cursor=0;overlap_checks=0
    with h5py.File(card['h5_path'],'r',rdcc_nbytes=64*1024*1024)as file,torch.inference_mode():
        for episode,group in zip(raw_ids,groups):
            n=int(lengths[group].sum());pixels=np.empty((n,224,224,3),np.uint8);controls=np.empty((n,20),np.float32)
            membership=[]
            for source in group:
                phase=int(ids[source]%5);offset=int(offsets[source]);length=int(lengths[source])
                assert length==len(range(phase,n,5))
                pixels[phase::5]=np.asarray(file['pixels'][offset:offset+length])
                controls[phase::5]=np.asarray(file['action'][offset:offset+length],np.float32)
                membership.append('train'if int(ids[source])in train else'test')
                assert int(ids[source])in(train if membership[-1]=='train'else test)
            assert len(set(membership))==1
            # Each PHASE's last macro row is padding,even when later native
            # frames exist in another phase. Recover the four final actual
            # primitive commands from the last valid five-command interval.
            native=np.empty((n,4),np.float32);native[-1]=0
            native[:n-5]=controls[:n-5,:4]
            native[n-5:n-1]=controls[n-6].reshape(5,4)[1:]
            reconstructed=np.stack([native[shift:shift+n-5]for shift in range(5)],1).reshape(n-5,20)
            assert np.array_equal(reconstructed,controls[:n-5]),episode
            overlap_checks+=n-5
            for start in range(0,n,128):
                value=torch.from_numpy(pixels[start:start+128]).cuda().permute(0,3,1,2).float()
                resized=F.interpolate(value,(128,128),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255).byte()
                images[cursor+start:cursor+start+len(value)]=resized.permute(0,2,3,1).cpu().numpy()
            inputs[cursor:cursor+n]=native
            records.append(dict(episode=int(episode),offset=cursor,length=n,split=membership[0]))
            cursor+=n
            if len(records)%25==0:print(json.dumps(dict(episodes=len(records),frames=cursor)),flush=True)
    images.flush();inputs.flush();assert cursor==total
    report=dict(passed=True,shard=a.shard,shards=a.shards,frames=total,native_pairs=total-len(records),records=records,
        native_action_overlap_checks=overlap_checks,terminal_macro_padding_recovered=True,last_native_input_zero_padding=True,
        physical_stride=1,temporal_phase_interleave_exact=True,all_phases_grouped_split=True,
        pixels_sha256=digest(a.output/'pixels.npy'),known_input_sha256=digest(a.output/'known_input.npy'),
        source_split_sha256=digest(a.data/'split-manifest.json'),model_updates=0,
        scope='Original LeRobot RGB and known executed native command only.No learned features,reward,state,success,taskID,or action targets.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,frames=total)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=12);run(p.parse_args())
