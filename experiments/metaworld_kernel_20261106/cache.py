"""Random-observation features and executed single-transition conditioning."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from as_jepa_lewm.predictive_kernel import ObservationFeatures


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    import h5py
    import hdf5plugin
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    data=a.data;card=json.loads((data/'dataset_card.json').read_text())
    ids,offsets,lengths=[np.load(data/(x+'.npy'))for x in['episode_ids','offsets','lengths']]
    train,test=[set(np.load(data/(s+'_episodes.npy')).tolist())for s in['train','test']]
    selected=np.flatnonzero((ids%5==0)&(lengths>=2))[a.shard::4]
    encoder=ObservationFeatures(a.seed).cuda().eval();state={k:v.cpu().clone()for k,v in encoder.state_dict().items()}
    total=int(lengths[selected].sum());features=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(total,32,16,16))
    inputs=np.lib.format.open_memmap(out/'executed_input.npy',mode='w+',dtype=np.float32,shape=(total,4))
    records=[];cursor=0;start=time.monotonic()
    with h5py.File(card['h5_path'],'r',rdcc_nbytes=64*1024*1024)as f,torch.inference_mode():
        for j,ep in enumerate(selected):
            offset,n=int(offsets[ep]),int(lengths[ep]);raw=np.asarray(f['pixels'][offset:offset+n])
            control=np.asarray(f['action'][offset:offset+n],np.float32)
            # This is the known,already executed command for each transition;
            # no action sequence is a label or a planning reference.
            inputs[cursor:cursor+n]=control.reshape(n,5,4).mean(1)
            for k in range(0,n,128):
                x=torch.from_numpy(raw[k:k+128]).cuda().permute(0,3,1,2)
                features[cursor+k:cursor+k+len(x)]=encoder(x).cpu().numpy()
            split='train'if int(ids[ep])in train else'test';assert int(ids[ep])in(train if split=='train'else test)
            records.append(dict(episode=int(ids[ep]),offset=cursor,length=n,split=split,source_offset=offset))
            cursor+=n
            if j%50==0:print(json.dumps(dict(episodes=j,frames=cursor,elapsed=time.monotonic()-start)),flush=True)
    features.flush();inputs.flush();assert cursor==total
    assert all(torch.equal(state[k],v.cpu())for k,v in encoder.state_dict().items())
    result=dict(passed=True,feature_seed=a.seed,shard=a.shard,shards=4,frames=total,records=records,
        encoder_random=True,encoder_trained=False,external_weights=False,model_updates=0,
        features_sha256=digest(out/'features.npy'),executed_input_sha256=digest(out/'executed_input.npy'),
        split_sha256=digest(data/'split-manifest.json'),
        scope='No reconstruction/policy/strongteacher weights. Current executed command is dynamics INPUT only;targets are future observation features.No future action labels,action references,reward,state,success,taskID inputs.')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,frames=total,seed=a.seed,shard=a.shard)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--shard',type=int,required=True);run(p.parse_args())
