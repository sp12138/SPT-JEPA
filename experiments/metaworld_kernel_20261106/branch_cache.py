"""Pack actual own-intervention observations;share prefixes without duplication."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def pack(a):
    torch.set_num_threads(2);source=a.root/f'tile{a.tile}/manifest.json';manifest=json.loads(source.read_text());assert manifest['passed']and manifest['policy_unchanged']
    records=manifest['records'];a.output.mkdir(parents=True,exist_ok=False);contexts={}
    for r in records:contexts.setdefault(r['context'],[]).append(r)
    n=sum(r['native_frames']-36 for r in records)
    pixels=np.lib.format.open_memmap(a.output/'pixels.npy',mode='w+',dtype=np.uint8,shape=(n,128,128,3))
    inputs=np.lib.format.open_memmap(a.output/'inputs.npy',mode='w+',dtype=np.float32,shape=(n,4))
    prefixes=np.lib.format.open_memmap(a.output/'prefix_pixels.npy',mode='w+',dtype=np.uint8,shape=(len(contexts),6,128,128,3))
    past=np.lib.format.open_memmap(a.output/'prefix_controls.npy',mode='w+',dtype=np.float32,shape=(len(contexts),5,4))
    def resize(x):
        return F.interpolate(torch.from_numpy(x.copy()).permute(0,3,1,2).float(),(128,128),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255).byte().permute(0,2,3,1).numpy()
    offset=0;rows=[];parents={}
    for context_index,(key,branches)in enumerate(sorted(contexts.items())):
        assert len(branches)==29 and len({b['parent']for b in branches})==1 and len({b['split']for b in branches})==1
        parent_path=Path(branches[0]['parent']);parent=np.load(parent_path);t=branches[0]['prefix_stop']-1
        prefixes[context_index]=resize(parent['pixels'][t-5:t+1]);past[context_index]=parent['controls'][t-5:t]
        parents[str(parent_path)]=digest(parent_path)
        for branch in branches:
            path=Path(branch['branch']);assert digest(path)==branch['branch_sha256'];data=np.load(path);length=len(data['controls'])
            assert length in[5,10]and len(data['pixels'])==length and np.isfinite(data['controls']).all()and np.abs(data['controls']).max()<=1
            pixels[offset:offset+length]=resize(data['pixels']);inputs[offset:offset+length]=data['controls']
            rows.append(dict(context=context_index,offset=offset,length=length,split=branch['split'],group=branch['group'],word=branch['word'],source=str(path),source_sha256=branch['branch_sha256']))
            offset+=length
        print(json.dumps(dict(context=key,packed_frames=offset)),flush=True)
    assert offset==n
    for x in[pixels,inputs,prefixes,past]:x.flush()
    report=dict(passed=True,tile=a.tile,records=rows,frames=n,contexts=len(contexts),parent_sha256=parents,source_manifest_sha256=digest(source),
        files={name:digest(a.output/name)for name in['pixels.npy','inputs.npy','prefix_pixels.npy','prefix_controls.npy']},
        learner_fields=['pixels','known_executed_controls'],state_reward_success_or_taskID_arrays=False,
        scope='Separateextra corpusonly;neverautomaticallyinoriginalmodels.Prefix6frames+5ownpastcommands fromsamephysicalparent,branchcontrolappliedBEFOREcorrespondingnextRGB.'
              'Norelabeling,expertcommands,pseudolabels,actiontargets,orprivilegedlearnerfields.Allbranchesofeachparentshareone split.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tile',type=int,required=True);p.add_argument('--output',type=Path,required=True);pack(p.parse_args())
