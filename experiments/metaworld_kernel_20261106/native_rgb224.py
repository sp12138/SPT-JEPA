"""Lossless original RGB224/native-command assets,without NN/GPU/feature reads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(a):
    import h5py,hdf5plugin
    a.output.mkdir(parents=True,exist_ok=False);source=a.native/str(a.shard);parent=json.loads((source/'manifest.json').read_text());assert parent['passed']
    assert digest(source/'known_input.npy')==parent['known_input_sha256']
    ids,offsets,lengths=[np.load(a.data/(name+'.npy'))for name in['episode_ids','offsets','lengths']]
    card=json.loads((a.data/'dataset_card.json').read_text());images=np.lib.format.open_memmap(a.output/'pixels.npy',mode='w+',dtype=np.uint8,shape=(parent['frames'],224,224,3))
    commands=np.load(source/'known_input.npy',mmap_mode='r');checks=0
    with h5py.File(card['h5_path'],'r',rdcc_nbytes=64*1024*1024)as file:
        for row in parent['records']:
            episode=row['episode'];group=np.flatnonzero(ids//5==episode);n=row['length'];offset=row['offset'];assert len(group)==5
            old=np.empty((n,20),np.float32)
            for index in group:
                phase=int(ids[index]%5);start=int(offsets[index]);length=int(lengths[index]);assert length==len(range(phase,n,5))
                raw=np.asarray(file['pixels'][start:start+length]);assert raw.shape==(length,224,224,3)and raw.dtype==np.uint8
                images[offset+phase:offset+n:5]=raw
                old[phase::5]=np.asarray(file['action'][start:start+length],np.float32)
            recovered=np.stack([commands[offset+j:offset+j+n-5]for j in range(5)],1).reshape(n-5,20)
            assert np.array_equal(recovered,old[:n-5]);checks+=n-5
            print(json.dumps(dict(episode=episode,frames=offset+n,command_overlap_checks=checks)),flush=True)
    images.flush();shutil.copyfile(source/'known_input.npy',a.output/'known_input.npy')
    assert digest(a.output/'known_input.npy')==parent['known_input_sha256']
    report=dict(parent,resolution=224,pixels_sha256=digest(a.output/'pixels.npy'),known_input_sha256=parent['known_input_sha256'],
        parent_manifest_sha256=digest(source/'manifest.json'),all_current5_overlap_checks=checks,no_model_or_feature_file_opened=True,
        scope='Originalall5phasesinterleavedEXACTRGB224bytes;noresizing/newdata/pretrainedfeatures.NN/GPU/SGDnotused.'
              'Fullrecordedcontrolsaredataonly;inverseCURRENTintervaltargetsdefinedbyseparatedataset,no futurecommandtrajectorylabels.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in['data','native','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shard',type=int,required=True);run(p.parse_args())
