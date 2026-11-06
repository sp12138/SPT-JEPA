"""Cache exact known commands within each single observed transition."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def run(a):
    import h5py
    import hdf5plugin
    a.output.mkdir(parents=True,exist_ok=False)
    card=json.loads((a.data/'dataset_card.json').read_text())
    with h5py.File(card['h5_path'],'r')as file:all_controls=np.asarray(file['action'],np.float32)
    for shard in range(4):
        feature=json.loads((a.features/str(shard)/'manifest.json').read_text());records=feature['records']
        controls=np.concatenate([all_controls[r['source_offset']:r['source_offset']+r['length']].reshape(-1,5,4)for r in records])
        path=a.output/str(shard);path.mkdir();np.save(path/'executed_sequence.npy',controls)
        assert controls.shape==(feature['frames'],5,4)and np.isfinite(controls).all()and np.abs(controls).max()<=1
        report=dict(passed=True,frames=len(controls),shape=list(controls.shape),
            records_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
            controls_sha256=hashlib.sha256((path/'executed_sequence.npy').read_bytes()).hexdigest(),
            source='Original HDF5 action20,five actual native4commands within CURRENT observed interval;no future label/reference',model_updates=0)
        (path/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,shards=4,model_updates=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in['data','features','output']:p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
