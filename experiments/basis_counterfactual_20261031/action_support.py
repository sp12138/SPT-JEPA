"""Read only all training-episode HDF5 actions; no images or heldout rows."""
import hashlib,json
from pathlib import Path
import h5py,hdf5plugin
import numpy as np

B=Path('/efs/prescottsun/SPT-JEPA');D=B/'data/hardtasks-v1/tworoom-metadata'
OUT=B/'reports/basis-counterfactual-20261031/action-support'


class Counts:
    def __init__(self):
        self.rows=0;self.finite_rows=0;self.finite_coordinates=0;self.out_rows=0;self.out_coords=0;self.out_tolerance=0
        self.minimum=np.full(2,np.inf);self.maximum=np.full(2,-np.inf);self.at_boundary=0
    def add(self,a):
        finite=np.isfinite(a);self.rows+=len(a);self.finite_rows+=int(finite.all(1).sum());self.finite_coordinates+=int(finite.sum())
        outside=finite&(np.abs(a)>1);self.out_rows+=int(outside.any(1).sum());self.out_coords+=int(outside.sum())
        self.out_tolerance+=int((finite&(np.abs(a)>1+1e-6)).sum());self.at_boundary+=int((finite&(np.abs(a)==1)).sum())
        for i in range(2):
            x=a[:,i][finite[:,i]]
            if len(x):self.minimum[i]=min(self.minimum[i],float(x.min()));self.maximum[i]=max(self.maximum[i],float(x.max()))
    def result(self):
        return dict(rows=self.rows,finite_rows=self.finite_rows,nonfinite_rows=self.rows-self.finite_rows,
            finite_coordinates=self.finite_coordinates,minimum=[float(x) if np.isfinite(x) else None for x in self.minimum],
            maximum=[float(x) if np.isfinite(x) else None for x in self.maximum],
            outside_action_rows=self.out_rows,outside_coordinates=self.out_coords,
            outside_coordinate_fraction=self.out_coords/self.finite_coordinates if self.finite_coordinates else None,
            outside_coordinates_tolerance1e6=self.out_tolerance,exact_boundary_coordinates=self.at_boundary)


def main():
    card=json.loads((D/'dataset_card.json').read_text());train=np.load(D/'train_episodes.npy')
    ids,offsets,lengths=[np.load(D/f) for f in ['episode_ids.npy','offsets.npy','lengths.npy']]
    selected=np.flatnonzero(np.isin(ids,train));assert len(selected)==len(train)==9000
    all_rows,used=Counts(),Counts();h=hashlib.sha256();eligible=0;terminal=Counts()
    with h5py.File(card['h5_path'],'r') as f:
        assert np.array_equal(np.asarray(f['ep_offset']),offsets) and np.array_equal(np.asarray(f['ep_len']),lengths)
        for ep in selected:
            offset,n=int(offsets[ep]),int(lengths[ep]);a=np.asarray(f['action'][offset:offset+n],dtype=np.float32)
            assert np.all(np.asarray(f['ep_idx'][offset:offset+n])==ids[ep])
            all_rows.add(a);terminal.add(a[-1:]);h.update(a.tobytes())
            # Union of all8-frame/stride5 training windows consumes rows0..n-2.
            if n>35:eligible+=1;used.add(a[:-1])
    result=dict(task='tworoom',training_episodes=9000,eligible_window8_stride5_episodes=eligible,
        all_training_rows=all_rows.result(),actual_training_transition_support=used.result(),terminal_rows=terminal.result(),
        ordered_training_action_bytes_sha256=h.hexdigest(),dataset_card_sha256=hashlib.sha256((D/'dataset_card.json').read_bytes()).hexdigest(),
        train_episodes_sha256=hashlib.sha256((D/'train_episodes.npy').read_bytes()).hexdigest(),h5_path=card['h5_path'],acquisition=card['acquisition'],
        scope='All and only training episode actions read, episode by episode; no images/model/heldout action rows. '
              'Actual support is the union of every valid BenchmarkWindows(window8,stride5) outgoing action block, excluding '
              'episode terminal rows and episodes too short for a window. This is possible training support, not sample multiplicity at2000 steps.')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
