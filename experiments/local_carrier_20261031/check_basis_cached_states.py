"""CPU floating-point contract on saved observed states; no encoder or labels."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from as_jepa_lewm.spatial_carrier import orthogonal_basis

B=Path('/efs/prescottsun/SPT-JEPA')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2);q=torch.from_numpy(orthogonal_basis()).double();records=[]
    for task in ('tworoom','cube'):
        for seed in (1,2):
            folder=B/f'runs/local_isotropic2000_20261031/transition-{task}-spatial-local-seed{seed}'
            audit=json.loads((folder/'state-audit.json').read_text());path=folder/'state-audit.npz'
            assert audit['state_dim']==1024 and audit['cases']==64
            with np.load(path) as saved:x=torch.from_numpy(saved['observed']).flatten(0,1).double()
            assert x.shape==(512,1024)
            y=(x@q).float().double();dx=torch.pdist(x);dy=torch.pdist(y)
            i,j=torch.triu_indices(len(x),len(x),offset=1);norm=x.norm(dim=-1)
            # Conservative error bound for FP32 output rounding plus FP64 accumulation.
            bound=4*torch.finfo(torch.float32).eps*(norm[i]+norm[j])+1e-10
            assert bool(((dx-dy).abs()<=bound).all())
            assert bool((dy[dx==0]==0).all())
            nz=dx>0;inverse=(y@q.T-x).abs().max().item()
            records.append(dict(task=task,seed=seed,path=str(path),state_audit_sha256=sha(folder/'state-audit.json'),
                cached_array_sha256=sha(path),frames=512,pairs=len(dx),distinct_pairs=int(nz.sum()),
                max_euclidean_distance_abs_error=float((dy-dx).abs().max()),
                max_mean_squared_goal_cost_abs_error=float(((dy.square()-dx.square())/1024).abs().max()),
                max_nonzero_goal_cost_relative_error=float(((dy[nz].square()-dx[nz].square()).abs()/dx[nz].square()).max()),
                max_inverse_abs_error=inverse,max_error_to_bound=float(((dy-dx).abs()/bound).max()),passed=True))
    result=dict(passed=True,device='CPU',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),records=records,
        basis_sha256=hashlib.sha256(q.float().numpy().tobytes()).hexdigest(),helper_sha256=sha(__file__),
        scope='Saved fixed local1024 observed states only. Q fixed before examining data. No model call, training, labels or validation tuning.')
    target=Path(os.environ['SPATIAL_CACHED_CONTRACT_OUTPUT']);target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
