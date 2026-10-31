"""Constructive output-family rank, not a real-data reachability claim."""
import hashlib,json,os
from pathlib import Path
import torch
from as_jepa_lewm.dynamic_readout import DynamicReadout

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(2)
dynamic=DynamicReadout('state');static=DynamicReadout('static')
with torch.no_grad():
    for p in dynamic.parameters():p.zero_()
    dynamic.input.weight[0,0]=1
    dynamic.output.weight[0,0]=1
static.load_state_dict(dynamic.state_dict(),strict=True)
#64 possible current states and16 possible old increments from one fixed16-D
#output subspace. The actual head is shared over all64 rows as in production.
states=torch.zeros(64,16,64,16)
for i in range(64):states[i,:,i,0]=1
delta=torch.zeros_like(states)
for c in range(16):delta[:,c,0,c]=1
states=states.flatten(-2);prediction=states+delta.flatten(-2)
with torch.no_grad():
    increments={k:(m(states,prediction)-states).reshape(64,16,64,16)for k,m in [('state',dynamic),('static',static)]}
t=torch.tanh(torch.tensor(1.));expected=t*torch.eye(64);expected[:,0]+=1
rows=[]
for c in range(16):
    matrix=increments['state'][:,c,:,c]
    torch.testing.assert_close(matrix,expected,atol=3e-7,rtol=0)
    others=[j for j in range(16)if j!=c]
    assert bool((increments['state'][:,c,:,others]==0).all())
    rank=int(torch.linalg.matrix_rank(matrix));static_rank=int(torch.linalg.matrix_rank(increments['static'][:,c,:,c]))
    assert rank==64 and static_rank==1
    rows.append(dict(channel=c,dynamic_rank=rank,static_rank=static_rank,
        minimum_singular_value=float(torch.linalg.svdvals(matrix)[-1])))
result=dict(passed=True,device='CPU',cuda_initialized=torch.cuda.is_initialized(),
    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    model_sha256=hashlib.sha256(Path(__import__('as_jepa_lewm.dynamic_readout',fromlist=['__file__']).__file__).read_bytes()).hexdigest(),
    old_increment_subspace_rank=16,state_conditioned_global_increment_span_rank=sum(x['dynamic_rank']for x in rows),
    static_global_increment_span_rank=sum(x['static_rank']for x in rows),rows=rows,
    construction='M_i0=tanh(u_i0);all other columns0.64 states have only slot i/channel0=1. Olddelta is e_(slot0,channelc). Eachchannel produces e0+t*ei; determinant=(1+t)*t**63 nonzero. Disjointchannel blocks yield rank1024.',
    limitation='Counterexample across arbitrary current states, not evidence these states occur or are reachable from one physical initial state. Fixed-state instantaneous action rank remains at most oldrank. Configured witness weights are not initialization/training results. No world-model/encoder/physics call or optimizer.')
assert not result['cuda_initialized']
path=Path(os.environ['DYNAMIC_READOUT_WITNESS_OUTPUT'])
with path.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='rows'}),flush=True)
