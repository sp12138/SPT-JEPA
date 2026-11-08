"""CPU mathematical checks and factual development diagnostics; never fitting."""
import argparse
import json
from pathlib import Path
import torch
from experiments.metaworld_transition_only_20261109.common import read,save,now,sha
from experiments.metaworld_transition_only_20261109.data import FactualWindows
from .metric import basis,cost


def run(output):
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    matrix=basis(128,'cpu');assert torch.allclose(matrix@matrix.T,torch.eye(128),atol=2e-6,rtol=1e-5)
    torch.manual_seed(7);a=torch.rand(2,3,128,128);b=torch.rand(1,3,128,128)
    assert torch.equal(cost(a,a),torch.zeros(2))
    assert torch.allclose(cost(a,b),torch.cat([cost(x[None],b) for x in a]),atol=1e-8,rtol=1e-5)
    assert (cost(a,b)>0).all()
    # A fixed8pixel square is moved horizontally away from the fixedgoal.
    goal=torch.zeros(1,3,128,128);goal[:,:,60:68,80:88]=1
    toy=[]
    for displacement in [0,1,2,4,8,12,16,24,32,48,64]:
        image=torch.zeros_like(goal);image[:,:,60:68,80-displacement:88-displacement]=1
        toy.append(dict(displacement=displacement,pixel_mse=float((image-goal).square().mean()),spatial_cost=float(cost(image,goal))))
    plateau=[r for r in toy if r['displacement']>=8]
    assert len({r['pixel_mse'] for r in plateau})==1
    assert all(a['spatial_cost']<b['spatial_cost'] for a,b in zip(toy,toy[1:]))
    root=Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_transition_only_20261109')
    diag=read(root/'factual-diagnostics/SUMMARY.json')
    selected=read(root/'factual-diagnostics/OBSERVATION_COPY_BASELINE.json')['selected_heldout_windows']
    data=FactualWindows('/tmp/prescottsun-factual-rgb128-20261109-v1','test',maximum_horizon=60)
    rows=[]
    for chosen in selected:
        r,t=data.locate(chosen['index']);pixels=data.shards[r['shard']]['pixels'];offset=r['offset']+t
        positions=[0,10,20,30,40,50,60]
        frames=torch.stack([torch.from_numpy(pixels[offset+p].copy()).permute(2,0,1) for p in positions]).float()/255.
        g=frames[-1:];raw=(frames-g).square().mean((1,2,3));spatial=cost(frames,g)
        pairs=[(i,j) for i in range(6) for j in range(i+1,6)]
        rows.append(dict(chosen,native_positions=positions,pixel=raw.tolist(),spatial=spatial.tolist(),
            ordered_pairs=len(pairs),pixel_correct=sum(float(raw[i])>float(raw[j]) for i,j in pairs),
            spatial_correct=sum(float(spatial[i])>float(spatial[j]) for i,j in pairs)))
    total=sum(r['ordered_pairs'] for r in rows);pixel=sum(r['pixel_correct'] for r in rows);spatial=sum(r['spatial_correct'] for r in rows)
    save(output,dict(passed=True,utc=now(),orthonormal_dct=True,zero_and_positive_distance=True,sample_independence=True,
        toy=toy,toy_disjoint_pixel_plateau=True,toy_spatial_strictly_increases=True,heldout_windows=rows,
        factual_pair_counts=dict(pixel=pixel,spatial=spatial,total=total),metric_source_sha256=sha(Path(__file__).with_name('metric.py')),
        model_updates=0,head_updates=0,gpu_contexts=0,physics_steps=0,goal_conditioned_training=False,
        scope='Fixedanalyticoperator/no fittedparameters.16correlatedheldoutclips;factual ordering is NOTactualcontrol,causalranking,or noveltyproof. '
            'A forecast-horizon problem and a spatial-cost dead-zone problem are separate hypotheses.'))
    print('CPUmetricPASS',dict(pixel=pixel,spatial=spatial,total=total),toy,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
