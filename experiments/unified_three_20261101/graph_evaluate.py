"""Same public NGC seam on the three native task evaluators; no learner."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

from common import B, read, sha, state_hash, write
from graph import encoding_state
from vforge.ops import OpsConfig, initialize_ops


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True)
    p.add_argument('--step',required=True,type=int);p.add_argument('--mode',choices=['l2','ngc'],required=True)
    p.add_argument('--limit',type=int,choices=[2,50],default=50)
    p.add_argument('--case-set',choices=['primary','fresh'],default='primary');a=p.parse_args()
    root=Path(a.root);protocol=read(root/'protocol.json');row=next(r for r in protocol['runs'] if r['id']==a.id)
    assert row['arm'] in ['flat','field']
    cfg=read(root/a.id/'config.json');checkpoint=root/a.id/f'model-step{a.step:07d}'
    graph_path=root/'graphs'/f'{row["task"]}-seed{row["seed"]}'/'graph.json';graph=read(graph_path)
    assert graph['passed'] and sha(graph['graph'])==graph['graph_sha256']
    current=torch.load(checkpoint/'weights.pt',map_location='cpu',weights_only=True,mmap=True)
    original=torch.load(graph['weights'],map_location='cpu',weights_only=True,mmap=True)
    left,right=encoding_state(current),encoding_state(original)
    assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
    del current,original,left,right
    public=Path(__file__).resolve().parents[1]/'public_ngc_20261031';sys.path.insert(0,str(public))
    import objective
    from stable_cost import StableNGCCost
    objective.NGCCost=StableNGCCost
    with np.load(graph['graph']) as f:nodes,distances=f['nodes'].copy(),f['graph_distance'].copy()
    suffix=('' if a.case_set=='primary' else '-fresh')+('' if a.limit==50 else '-limit2')
    out=root/'graph-evaluations'/f'{a.id}-step{a.step}-{a.mode}{suffix}';out.mkdir(parents=True,exist_ok=False)
    recorder=objective.Recorder(out);previous=sys.argv
    try:
        if row['task']=='pusht':
            from as_jepa_tools import planner_entry
            cache=B/'data/hardtasks-v1/planning';assert(cache/'datasets/pusht_expert_train.h5').exists()
            seed=42 if a.case_set=='primary' else 1042
            env=dict(ASJEPA_UPSTREAM=str(B/'sources/stable-worldmodel'),ASJEPA_PLANNER_OUTPUT=str(out),
                     ASJEPA_PLANNER_SPLIT='heldout',ASJEPA_PLANNER_SEED=str(seed),
                     ASJEPA_TEST_EPISODES=str(Path(cfg['data'])/'test_episodes.npy'),STABLEWM_HOME=str(cache))
            os.environ.update(env)
            sys.argv=[planner_entry.__file__,f'policy={checkpoint}',f'+cache_dir={cache}',f'eval.num_eval={a.limit}',f'seed={seed}',
                      '+plan_config.history_len=3','bf16=false','compile=false',f'hydra.run.dir={out}/hydra',
                      f'world.num_envs={a.limit}','--config-path',str(B/'sources/stable-worldmodel/scripts/plan/config')]
            initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
            with objective.install(nodes,distances,a.mode,recorder) as records:planner_entry.main()
        else:
            from as_jepa_tools import benchmark_evaluate
            manifest=str(Path(cfg['data'])/'manifests/validation-strict.json')
            if a.case_set=='fresh':
                plan=read(B/'runs/trajectory_search_local_20261101/candidate-plan.json')
                candidates=[read(j['spec']) for j in plan['jobs']]
                selected=[s['manifest'] for s in candidates if s['data']==cfg['data'] and s['case_set'] in ['fresh',row['task']+'-fresh']]
                assert selected and len(set(selected))==1
                manifest=selected[0]
            sys.argv=[benchmark_evaluate.__file__,'--manifest',manifest,
                      '--checkpoint',str(checkpoint),'--data',cfg['data'],'--output',str(out)]
            if a.limit!=50:sys.argv+=['--limit',str(a.limit)]
            spec=dict(model_source=protocol['source'],data=cfg['data'])
            with objective.install(nodes,distances,a.mode,recorder,spec) as records:benchmark_evaluate.main()
    finally:sys.argv=previous
    assert len(records)==1;r=records[0];model=r.pop('model');after=state_hash(model)
    assert after==r['state_before']
    report=read(out/'metrics.json');assert len(report['metrics']['episode_successes'])==a.limit
    assert Path(r['model_source']).is_relative_to(Path(protocol['source']['source']))
    write(out/'graph-audit.json',dict(passed=True,mode=a.mode,step=a.step,case_set=a.case_set,model_audit=r,
          state_after=after,graph_report=str(graph_path),graph_report_sha256=sha(graph_path),
          paired_recording=recorder.finish(),weights_sha256=sha(checkpoint/'weights.pt'),
          metrics_sha256=sha(out/'metrics.json'),extra_model_calls=0,
          scope='Public NGC control only;both costs recorded with identical planner/data/model contract. Original task predicates retained.'))
    print('Graph evaluation complete',a.id,a.mode,flush=True)


if __name__=='__main__':main()
