"""Same public NGC seam on the three native task evaluators; no learner."""
import argparse
from contextlib import ExitStack, contextmanager
import importlib.util
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
    p.add_argument('--case-set',choices=['primary','fresh'],default='primary')
    p.add_argument('--proposal',choices=['original','random','retrieval'])
    p.add_argument('--bank-report');a=p.parse_args()
    assert bool(a.proposal)==bool(a.bank_report),'Proposal controls require an explicit fixed train-only bank'
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
    bank=None;trajectory_records=[]
    if a.proposal:
        bank=read(a.bank_report)
        assert bank['passed'] and bank['count']==1024 and bank['no_validation_test_overlap']
        assert bank['task']==row['task'] and sha(bank['bank'])==bank['bank_sha256']
        with np.load(bank['bank']) as f:features,actions=f['features'].copy(),f['actions'].copy()
        adapter_path=Path(__file__).resolve().parents[1]/'trajectory_search_20261101/adapter.py'
        spec=importlib.util.spec_from_file_location('unified_fixed_trajectory_adapter',adapter_path)
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)

    @contextmanager
    def controls(specification=None):
        with ExitStack() as stack:
            records=stack.enter_context(objective.install(nodes,distances,a.mode,recorder,specification))
            if a.proposal:
                states=stack.enter_context(adapter.install(a.proposal,features,actions,bank,protocol['source']['source']))
            yield records
            if a.proposal:trajectory_records.extend(states)

    suffix=('' if a.case_set=='primary' else '-fresh')+('' if a.limit==50 else '-limit2')
    if a.proposal:suffix+='-'+a.proposal
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
            with controls() as records:planner_entry.main()
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
            with controls(spec) as records:benchmark_evaluate.main()
    finally:sys.argv=previous
    assert len(records)==1;r=records[0];model=r.pop('model');after=state_hash(model)
    assert after==r['state_before']
    report=read(out/'metrics.json');assert len(report['metrics']['episode_successes'])==a.limit
    assert Path(r['model_source']).is_relative_to(Path(protocol['source']['source']))
    if a.proposal:
        assert len(trajectory_records)==1
        state=trajectory_records[0]
        assert state.model is model and state.state_before==after
        assert state.predict_calls==5*state.calls and len(state.injections)*30==state.calls
        assert sha(bank['bank'])==bank['bank_sha256']
        write(out/'trajectory-search.json',dict(passed=True,mode=a.proposal,
              bank_report=a.bank_report,bank_report_sha256=sha(a.bank_report),
              adapter_sha256=sha(adapter_path),state_before=state.state_before,state_after=after,
              cost_calls=state.calls,predict_calls=state.predict_calls,encoder_calls=state.encoder_calls,
              solves=state.solves,injections=state.injections,candidates_per_cost=300,
              first_iteration_replacements=0 if a.proposal=='original' else 64,
              extra_model_calls=0,training_steps=0,optimal_action_labels=0,
              action_execution='Unmodified native CEM elite return,policy inverse scaler and action history.'))
    write(out/'graph-audit.json',dict(passed=True,mode=a.mode,step=a.step,case_set=a.case_set,model_audit=r,
          state_after=after,graph_report=str(graph_path),graph_report_sha256=sha(graph_path),
          paired_recording=recorder.finish(),weights_sha256=sha(checkpoint/'weights.pt'),
          metrics_sha256=sha(out/'metrics.json'),extra_model_calls=0,
          proposal=a.proposal,
          scope='Public NGC and optional fixed train-library controller control;both costs recorded with identical planner/data/model contract. Original task predicates retained.'))
    print('Graph evaluation complete',a.id,a.mode,flush=True)


if __name__=='__main__':main()
