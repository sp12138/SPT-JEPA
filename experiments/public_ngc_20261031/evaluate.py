"""Existing native evaluator, original commands, official NGC terminal-cost seam."""
import argparse
from pathlib import Path
import sys
import numpy as np
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--inner',action='store_true');a=p.parse_args();s=read(a.spec)
    if not a.inner:reexec(s,__file__)
    from objective import Recorder,install
    verify(s);gate=read(s['graph_gate']);assert gate['passed']and len(gate['graphs'])==12
    entry=next(x for x in gate['graphs']if x['id']==s['model_id']);assert sha(entry['report'])==entry['report_sha256']
    graph=read(entry['report']);assert graph['passed']and graph['checkpoint']==s['checkpoint']
    path=Path(entry['report']).with_name('graph.npz');assert sha(path)==graph['graph_sha256']
    assert graph['weights_sha256']==s['immutable_inputs'][str(Path(s['checkpoint'])/'weights.pt')]
    assert graph['cohort_sha256']==sha(s['cohort'])
    with np.load(path)as f:nodes,distances=f['nodes'].copy(),f['graph_distance'].copy()
    assert nodes.shape==(5000,s['dimension'])and np.isfinite(nodes).all()and np.isfinite(distances).all()
    out=Path(s['output']);out.mkdir(exist_ok=True);assert not(out/'metrics.json').exists()
    reference=read(s['reference_metrics']);recorder=Recorder(out)
    from as_jepa_tools import benchmark_evaluate
    assert Path(benchmark_evaluate.__file__).resolve()==Path(s['native_evaluator']).resolve()
    saved=sys.argv;sys.argv=[benchmark_evaluate.__file__]
    for key in ['manifest','checkpoint','data','output']:sys.argv+=['--'+key,s[key]]
    try:
        with install(nodes,distances,s['mode'],recorder,s)as records:benchmark_evaluate.main()
    finally:sys.argv=saved
    assert len(records)==1;r=records[0];model=r.pop('model');r['state_after']=state_hash(model)
    assert r['state_before']==r['state_after']and r['actual_scaler_checked']
    report=read(out/'metrics.json')
    for key in ['pairs','manifest_sha256','checkpoint','solver','plan','normalization','split','protocol','checkpoint_audit']:
        assert report[key]==reference[key],key
    assert len(report['pairs'])==50 and report['normalization']=='train episodes';verify(s)
    replay=None if s['mode']!='l2'else report['metrics']['episode_successes']==reference['metrics']['episode_successes']
    report['ngc_diagnostic']=dict(mode=s['mode'],graph_report=entry['report'],graph_report_sha256=entry['report_sha256'],
        graph_gate_sha256=sha(s['graph_gate']),spec_sha256=sha(a.spec),model_audit=r,shared_bank_audit=recorder.finish(),
        original_replay_successes_exact=replay,reference_metrics=s['reference_metrics'],
        scope='Only terminal cost differs. Original command CEM, scaler, history, initial state, strict success, cases, weights and predictor unchanged. All contemporaneous L2 results retained; no best-of historical selection.')
    write(out/'metrics.json',report);print(json.dumps(dict(id=s['id'],completed=True,replay_equal=replay)),flush=True)


if __name__=='__main__':main()
