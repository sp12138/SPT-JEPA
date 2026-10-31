"""Require all twelve complete derived graph gates before evaluation review."""
import argparse
import numpy as np
from common import *


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);plan=read(root/'plan.json')
    assert len(plan['jobs'])==12;rows=[];cohorts={};kinds=[]
    for j in plan['jobs']:
        assert read(root/'exits'/(j['id']+'.json'))['returncode']==0 and sha(j['spec'])==j['spec_sha256']
        s=read(j['spec']);verify(s);verify_cohort(s['cohort'],s['data']);path=Path(j['output'])/'graph.json';r=read(path)
        assert r['passed']and r['spec_sha256']==sha(j['spec'])and r['numeric_variant']=='uniform-fp32-direct-attachment'
        assert r['counts']==dict(rgb=0,encoder=0,predict=0,action_encoder=0,renderer=0)
        assert r['state_before']==r['state_after']and r['rng_before']==r['rng_after']and r['radius_quantiles'][0]>0
        assert r['distance_reference']['ordered_pairs']==25000000 and len(r['bridges'])==r['original_components']-1
        assert sha(path.with_name('graph.npz'))==r['graph_sha256']and sha(path.with_name('radius-audit.npz'))==r['radius_audit_sha256']
        assert sha(r['origin_report'])==r['origin_report_sha256']and sha(r['origin_array'])==r['origin_array_sha256']
        if r['origin_kind']=='reuse-exact-graph':assert r['graph_sha256']==r['origin_array_sha256']
        with np.load(path.with_name('graph.npz'))as f:
            assert f['nodes'].shape==(5000,s['dimension'])and f['graph_distance'].shape==(5000,5000)
            np.testing.assert_array_equal(f['rows'],[x['row']for x in read(s['cohort'])['nodes']])
            assert np.isfinite(f['nodes']).all()and np.isfinite(f['graph_distance']).all()
        if s['task']in cohorts:assert cohorts[s['task']]==r['cohort_sha256']
        cohorts[s['task']]=r['cohort_sha256'];kinds.append(r['origin_kind'])
        rows.append(dict(id=s['id'],report=str(path),report_sha256=sha(path),dimension=s['dimension'],origin_kind=r['origin_kind'],
            radius_quantiles=r['radius_quantiles'],components=r['original_components'],bridges=len(r['bridges']),
            distance_reference=r['distance_reference'],original_cuda_mm=r['original_cuda_mm'],total_seconds=r['total_seconds']))
    assert kinds.count('reuse-exact-graph')==10 and kinds.count('rebuild-fp64-from-saved-nodes')==2
    write(root/'graph-gate.json',dict(passed=True,graphs=rows,cohorts=cohorts,new_rgb=0,new_encoder_calls=0,new_model_calls=0,
        numeric_variant='uniform-fp32-direct-attachment',scope='All12 node-only gates complete. Report to root before any48eval activation.'))
    print(json.dumps(dict(passed=True,graphs=12,new_model_calls=0)))
