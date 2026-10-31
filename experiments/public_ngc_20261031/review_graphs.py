"""All12 completed graphs are mandatory before any independent eval activation."""
import argparse
from pathlib import Path
import numpy as np
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    plan=read(root/'plan.json');assert len(plan['jobs'])==12;rows=[];cohorts={}
    for j in plan['jobs']:
        assert read(root/'exits'/(j['id']+'.json'))['returncode']==0
        assert sha(j['spec'])==j['spec_sha256'];s=read(j['spec']);verify(s);verify_cohort(s['cohort'],s['data'])
        rpath=Path(j['output'])/'graph.json';r=read(rpath)
        assert r['passed']and r['spec_sha256']==sha(j['spec'])and r['state_before']==r['state_after']and r['rng_before']==r['rng_after']
        assert r['counts']==dict(rgb=5000,encoder=157,predict=0,action_encoder=0,renderer=0)
        assert r['radius_quantiles'][0]>0 and len(r['bridges'])==r['original_components']-1
        assert sha(rpath.with_name('graph.npz'))==r['graph_sha256']
        with np.load(rpath.with_name('graph.npz'))as f:
            assert f['nodes'].shape==(5000,s['dimension'])and f['graph_distance'].shape==(5000,5000)
            assert np.isfinite(f['nodes']).all()and np.isfinite(f['graph_distance']).all()
            np.testing.assert_array_equal(f['rows'],[x['row']for x in read(s['cohort'])['nodes']])
        if s['task']in cohorts:assert cohorts[s['task']]==r['cohort_sha256']
        cohorts[s['task']]=r['cohort_sha256']
        rows.append(dict(id=s['id'],report=str(rpath),report_sha256=sha(rpath),dimension=s['dimension'],
            radius_quantiles=r['radius_quantiles'],components=r['original_components'],bridges=len(r['bridges']),
            encoded_seconds=r['encoded_seconds'],graph_seconds=r['graph_seconds']))
    write(root/'graph-gate.json',dict(passed=True,graphs=rows,cohorts=cohorts,rgb_frames=60000,encoder_calls=1884,
        planner_calls=0,scope='Integrity gate only; no planning or performance evidence. Root must separately authorize48 evaluation activation.'))
    print(json.dumps(dict(passed=True,graphs=12,graph_gate=str(root/'graph-gate.json'))))


if __name__=='__main__':main()
