"""Derive twelve node-only gates and forty-eight held stable-NGC evaluations."""
import argparse
from pathlib import Path
from common import *

OLD=B/'runs/public_ngc_nodes_20261031'
CAPTURE=B/'runs/public_ngc_radius_diagnostic_20261031'
EVAL=B/'runs/public_ngc_evaluation_20261031'
EXTRA=['stable_cost.py','stable_contracts.py','stable_prepare.py','stable_graph.py','stable_review.py','stable_evaluate.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--graph-root',required=True)
    p.add_argument('--source-record',required=True);a=p.parse_args();root,graphroot=Path(a.root),Path(a.graph_root)
    assert not root.exists()and not graphroot.exists()and root!=graphroot
    helper=read(a.source_record);source_check(helper);here=Path(__file__).parent
    frozen=Path(helper['source'])/'experiments/public_ngc_20261031'
    hashes={f:sha(frozen/f)for f in FILES+EXTRA}
    assert all(sha(here/f)==h for f,h in hashes.items())
    contract=read(here/'stable-cpu-contracts.json');assert contract['passed']and contract['tests']==12
    assert all(hashes[f]==h for f,h in contract['files'].items())
    source=read(OLD/'plan.json');jobs=[];specs={};reuse=0;rebuild=0
    for oldjob in source['jobs']:
        assert sha(oldjob['spec'])==oldjob['spec_sha256'];s=read(oldjob['spec']);verify(s);verify_cohort(s['cohort'],s['data'])
        oldout=Path(s['output']);code=read(OLD/'exits'/(s['id']+'.json'))['returncode']
        if code==0:
            origin_report=oldout/'graph.json';origin_array=oldout/'graph.npz';r=read(origin_report)
            assert r['passed']and sha(origin_array)==r['graph_sha256']and r['state_before']==r['state_after']
            assert r['counts']==dict(rgb=5000,encoder=157,predict=0,action_encoder=0,renderer=0)
            kind='reuse-exact-graph';reuse+=1
        else:
            assert code==1 and s['id']in ['tworoom-TF64-seed1','tworoom-TF64-seed2']
            assert read(CAPTURE/'exits'/(s['id']+'.json'))['returncode']==0
            origin_report=CAPTURE/s['id']/'capture.json';origin_array=origin_report.with_name('nodes.npz');r=read(origin_report)
            assert r['passed']and sha(origin_array)==r['nodes_sha256']and r['state_before']==r['state_after']
            assert r['state_before']==read(oldout/'startup.json')['state_before']
            assert r['original_spec_sha256']==sha(oldjob['spec'])and r['cohort_sha256']==sha(s['cohort'])
            kind='rebuild-fp64-from-saved-nodes';rebuild+=1
        assert r['rng_before']==r['rng_after']
        immutable=dict(s['immutable_inputs'])
        for f in [origin_report,origin_array,Path(oldjob['spec']),OLD/'exits'/(s['id']+'.json'),oldout/'startup.json']:
            immutable[str(f)]=sha(f)
        s.update(output=str(graphroot/s['id']),helper_source=helper,helper_files=hashes,immutable_inputs=immutable,
            numeric_variant='uniform-fp32-direct-attachment',origin_kind=kind,origin_report=str(origin_report),
            origin_array=str(origin_array),origin_spec=oldjob['spec'],graph_gate=str(graphroot/'graph-gate.json'))
        sp=graphroot/'specs'/(s['id']+'.json');write(sp,s);specs[s['id']]=s
        jobs.append(dict(oldjob,hold=True,source=helper,output=s['output'],spec=str(sp),spec_sha256=sha(sp),
            command=[PY,'-B','-u',str(frozen/'stable_graph.py'),'--spec',str(sp)]))
    assert (len(jobs),reuse,rebuild)==(12,10,2)
    oldeval=read(EVAL/'candidate-plan.json');evaljobs=[]
    for j in oldeval['jobs']:
        assert sha(j['spec'])==j['spec_sha256'];s=read(j['spec']);verify(s)
        assert s['checkpoint']==specs[s['model_id']]['checkpoint']
        s.update(output=str(root/s['id']),helper_source=helper,helper_files=hashes,
            numeric_variant='uniform-fp32-direct-attachment',graph_gate=str(graphroot/'graph-gate.json'))
        sp=root/'specs'/(s['id']+'.json');write(sp,s)
        evaljobs.append(dict(j,hold=True,source=helper,output=s['output'],spec=str(sp),spec_sha256=sha(sp),
            requires=[str(graphroot/'graph-gate.json')],command=[PY,'-B','-u',str(frozen/'stable_evaluate.py'),'--spec',str(sp)]))
    assert len(evaljobs)==48
    for target,old,js in [(graphroot,source,jobs),(root,oldeval,evaljobs)]:
        write(target/'candidate-plan.json',dict(old,root=str(target),jobs=js))
        (target/'HOLD_DISPATCH').write_text('New held candidate; original roots and failures retained. No automatic evaluation activation.\n')
    protocol=dict(read(EVAL/'protocol.json'),helper_source=helper,calibration_root=str(graphroot),contracts=contract,
        numeric_variant='uniform-fp32-direct-attachment',original_graph_root=str(OLD),original_evaluation_root=str(EVAL),
        graph_provenance=dict(reused_exact_npz=10,rebuilt_same_fp64_rule=2,saved_capture_root=str(CAPTURE)),
        totals=dict(graph_gates=12,new_images=0,new_encoder_calls=0,new_model_calls=0,eval_jobs=48,cases_per_eval=50),
        gpu_gate='Every original node, FP32 direct vs promoted same-node FP64 direct, finite positive radii and exact self-goal zero. Original CUDA-mm additionally audited for the two failed TF models.',
        activation='Graph gates only after source freeze/actual-free. All12 required and report to root before48eval activation.',
        adaptation='Only NGCCost.attach distance kernel changed, uniformly for all12. FP32 stable sort/k15/anchors4 unchanged. Vendor and FP64 graph algorithm unchanged. No floors, center, deduplication, resampling or tuning.')
    write(root/'protocol.json',protocol);write(graphroot/'protocol.json',protocol)
    print(json.dumps(dict(graph_gates=12,reused_graphs=10,rebuilt_graphs=2,new_rgb=0,evals_held=48,graph_root=str(graphroot),evaluation_root=str(root))))


if __name__=='__main__':main()
