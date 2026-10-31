"""Prepare exactly48 held correlated-proposal L2/NGC endpoints."""
import argparse
import numpy as np
from run import *

WHITE=B/'runs/public_ngc_stable_evaluation_v2_20261031'
F4=B/'runs/action_covariance_evaluation_20261031'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-record');a=p.parse_args();root=Path(a.root)
    assert not root.exists();source=read(a.source_record)if a.source_record else None
    here=Path(__file__).parent;frozen=Path(source['source'])/'experiments/public_ngc_covariance_20261031'if source else here
    files={f:sha(frozen/f)for f in FILES};assert all(sha(here/f)==h for f,h in files.items())
    contract=read(here/'cpu-contracts.json');assert contract['passed']and contract['helper_files']==files
    ngc,corr=read(NGC),read(CORR);source_check(ngc);source_check(corr)
    if source:source_check(source)
    old=read(WHITE/'plan.json');assert len(old['jobs'])==48 and all(read(WHITE/'exits'/(j['id']+'.json'))['returncode']==0 for j in old['jobs'])
    assert (WHITE/'HOLD_DISPATCH').exists();gate=read(B/'runs/public_ngc_stable_nodes_v2_20261031/graph-gate.json');assert gate['passed']and len(gate['graphs'])==12
    calibrations={};cardhash={}
    for task in ['tworoom','cube']:
        path=F4/'calibration'/task/'calibration.json';cal=read(path);matrix=path.with_name('matrix.npz')
        assert sha(matrix)==cal['matrix_sha256']and cal['zero_variance_coordinates']==0 and cal['no_validation_overlap']
        assert cal['cohort_count']==cal['distinct_training_episodes']==128 and sha(cal['manifest'])==cal['manifest_sha256']
        with np.load(matrix)as f:L=f['cholesky'];R=f['correlation']
        assert np.array_equal(L,np.tril(L))and np.all(np.diag(L)>0)
        np.testing.assert_allclose(L@L.T,R,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(np.square(L).sum(1),1,rtol=1e-12,atol=1e-12)
        calibrations[task]=dict(path=str(path),calibration_sha256=sha(path),matrix_sha256=sha(matrix),dimension=len(L),cohort_sha256=cal['manifest_sha256'])
        cardhash[task]=cal['dataset_card_sha256']
    jobs=[];modelbindings={};checked_sources=set()
    digest={}
    def checked(path,h):
        if path not in digest:digest[path]=sha(path)
        assert digest[path]==h,path
    for j in old['jobs']:
        assert sha(j['spec'])==j['spec_sha256'];s=read(j['spec']);ref=Path(j['output'])/'metrics.json';m=read(ref)
        assert s['helper_source']==ngc and m['ngc_diagnostic']['numeric_variant']=='uniform-fp32-direct-attachment'
        if s['model_source']['manifest']not in checked_sources:
            source_check(s['model_source']);checked_sources.add(s['model_source']['manifest'])
        for path,h in s['immutable_inputs'].items():checked(path,h)
        assert cardhash[s['task']]==sha(Path(s['data'])/'dataset_card.json')
        cal=read(calibrations[s['task']]['path']);card=read(Path(s['data'])/'dataset_card.json')
        np.testing.assert_array_equal(cal['action_mean'],card['action_mean']);np.testing.assert_array_equal(cal['action_std'],card['action_std'])
        assert sha(s['graph_gate'])==m['ngc_diagnostic']['graph_gate_sha256']
        checked(m['ngc_diagnostic']['graph_report'],m['ngc_diagnostic']['graph_report_sha256'])
        graph=read(m['ngc_diagnostic']['graph_report']);checked(str(Path(m['ngc_diagnostic']['graph_report']).with_name('graph.npz')),graph['graph_sha256'])
        for key in ['pairs','manifest_sha256','checkpoint','solver','plan','normalization','split','protocol','checkpoint_audit']:
            assert m[key]==read(s['reference_metrics'])[key]
        assert m['ngc_diagnostic']['model_audit']['state_before']==m['ngc_diagnostic']['model_audit']['state_after']
        c=calibrations[s['task']];immutable=dict(s['immutable_inputs']);immutable[str(ref)]=sha(ref)
        for path in [c['path'],str(Path(c['path']).with_name('matrix.npz')),str(NGC),str(CORR),s['graph_gate']]:immutable[path]=sha(path)
        ident='corr-'+j['id'];s.update(id=ident,output=str(root/ident),controller_mode='correlated',reference_metrics=str(ref),
            immutable_inputs=immutable,combined_immutable_inputs=immutable,combined_helper_source=source,combined_helper_files=files,
            correlation_source=corr,calibration=c['path'],white_spec=j['spec'],white_spec_sha256=j['spec_sha256'])
        s['combined_immutable_inputs'][j['spec']]=j['spec_sha256']
        sp=root/'specs'/(ident+'.json');write(sp,s)
        jobs.append(dict(j,id=ident,hold=True,source=source,output=s['output'],spec=str(sp),spec_sha256=sha(sp),
            depends_on=['corr-'+x for x in j.get('depends_on',[])],
            command=[PY,'-B','-u',str(frozen/'run.py')if source else'<FROZEN_HELPER_REQUIRED>','--spec',str(sp)]))
        modelbindings[s['model_id']]=dict(checkpoint=s['checkpoint'],model_source=s['model_source'],dimension=s['dimension'],step=s['step'],
            weights_sha256=immutable[str(Path(s['checkpoint'])/'weights.pt')])
    assert len(jobs)==48 and len(modelbindings)==12
    write(root/'candidate-plan.json',dict(old,root=str(root),jobs=jobs));(root/'HOLD_DISPATCH').write_text('New48 held candidate. Root protocol/source review required before activation. All old holds remain.\n')
    write(root/'protocol.json',dict(helper_source=source,ngc_source=ngc,correlation_source=corr,white_root=str(WHITE),
        graph_gate=str(B/'runs/public_ngc_stable_nodes_v2_20261031/graph-gate.json'),graph_gate_sha256=sha(B/'runs/public_ngc_stable_nodes_v2_20261031/graph-gate.json'),
        models=modelbindings,calibrations=calibrations,contracts=contract,slots=sorted({(j['host'],j['gpu'])for j in jobs}),
        totals=dict(new_evaluations=48,reference_evaluations=48,new_rgb_for_graph=0,new_graphs=0,training=0,fits=0,cases_per_eval=50),
        definition='Frozen f4 correlated proposals with stable-v2 L2/NGC goal costs. Same12 frozen checkpoints, both tasks/seeds, original primary/fresh50. Existing white48 supply other2cells of2x2.',
        fairness='Same Cholesky per task, unit initial coordinate marginals, exact original scalers/commands,300x30 top30,H5/history3/block5/receding5/budget50. No clipping, warmstart tail, extra model rollout, changed metric parameters or graph nodes.',
        comparisons='Within newcorrelated: NGC−L2. Factorial: (corrNGC−corrL2)−(whiteNGC−whiteL2), case-paired and bothseeds. Each cell retained; no best-of selection. Marginal variances can evolve differently under intended elite updates.',
        boundaries='Public controller combination, not new learning algorithm. Fixed-model/node-query checks do not guarantee planning; errors propagate without fallback. Costs instrumented inboth arms, no bare-L2 speed claim. Model capacities/exposures differ; prior repeat variability retained.',
        activation='CPU contracts and frozen source reviewed byroot first; actual16slots free check; no occupation ofpatch8. All48 complete before outcome analysis; no automatic followup.'))
    print(json.dumps(dict(root=str(root),held=True,jobs=48,models=12,source_frozen=source is not None)))


if __name__=='__main__':main()
