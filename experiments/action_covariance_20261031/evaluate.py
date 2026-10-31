"""Original model-specific evaluator under a fixed action-correlation hook."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from adapter import install

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);args=ap.parse_args();spec=read(args.spec)
    out=Path(spec['output']);assert not(out/'metrics.json').exists()
    source=spec['source'];sys.path[:0]=source['pythonpath'].split(':')
    for path,h in spec['immutable_inputs'].items():assert sha(path)==h,path
    for mk,sk in [('manifest','source'),('runtime_manifest','runtime_source')]:
        for path,h in read(source[mk])['files'].items():assert sha(Path(source[sk])/path)==h,path
    for name,h in spec['helper_files'].items():assert sha(Path(__file__).with_name(name))==h,name
    calibration=read(spec['calibration']);matrix=Path(spec['calibration']).with_name('matrix.npz')
    assert sha(matrix)==calibration['matrix_sha256']and calibration['zero_variance_coordinates']==0
    assert calibration['task']==spec['task']and calibration['dataset_card_sha256']==sha(Path(spec['data'])/'dataset_card.json')
    with np.load(matrix)as f:L=f['cholesky'].copy();R=f['correlation'].copy()
    assert np.isfinite(R).all()and np.isfinite(L).all();np.testing.assert_allclose(L@L.T,R,rtol=1e-12,atol=1e-12)
    reference=read(spec['reference_metrics'])
    from as_jepa_tools import benchmark_evaluate
    assert Path(benchmark_evaluate.__file__).resolve()==Path(spec['native_evaluator']).resolve()
    argv=[benchmark_evaluate.__file__]
    for key in ['manifest','checkpoint','data','output']:argv+=['--'+key,spec[key]]
    saved=sys.argv
    try:
        sys.argv=argv
        with install(spec['mode'],L,expected_scaler=calibration,startup_path=out/'first-plan.json',expected_model_root=source['source'])as records:benchmark_evaluate.main()
    finally:sys.argv=saved
    assert len(records)==1
    report=read(out/'metrics.json')
    for key in ['pairs','manifest_sha256','checkpoint','solver','plan','normalization','split','protocol','checkpoint_audit']:
        assert report[key]==reference[key],key
    assert len(report['pairs'])==50 and report['normalization']=='train episodes'
    for name in ['weights.pt','config.json']:
        path=str(Path(spec['checkpoint'])/name);assert sha(path)==spec['immutable_inputs'][path]
    replay=None if spec['mode']!='original'else report['metrics']['episode_successes']==reference['metrics']['episode_successes']
    report['controller_diagnostic']=dict(mode=spec['mode'],calibration=spec['calibration'],calibration_sha256=sha(spec['calibration']),
        matrix_sha256=sha(matrix),source=source,helper_files=spec['helper_files'],spec_sha256=sha(args.spec),stats=records[0].report(),
        original_replay_successes_exact=replay,checkpoint_files_unchanged=True,matched_cases_checkpoint_solver_plan_normalization=True,
        reference_metrics=spec['reference_metrics'],reference_sha256=sha(spec['reference_metrics']),
        scope='Fixed training-only Cholesky coordinates around the native CEM. Model/objective and executed normalized commands both receive x=L y; original diagonal elite moments remain in white y. Original loss, weights, action history, scaler, goal metric,300x30 budget,5-step horizon and unbounded commands remain. Native original/L=I bypasses all mapping. Historical Cube outcomes are not bitwise reproducible; any replay differences are retained, not relabelled or selectively replaced. Controller diagnostic, not a new learning algorithm.')
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(mode=spec['mode'],success_rate=report['metrics']['success_rate'],replay_equal=replay)),flush=True)
if __name__=='__main__':main()
