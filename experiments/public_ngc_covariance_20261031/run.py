"""Compose frozen f4 action mapping with frozen stable-v2 NGC evaluation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

B=Path('/efs/prescottsun/SPT-JEPA')
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
NGC=B/'runs/public_ngc_stable_source_20261031/source-record-v2.json'
CORR=B/'runs/action_covariance_source_20261031/source-record.json'
FILES=['run.py','prepare.py','contracts.py']


def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def source_check(s):
    for key in ['manifest','runtime_manifest']:
        m=read(s[key])
        for p,h in m['files'].items():assert sha(Path(m['source'])/p)==h,p
def dependencies(ngc,corr):
    path=Path(ngc['source'])/'experiments/public_ngc_20261031';sys.path.insert(0,str(path))
    spec=importlib.util.spec_from_file_location('frozen_f4_adapter',Path(corr['source'])/'experiments/action_covariance_20261031/adapter.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import objective
    from stable_cost import StableNGCCost
    objective.NGCCost=StableNGCCost
    return module,objective


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--inner',action='store_true');a=p.parse_args();s=read(a.spec)
    if not a.inner:
        env=dict(os.environ,PYTHONPATH=s['model_source']['pythonpath'],PYTHONDONTWRITEBYTECODE='1')
        os.chdir(s['model_source']['source']);os.execve(PY,[PY,'-B','-u',str(Path(__file__).resolve()),*sys.argv[1:],'--inner'],env)
    assert s['controller_mode']=='correlated'and s['numeric_variant']=='uniform-fp32-direct-attachment'
    for p,h in s['combined_immutable_inputs'].items():assert sha(p)==h,p
    for f,h in s['combined_helper_files'].items():assert sha(Path(__file__).with_name(f))==h,f
    for source in [s['combined_helper_source'],s['helper_source'],s['correlation_source']]:source_check(source)
    import numpy as np
    calibration=read(s['calibration']);matrix=Path(s['calibration']).with_name('matrix.npz')
    assert calibration['task']==s['task']and calibration['zero_variance_coordinates']==0
    assert calibration['matrix_sha256']==sha(matrix)and calibration['dataset_card_sha256']==sha(Path(s['data'])/'dataset_card.json')
    with np.load(matrix)as f:L=f['cholesky'].copy();R=f['correlation'].copy()
    np.testing.assert_allclose(L@L.T,R,rtol=1e-12,atol=1e-12)
    corr,objective=dependencies(s['helper_source'],s['correlation_source']);native_recorder=objective.Recorder
    with corr.install('correlated',L,expected_scaler=calibration,startup_path=Path(s['output'])/'first-plan.json',expected_model_root=s['model_source']['source'])as mappings:
        class MappedRecorder(native_recorder):
            def record(self,info,native,ngc):
                assert len(mappings)==1 and mappings[0].stats['cost_calls']==len(self.rows)+1
                if not self.rows:
                    actual=corr.array_sha(info['action_candidates'].detach().cpu().numpy())
                    assert actual==mappings[0].stats['first_bank']['normalized_action_sha256']
                return super().record(info,native,ngc)
        objective.Recorder=MappedRecorder
        try:runpy.run_path(str(Path(s['helper_source']['source'])/'experiments/public_ngc_20261031/stable_evaluate.py'),run_name='__main__')
        finally:objective.Recorder=native_recorder
    assert len(mappings)==1
    path=Path(s['output'])/'metrics.json';r=read(path);d=r['ngc_diagnostic'];stats=mappings[0].report()
    assert stats['cost_calls']==d['shared_bank_audit']['calls']and stats['solves']>=1
    # The reference now differs by the planned controller treatment, not replay noise.
    d['white_reference_success_vector_equal']=r['metrics']['episode_successes']==read(s['reference_metrics'])['metrics']['episode_successes']
    d['original_replay_successes_exact']=None
    d['inherited_scope']=d['scope']
    d['scope']='Within the fixed correlated-proposal regime, L2 versus NGC changes only terminal cost. Relative to the designated white-proposal reference, this run additionally applies the frozen f4 correlated proposal and returned-command mapping. Original command execution/scaler/history and CEM budget remain; this is a controller intervention, not a replay or learning change.'
    r['controller_diagnostic']=dict(mode='correlated',calibration=s['calibration'],calibration_sha256=sha(s['calibration']),matrix_sha256=sha(matrix),
        source=s['correlation_source'],combined_source=s['combined_helper_source'],combined_spec_sha256=sha(a.spec),stats=stats,
        mapped_objective_metadata_verified=True,reference_white_metrics=s['reference_metrics'],reference_white_sha256=sha(s['reference_metrics']),
        scope='Exact frozen f4 Cholesky proposal and returned-command mapping combined with exact stable-v2 graph cost; no new graph, fitting, model, scaler, case, marginal-scale or CEM-budget change. Both scores recorded on mapped actions. L2 versus white is an intervention contrast, not an exact-replay gate.')
    for p,h in s['combined_immutable_inputs'].items():assert sha(p)==h,p
    write(path,r);print(json.dumps(dict(id=s['id'],completed=True,mapped_cost_calls=stats['cost_calls'])),flush=True)


if __name__=='__main__':main()
