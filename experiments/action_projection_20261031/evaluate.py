"""Run the ORIGINAL model-specific evaluator under a bounded-controller hook."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from adapter import install

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text());out=Path(spec['output'])
    assert not (out/'metrics.json').exists(), 'Refuse to overwrite an evaluation'
    source=spec['source'];sys.path[:0]=source['pythonpath'].split(':')
    for path,digest in spec['immutable_inputs'].items():assert sha(path)==digest,path
    for manifest_key,source_key in [('manifest','source'),('runtime_manifest','runtime_source')]:
        manifest=json.loads(Path(source[manifest_key]).read_text())
        for path,digest in manifest['files'].items():assert sha(Path(source[source_key])/path)==digest,path
    for name,digest in spec['helper_files'].items():assert sha(Path(__file__).with_name(name))==digest,name
    reference=json.loads(Path(spec['reference_metrics']).read_text())
    from as_jepa_tools import benchmark_evaluate
    assert Path(benchmark_evaluate.__file__).resolve()==Path(spec['native_evaluator']).resolve()
    argv=[benchmark_evaluate.__file__]
    for k in ['manifest','checkpoint','data','output']:argv+=['--'+k,spec[k]]
    saved=sys.argv
    try:
        sys.argv=argv
        with install(spec['mode']) as projections:
            benchmark_evaluate.main()
    finally:sys.argv=saved
    assert len(projections)==1
    projection=projections[0];card=json.loads((Path(spec['data'])/'dataset_card.json').read_text())
    np.testing.assert_array_equal(projection.scaler.mean_,card['action_mean'])
    np.testing.assert_array_equal(projection.scaler.scale_,card['action_std'])
    report=json.loads((out/'metrics.json').read_text())
    for k in ['pairs','manifest_sha256','checkpoint','solver','plan','normalization','split','protocol']:
        assert report[k]==reference[k],k
    assert report['checkpoint_audit']==reference['checkpoint_audit']
    assert len(report['pairs'])==50 and report['normalization']=='train episodes'
    if spec['mode']=='bounded':
        assert projection.stats['score_calls']>0 and projection.stats['execution_calls']>0
        assert projection.stats['returned_command_violations']==0
    replay_equal=None
    if spec['mode']=='original':
        replay_equal=report['metrics']['episode_successes']==reference['metrics']['episode_successes']
    # Recheck the checkpoint files; no modified copy/calibration is produced.
    for name in ['weights.pt','config.json']:
        path=str(Path(spec['checkpoint'])/name);assert sha(path)==spec['immutable_inputs'][path]
    report['controller_diagnostic']=dict(mode=spec['mode'],source=source,stats=projection.report(),
        input_contract=dict(spec_sha256=sha(a.spec),reference_metrics=spec['reference_metrics'],reference_sha256=spec['immutable_inputs'][spec['reference_metrics']],
                            matched_cases_checkpoint_solver_plan_normalization=True,checkpoint_files_unchanged=True),
        helper_files={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n))for n in ['adapter.py','evaluate.py']},
        original_replay_successes_exact=replay_equal,
        scope='Uniform Box projection before model candidate rollout AND before raw command execution. Original CEM proposal/moment/warm-start algorithms unchanged; future proposals change through the treated costs. No training or recalibration. TwoRoom physical clipping is equivalent, but bounded returned commands also enter subsequent logged action history. Cube has no original input clipping, so this is an explicit bounded-controller intervention, not an actuator-equivalent rewrite or new learning algorithm.')
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if replay_equal is False:raise AssertionError('Original-mode P0 replay differs; preserve evidence and inspect before interpreting bounded comparisons')
    print(json.dumps(dict(mode=spec['mode'],success_rate=report['metrics']['success_rate'],stats=projection.report())),flush=True)

if __name__=='__main__':main()
