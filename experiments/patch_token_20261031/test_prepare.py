"""Small CPU-only config/DAG/strict-preflight contract; no outputs activated."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import prepare_preflight as prepare


def main():
    root=prepare.B/'runs/patch_token_preflight_20261031'
    source=dict(source='/frozen/test-source',source_commit='test',manifest='/test/manifest')
    configs={};rows=[]
    for a in prepare.allocations():
        old=prepare.read(prepare.OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}/config.json')
        unchanged=copy.deepcopy(old);cfg=prepare.configure(old,a,root/a['id']);assert old==unchanged
        back=copy.deepcopy(cfg)
        for key in ['output','model_class','checkpoint_every','retain_steps','record_initial_weights','audit_expectations']:
            if key in old:back[key]=copy.deepcopy(old[key])
            else:back.pop(key,None)
        back['model'].pop('patch_basis');back['model'].pop('patch_predictor_seed')
        assert back==old,'Undeclared config difference'
        assert cfg['model']['patch_predictor_seed']==a['training_seed']
        for field,value in [('batch_size',64),('max_steps',600),('freeze_encoder_steps',2)]:
            wrong=copy.deepcopy(old);wrong[field]=value
            try:prepare.configure(wrong,a,root/a['id'])
            except AssertionError:pass
            else:raise AssertionError('Invalid native budget accepted')
        configs[a['id']]=cfg;rows.append(dict(id=a['id'],unchanged_config_fields_exact=True))
    jobs=prepare.jobs(root,source,configs);ids={j['id'] for j in jobs}
    assert len(jobs)==len(ids)==16
    training=[j for j in jobs if j['phase']=='train2'];audits=[j for j in jobs if j['phase']=='fullstate']
    assert len(training)==len(audits)==8
    assert [(j['host'],j['gpu']) for j in training]==[('dev2',0),('dev2',1),('dev2',2),('dev13',0),('dev13',1),('dev13',2),('dev15',0),('dev15',1)]
    for job in jobs:
        assert job['source']==source
        assert set(job.get('depends_on',[]))<=ids
        if job['phase']=='train2':assert job['command'][-2:]==['--stop-step','2'] and '-m' in job['command']
        else:assert len(job['depends_on'])==2 and all(x.startswith('train2-') for x in job['depends_on'])
    text=Path(__file__).with_name('preflight.py').read_text()
    assert 'native.main()' in text and 'native_result[\'passed\']' in text
    assert 'additional_backward_calls=0' in text and 'configure_optimizers()' in text
    assert 'optimizer.load_state_dict' in text and 'same(old_common,new_common)' in text
    result=dict(passed=True,configs=8,endpoints=16,only_native_two_updates=True,old_configs_unchanged=True,
        declared_delta_only=True,allocations_exact=True,dependency_DAG_passed=True,
        fullstate='Original strict preflight called without changing checks; adds per-layer gradients and CPU native restore, never a replacement Trainer.',
        source_sha256={name:prepare.sha(Path(__file__).with_name(name)) for name in ['prepare_preflight.py','preflight.py','test_prepare.py']},rows=rows)
    output=Path(sys.argv[1]);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,configs=8,endpoints=16)))


if __name__=='__main__':main()
