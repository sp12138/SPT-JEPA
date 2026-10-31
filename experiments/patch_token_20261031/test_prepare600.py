"""CPU config/DAG contract for the single authorized600 gate."""
import copy,json,sys
from pathlib import Path
import prepare600 as p

def main():
    root=p.B/'runs/patch_token600_20261031';source={'source':'/model1018'};helper={'source':'/helper'}
    configs={}
    for a in p.allocations():
        old=p.read(p.OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}/config.json')
        cfg=p.fresh_config(old,a,root/a['id']);reference=p.read(p.PRE/a['id']/'config.json')
        reference.update(output=cfg['output'],checkpoint_every=200,retain_steps=[200,400,600])
        assert cfg==reference and cfg['model']['patch_predictor_seed']==a['training_seed']
        configs[a['id']]=cfg
    jobs=p.make_jobs(root,source,helper,configs);ids={j['id'] for j in jobs}
    assert len(jobs)==len(ids)==40
    assert all(j['source']==source for j in jobs)
    assert all(set(j.get('depends_on',[]))<=ids for j in jobs)
    phases=['train600','fullstate','health','transition','planning']
    assert [j['phase'] for j in jobs]==phases*8
    for j in jobs:
        if j['phase']=='train600':assert j['command'][-2:]==['--stop-step','600']
        if j['phase']=='fullstate':assert '/helper/' in j['command'][3] and j['command'][-2:]==['--expected-step','600']
        if j['phase']=='transition':assert '/model1018/' in j['command'][3] and j['command'][j['command'].index('--cases')+1]=='64'
        assert not any('NGC' in str(x) or 'covariance' in str(x) for x in j['command'])
    old=Path(__file__).with_name('preflight.py').read_text();new=Path(__file__).with_name('preflight600.py').read_text()
    expected=old.replace('default=2);args=p.parse_args()','default=600);args=p.parse_args()')
    expected=expected.replace("assert args.expected_step==2,'This helper authorizes only the two-update technical stage'","assert args.expected_step==600,'This helper is the bounded600 final audit only'")
    for a,b in [("'--expected-step','2'","'--expected-step','600'"),("assert cursor==256","assert cursor==600*cfg['batch_size']"),
                ("status['step']==2","status['step']==600"),("root/'model-step0000002/weights.pt'","root/'model-step0000600/weights.pt'"),
                ('expected_step=2,run=','expected_step=600,run='),
                ("'Two-update technical gate only; no task learning claim, no600 authorization.'","'Bounded600 final audit; no planning superiority or continuation authorization.'")]:expected=expected.replace(a,b)
    assert new==expected,'Only explicit expected-step/cursor/path/scope differ; no check relaxed'
    result=dict(passed=True,endpoints=40,fresh_from_zero=True,configs_match_reviewed_preflight=True,
        model_source_1018_all_jobs=True,native_checks_unchanged_except_step_cursor_paths=True,
        source_sha256={name:p.sha(Path(__file__).with_name(name)) for name in ['prepare600.py','preflight600.py','test_prepare600.py']})
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,endpoints=40)))
if __name__=='__main__':main()
