"""CPU-only config/DAG and strict-auditor step adaptation contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile

import prepare2000 as p


def main(output):
    source={'source':'/model1018'};helper={'source':'/newhelper'};original=[];jobs=[]
    allocations=p.allocations()
    assert [(a['host'],a['gpu'])for a in allocations]==[('dev2',0),('dev2',1),('dev2',2),('dev13',0),('dev13',1),('dev13',2),('dev15',0),('dev15',1)]
    with tempfile.TemporaryDirectory(prefix='patch2000-contract-')as temp:
        root=Path(temp)
        for a in allocations:
            parent=p.PARENT/a['id'];old=p.read(parent/'config.json');before=copy.deepcopy(old)
            cfg=p.config(old,root/a['id']);assert old==before and p.semantic(cfg)==p.semantic(old)
            assert cfg['retain_steps']==[1200,2000] and cfg['checkpoint_every']==1200
            p.write(root/a['id']/'config.json',cfg)
            references=p.existing_references(a['task'],a['training_seed'],cfg['data'])
            rec=dict(**a,parent=str(parent),initial=str(parent/'initial-model/weights.pt'),
                reference_initial='/original/initial.pt',config=cfg,references=references)
            jj=p.jobs_for(rec,root,helper,source);jobs.extend(jj)
            assert [j['phase']for j in jj]==['learn1200','archive1200','learn2000','archive2000','fullstate2000','health','transition','planning','fresh-planning']
            assert jj[0]['depends_on']==[]
            assert all(jj[i]['depends_on']==[jj[i-1]['id']]for i in range(1,9))
            for i,step in [(0,'1200'),(2,'2000')]:
                c=jj[i]['command'];assert '--resume'in c and c[c.index('--stop-step')+1]==step
                assert c[3:5]==['-m','as_jepa_tools.train_stage']
            c=jj[4]['command'];assert c[c.index('--expected-step')+1]=='2000'and '--paired-initial'in c
            assert jj[5]['command'][3:5]==['-m','as_jepa_tools.benchmark_health']and jj[5]['command'][-2:]==['--cases','16']
            assert jj[6]['command'][jj[6]['command'].index('--cases')+1]=='64'
            assert all(j['source']==source for j in jj)
            for j,key in [(jj[7],'primary'),(jj[8],'fresh')]:
                c=j['command'];assert c[c.index('--manifest')+1]==references[key+'_manifest']
                assert j['manifest_sha256']==references[key+'_manifest_sha256']
                assert c[c.index('--checkpoint')+1].endswith('/model-step0002000')
            original.append(dict(id=a['id'],parent_config_sha256=p.sha(parent/'config.json'),references=references))
    assert len(jobs)==len({j['id']for j in jobs})==72
    assert all(set(j['depends_on'])<={j['id']for j in jobs}for j in jobs)
    previous=Path(__file__).with_name('preflight600.py').read_text()
    for a,b in [('default=600','default=2000'),('args.expected_step==600','args.expected_step==2000'),('bounded600','bounded2000'),
                ("'--expected-step','600'","'--expected-step','2000'"),('cursor==600*','cursor==2000*'),
                ("status['step']==600","status['step']==2000"),('model-step0000600','model-step0002000'),('Bounded600','Bounded2000')]:previous=previous.replace(a,b)
    assert Path(__file__).with_name('preflight2000.py').read_text()==previous
    files=['prepare2000.py','resume_audit.py','preflight2000.py','test_prepare2000.py']
    result=dict(passed=True,endpoints=72,runs=8,only_original_tasks_seeds_bases=True,
        original_reserved_slots=True,all_jobs_model1018=True,archive1200_before2000=True,
        original_config_semantics_exact=True,native_final_auditor_only_step_paths_scope_changed=True,
        existing_reference_manifests_frozen=original,
        native_actual_restore_proof='Required separately on all8 real600 parents and all8 copied states before activation; this DAG test does not substitute for those checks.',
        source_sha256={'experiments/patch_token_20261031/'+name:p.sha(Path(__file__).with_name(name))for name in files})
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,endpoints=72)))


if __name__=='__main__':main(sys.argv[1])
