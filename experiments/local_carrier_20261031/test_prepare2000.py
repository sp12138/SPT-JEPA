"""Fast contracts for bounded immutable continuation preparation; no GPU."""
import copy,importlib.util,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('spatial_maturity_prepare',Path(__file__).with_name('prepare2000.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def cfg(seed=1):
    return dict(output='/old',data='/readonly/data',seed=seed,batch_size=128,workers=4,max_steps=124910,
        model_class='spatial_carrier',checkpoint_every=200,retain_steps=[200,400,600],
        model=dict(state_slots=64,state_width=16,spatial_scale=[.2]*1024,spatial_mean=[0.]*1024,
                   sigma0=.1,beta=.25,rho=.8,forecast_fraction=.5))


class MaturityContracts(unittest.TestCase):
    def test_allocation_and_semantics(self):
        rows=list(p.allocations());self.assertEqual(len(rows),8)
        self.assertEqual([x['gpu'] for x in rows],list(range(8)));self.assertEqual({x['host'] for x in rows},{'dev15'})
        before=cfg();new=p.continuation_config(before,'/new');self.assertEqual(p.semantic(before),p.semantic(new))
        self.assertEqual(before['retain_steps'],[200,400,600]);self.assertEqual(new['retain_steps'],[1200,2000])
        self.assertEqual(new['checkpoint_every'],1200)
        bad=cfg();bad['model']['spatial_scale'][0]=.3
        with self.assertRaises(AssertionError):p.continuation_config(bad,'/new')

    def test_two_native_segments_are_blocked_by_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rec=dict(next(p.allocations()),parent='/old',initial_weights='/old/initial',config=cfg())
            out=root/rec['id'];out.mkdir();p.write(out/'config.json',rec['config'])
            old=dict(source='/frozen32',source_commit=p.TRAINING_COMMIT);helper=dict(source='/newhelper',source_commit='helper')
            jobs=p.jobs_for(rec,root,helper,old);self.assertEqual(len(jobs),8)
            self.assertEqual(jobs[2]['depends_on'],[jobs[1]['id']]);self.assertEqual(jobs[4]['depends_on'],[jobs[3]['id']])
            for job in jobs:
                self.assertEqual(job['source'],helper if job['phase'].startswith('fullstate') else old)
                if job['phase'].startswith('learn'):
                    command=job['command'];self.assertIn('--resume',command)
                    self.assertIn(int(command[command.index('--stop-step')+1]),(1200,2000))
            self.assertFalse((root/'plan.json').exists())

    def test_preparation_writes_only_candidate_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'old';parent.mkdir();target=root/'new';helperfile=root/'helper.json'
            training=dict(source='/frozen32',source_commit=p.TRAINING_COMMIT);helper=dict(source='/helper',source_commit='helper')
            p.write(parent/'source-record.json',training);p.write(helperfile,helper);(parent/'HOLD_DISPATCH').write_text('retain')
            (parent/'exits').mkdir();jobs=[]
            cal=root/'calibration.json';p.write(cal,dict(normalization='global_rms'))
            for rec in p.allocations():
                run=parent/rec['id'];run.mkdir();c=cfg(rec['training_seed']);c['spatial_calibration']=dict(path=str(cal),sha256=p.digest(cal))
                p.write(run/'config.json',c);p.write(run/'stage-complete.json',dict(stage_complete=True,stop_step=600))
                (run/'last.ckpt').write_bytes(('fullstate-'+rec['id']).encode())
                for prefix in ('','health-','info-','transition-','planning-'):
                    name=prefix+rec['id'];jobs.append(dict(id=name,config_sha256=p.digest(run/'config.json')))
                    p.write(parent/'exits'/(name+'.json'),dict(returncode=0))
            p.write(parent/'plan.json',dict(jobs=jobs,environment={}))
            def fake_audit(run,*args):
                return dict(passed=True,checkpoint_sha256=p.digest(Path(run)/'last.ckpt'),rng_component_sha256=dict(torch='exact'),sampler=dict(passed=True,next_indices=[1,2]))
            args=SimpleNamespace(root=str(target),parent=str(parent),source_record=str(helperfile),dry_run=False)
            with patch.object(p,'source_checks',return_value={}),patch.object(p,'audit_state',side_effect=fake_audit) as audit:
                p.prepare(args);self.assertEqual(audit.call_count,16)
            self.assertEqual(len(p.read(target/'candidate-plan.json')['jobs']),64);self.assertFalse((target/'plan.json').exists())
            self.assertEqual((parent/'HOLD_DISPATCH').read_text(),'retain')
            for rec in p.allocations():self.assertEqual((parent/rec['id']/'last.ckpt').read_bytes(),(target/rec['id']/'last.ckpt').read_bytes())
            with self.assertRaises(AssertionError):p.prepare(args)


if __name__=='__main__':unittest.main()
