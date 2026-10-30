"""Focused stdlib contracts for the bounded followup; no forward or GPU."""
import importlib.util,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('spatial6000',Path(__file__).with_name('prepare6000.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class FollowupContracts(unittest.TestCase):
    def cfg(self):
        return dict(output='/old',data='/data',seed=1,batch_size=128,max_steps=124910,model_class='spatial_carrier',
            checkpoint_every=1200,retain_steps=[1200,2000],model=dict(state_slots=64,state_width=16,spatial_scale=[.2]*1024,
                spatial_mean=[0.]*1024,rho=.8,forecast_fraction=.5,beta=.25,sigma0=.1))

    def test_config_semantics_and_original_all8(self):
        cfg=self.cfg();new=p.continuation_config(cfg,'/new')
        self.assertEqual(p.semantic(new),p.semantic(cfg));self.assertEqual(new['checkpoint_every'],2000)
        self.assertEqual(new['retain_steps'],[4000,6000]);self.assertEqual(cfg['retain_steps'],[1200,2000])
        a=list(p.core.allocations());self.assertEqual(len(a),8);self.assertEqual([x['gpu'] for x in a],list(range(8)))
        self.assertEqual({x['host'] for x in a},{'dev15'})

    def test_archive_dependency_and_mandatory_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rec=dict(next(p.core.allocations()),config=self.cfg(),parent='/old',initial_weights='/initial',
                primary_manifest=dict(path='/primary.json',sha256='p'),fresh_manifest=dict(path='/fresh.json',sha256='f'))
            out=root/rec['id'];out.mkdir();p.write(out/'config.json',rec['config'])
            training=dict(source='/immutable32',source_commit=p.core.TRAINING_COMMIT);helper=dict(source='/helper',source_commit='new')
            jobs=p.jobs_for(rec,root,helper,training);self.assertEqual(len(jobs),8)
            self.assertEqual([j['phase'] for j in jobs],['learn4000','fullstate4000','learn6000','fullstate6000','health','transition','planning','fresh-planning'])
            self.assertEqual(jobs[2]['depends_on'],[jobs[1]['id']]);self.assertEqual(jobs[4]['depends_on'],[jobs[3]['id']])
            self.assertEqual(jobs[-1]['depends_on'],[jobs[-2]['id']])
            for j in jobs:
                self.assertEqual(j['source'],helper if j['phase'].startswith('fullstate') else training)
                if j['phase'].startswith('learn'):
                    c=j['command'];self.assertIn('--resume',c);self.assertIn(int(c[c.index('--stop-step')+1]),[4000,6000])
            self.assertEqual(jobs[-1]['manifest_sha256'],'f');self.assertIn('/fresh.json',jobs[-1]['command'])
            self.assertFalse((root/'plan.json').exists())

    def test_failed_fresh_confirmation_blocks_prepare(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'exits').mkdir();jobs=[dict(id=f'job{i}') for i in range(12)]
            p.write(root/'plan.json',dict(jobs=jobs))
            for j in jobs:p.write(root/'exits'/(j['id']+'.json'),dict(returncode=1 if j['id']=='job7' else 0))
            with self.assertRaises(AssertionError):p.fresh_references(root,'/old')


if __name__=='__main__':unittest.main()
