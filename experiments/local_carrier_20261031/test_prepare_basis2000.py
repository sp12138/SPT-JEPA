"""Two focused contracts: semantic resume and fail-closed native chain."""
import tempfile, unittest
from pathlib import Path
from experiments.local_carrier_20261031 import prepare_basis2000 as p


class BasisMaturityContracts(unittest.TestCase):
    def cfg(self):
        return dict(output='/old',data='/data',seed=1,batch_size=128,max_steps=124910,
            model_class='spatial_carrier',checkpoint_every=200,retain_steps=[200,400,600],
            audit_expectations=dict(initial_unchanged_prefixes=[p.QKEY]),
            model=dict(state_slots=64,state_width=16,spatial_scale=[.2]*1024,
                spatial_mode='local',spatial_basis='orthogonal',sigma0=.1,beta=.25))

    def test_semantics_and_only_four_original_arms(self):
        cfg=self.cfg();new=p.continuation_config(cfg,'/new')
        self.assertEqual(p.semantic(new),p.semantic(cfg))
        self.assertEqual((new['checkpoint_every'],new['retain_steps']),(1200,[1200,2000]))
        self.assertEqual(cfg['retain_steps'],[200,400,600])
        a=list(p.allocations());self.assertEqual(len(a),4)
        self.assertEqual([x['gpu'] for x in a],[2,3,4,5]);self.assertEqual({x['host'] for x in a},{'dev13'})
        self.assertEqual({(x['task'],x['training_seed']) for x in a},{(t,s) for t in ('tworoom','cube') for s in (1,2)})

    def test_archive_dependency_sources_and_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rec=dict(next(p.allocations()),parent='/old',initial_weights='/initial',config=self.cfg(),
                primary_manifest=dict(path='/data/manifests/validation-strict.json',sha256='primary'),
                fresh_manifest=dict(path='/fresh.json',sha256='fresh'))
            (root/rec['id']).mkdir();p.write(root/rec['id']/'config.json',rec['config'])
            training=dict(source='/immutableda',source_commit=p.TRAINING_COMMIT)
            helper=dict(source='/newhelper',source_commit='helper')
            jobs=p.jobs_for(rec,root,helper,training)
            self.assertEqual([j['phase'] for j in jobs],['learn1200','fullstate1200','learn2000','fullstate2000','health','transition','planning','fresh-planning'])
            previous=[]
            for j in jobs:
                self.assertEqual(j['depends_on'],previous);previous=[j['id']]
                self.assertEqual(j['source'],helper if j['phase'].startswith('fullstate') else training)
                if j['phase'].startswith('fullstate'):
                    self.assertEqual(j['command'][3],str(Path(helper['source'])/p.SELF))
                if j['phase'].startswith('learn'):
                    self.assertIn('--resume',j['command'])
            self.assertEqual(jobs[-1]['manifest_sha256'],'fresh');self.assertIn('/fresh.json',jobs[-1]['command'])
            self.assertIn(str(root/rec['id']/'model-step0002000'),jobs[-1]['command'])
            self.assertFalse((root/'plan.json').exists())


if __name__=='__main__':unittest.main()
