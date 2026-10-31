import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.dynamic_readout_20261031 import prepare2000 as p


class ContinuationContracts(unittest.TestCase):
    def test_only_operational_configuration_changes(self):
        old=dict(output='old',checkpoint_every=200,retain_steps=[200,400,600],batch_size=128,max_steps=124910,
            seed=2,model_class='spatial_carrier',model={'spatial_mode':'local','spatial_readout_mode':'state','rho':.8,'beta':.25},
            audit_expectations={'gradients':{'dynamic_readout_module.input.weight':'active'}})
        original=copy.deepcopy(old);new=p.config(old,'new')
        self.assertEqual(old,original);self.assertEqual(p.semantic(old),p.semantic(new))
        self.assertEqual(new['retain_steps'],[1200,2000]);self.assertEqual(new['checkpoint_every'],1200)

    def test_exact_eight_allocations_and_nine_stage_dependencies(self):
        records=list(p.allocations());self.assertEqual([r['gpu']for r in records],list(range(8)))
        self.assertEqual({r['host']for r in records},{'dev15'})
        self.assertEqual({(r['task'],r['training_seed'],r['arm'])for r in records},
                         {(t,s,a)for t in['tworoom','cube']for s in[1,2]for a in['static','state']})
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);r=records[0];run=root/r['id'];run.mkdir();(run/'config.json').write_text('{}')
            r.update(parent='immutable-parent',initial='immutable-initial',reference_initial='old-original-initial',
                config={'data':'original-data'},references={'primary_manifest':'old-primary50','primary_manifest_sha256':'a',
                    'fresh_manifest':'old-fresh50','fresh_manifest_sha256':'b'})
            model={'source':'immutable-d6'};helper={'source':'frozen-helper','manifest':'helper-manifest'}
            jobs=p.jobs_for(r,root,helper,model)
            self.assertEqual([j['phase']for j in jobs],['learn1200','archive1200','learn2000','archive2000','fullstate2000','health','transition','planning','fresh-planning'])
            for i,j in enumerate(jobs):
                self.assertEqual(j['source'],model);self.assertEqual(j['depends_on'],[]if i==0 else[jobs[i-1]['id']])
            for i,step in[(0,'1200'),(2,'2000')]:
                self.assertIn('--resume',jobs[i]['command']);self.assertEqual(jobs[i]['command'][jobs[i]['command'].index('--stop-step')+1],step)
            health=jobs[5]['command'];self.assertEqual(health[health.index('--expected-step')+1],'2000')
            self.assertIn('old-primary50',jobs[-2]['command']);self.assertIn('old-fresh50',jobs[-1]['command'])


if __name__=='__main__':unittest.main()
