"""CPU configuration/dependency checks; no model, dataset or GPU."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.dynamic_readout_20261031 import prepare as p
class Contracts(unittest.TestCase):
    def cfg(self):
        return dict(output='/old',data='/data',model_class='spatial_carrier',seed=1,batch_size=128,max_steps=124910,
            record_initial_weights=True,checkpoint_every=200,retain_steps=[200,400,600],
            audit_expectations=dict(gradients=dict(action_encoder='active'),initial_unchanged_prefixes=['spatial_projection','spatial_mean','spatial_scale']),
            model=dict(spatial_mode='local',state_slots=64,state_width=16,beta=.25,sigma0=.1,rho=.8,forecast_fraction=.5))
    def test_exact_allocations_and_semantics(self):
        allocations=list(p.allocations());self.assertEqual(len(allocations),8)
        self.assertEqual([a['gpu']for a in allocations],list(range(8)));self.assertEqual({a['host']for a in allocations},{'dev15'})
        self.assertEqual({(a['task'],a['training_seed'],a['arm'])for a in allocations},
            {(t,s,a)for t in ['tworoom','cube']for s in [1,2]for a in ['static','state']})
        for step in [3,600]:
            old=self.cfg();original=copy.deepcopy(old);cfg=p.update_config(old,allocations[0],'/new',step)
            self.assertEqual(old,original);self.assertEqual(cfg['max_steps'],old['max_steps']);self.assertEqual(cfg['batch_size'],128)
            changed={k for k in cfg if cfg[k]!=old[k]};self.assertEqual(changed,{'output','checkpoint_every','retain_steps','model','audit_expectations'}if step==3 else{'output','model','audit_expectations'})
            model=dict(cfg['model']);self.assertEqual(model.pop('spatial_readout_mode'),'static');self.assertEqual(model,old['model'])
    def test_fail_closed_chains_no_fresh_or_resume(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);a=next(p.allocations());out=root/a['id'];out.mkdir();cfg=p.update_config(self.cfg(),a,out,3);p.write(out/'config.json',cfg)
            source=dict(source='/frozen',source_commit='pinned')
            with patch.object(p,'sha',return_value='pinned'):
                for step,phases in [(3,['train3','fullstate']),(600,['train600','fullstate','health','transition','planning'])]:
                    jobs=p.jobs_for(a,cfg,Path('/original'),root,source,step);self.assertEqual([x['phase']for x in jobs],phases);prev=[]
                    for j in jobs:
                        self.assertEqual(j['depends_on'],prev);self.assertEqual(j['source'],source);self.assertNotIn('--resume',j['command']);prev=[j['id']]
                    self.assertIn('--expected-step',jobs[1]['command']);self.assertIn(str(step),jobs[1]['command'])
            self.assertFalse((root/'plan.json').exists())
if __name__=='__main__':unittest.main()
