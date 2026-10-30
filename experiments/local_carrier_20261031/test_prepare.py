"""Fast stdlib-only contracts for the three-stage preparation boundary."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('spatial_prepare',Path(__file__).with_name('prepare.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PrepareContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        for name in ('CAL','PRE','ROOT','OLD'):
            replacement=self.root/name.lower();mock=patch.object(p,name,replacement);mock.start();self.addCleanup(mock.stop)
        p.CAL.mkdir();self.src=dict(source='/immutable/source',source_commit='fixed')
        self.cfg=dict(output='/old/run',data='/readonly/data',dataset_kind='benchmark_hdf5',seed=1,
            batch_size=128,workers=4,max_steps=124910,robust_variant=True,
            model=dict(mode='L2',window=8,sigma0=.1,beta=.25,rho=.8,forecast_fraction=.5,action_dim=10))
        for task in ('tworoom','cube'):
            for seed in (1,2):
                directory=p.CAL/f'calibrate-{task}-seed{seed}';directory.mkdir()
                cfg=copy.deepcopy(self.cfg);cfg['seed']=seed;reference=directory/'reference.json';p.write(reference,cfg)
                arms={a:dict(model_kwargs=dict(spatial_mode=a,state_slots=64,state_width=16,
                    spatial_mean=[0.]*1024,spatial_scale=[1.]*1024)) for a in ('local','dense')}
                p.write(directory/'calibration.json',dict(config=str(reference),config_sha256=p.digest(reference),
                    arms=arms,measured_frames=128,model_seed=seed,source_commit='fixed'))

    def test_exact_allocation_and_preserved_training_semantics(self):
        allocation=list(p.assignments());self.assertEqual(len(allocation),8)
        self.assertEqual([a['gpu'] for a in allocation],list(range(8)))
        self.assertEqual({a['host'] for a in allocation},{'dev15'})
        for a in allocation:
            cfg=p.config(a,self.root/a['id'],600)
            for key in ('batch_size','workers','max_steps','dataset_kind','robust_variant'):
                self.assertEqual(cfg[key],self.cfg[key])
            expected=copy.deepcopy(self.cfg['model']);expected.update(cfg['model'])
            for key in self.cfg['model']:self.assertEqual(cfg['model'][key],self.cfg['model'][key])
            self.assertEqual(cfg['model_class'],'spatial_carrier');self.assertEqual(cfg['retain_steps'],[200,400,600])
            self.assertNotIn('initial_weights',cfg)
        # Altered reference bytes must be rejected, not silently accepted.
        reference=p.CAL/'calibrate-tworoom-seed1/reference.json';reference.write_text(reference.read_text()+' ')
        with self.assertRaises(AssertionError):p.config(allocation[0],self.root/'bad',600)

    def test_preflight_requires_calibration_success(self):
        p.write(p.CAL/'source-record.json',self.src)
        jobs=[dict(id='calibrate-tworoom-seed1',output=str(p.CAL/'calibrate-tworoom-seed1'),training_seed=1)]*4
        p.write(p.CAL/'plan.json',dict(jobs=jobs,environment={}))
        exits=p.CAL/'exits';exits.mkdir();p.write(exits/'calibrate-tworoom-seed1.json',dict(returncode=1))
        with self.assertRaises(AssertionError):p.calibration_ready(self.src)

    def test_paired_preflight_and_reviewed_fresh600(self):
        args=type('Args',(),dict(source_record='unused'))()
        with patch.object(p,'source',return_value=self.src),patch.object(p,'calibration_ready',return_value={}):
            p.preflight(args)
            plan=p.read(p.PRE/'plan.json');self.assertEqual(len(plan['jobs']),16)
            for job in plan['jobs']:
                if job['phase']=='train2':
                    cfg=p.read(job['config']);self.assertEqual(cfg['batch_size'],128);self.assertEqual(cfg['retain_steps'],[2])
                elif job['arm']=='dense':self.assertEqual(len(job['depends_on']),2)
            with self.assertRaises(FileNotFoundError):p.training(args)
            self.assertFalse(p.ROOT.exists())
            p.write(p.PRE/'review.json',dict(passed=True));(p.PRE/'exits').mkdir()
            for job in plan['jobs']:
                p.write(p.PRE/'exits'/(job['id']+'.json'),dict(returncode=0))
                if job['phase']=='gradient':
                    out=Path(job['output']);out.mkdir();p.write(out/'preflight.json',dict(passed=True))
            p.training(args)
            jobs=p.read(p.ROOT/'plan.json')['jobs'];self.assertEqual(len(jobs),40)
            trains=[j for j in jobs if j['phase']=='train600'];self.assertEqual(len(trains),8)
            for j in trains:
                self.assertEqual(j['command'][-2:],['--stop-step','600'])
                self.assertFalse((Path(j['output'])/'last.ckpt').exists())
            with self.assertRaises(AssertionError):p.training(args)


if __name__=='__main__':unittest.main()
