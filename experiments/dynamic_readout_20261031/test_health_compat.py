import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.dynamic_readout_20261031 import health_compat as h


class EvidenceContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.run = self.base / 'training'
        self.audit = self.base / 'recovery/fullstate-training'
        self.audit.mkdir(parents=True)
        self.source = {'source_commit': h.MODEL_COMMIT, 'source': 'pinned-model-source'}
        self.write(self.run / 'config.json', {'model': {'spatial_readout_mode': 'state'}})
        for name in ['last.ckpt', 'model-step0000600/weights.pt', 'initial-model/weights.pt']:
            path = self.run / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(name.encode())
        self.reference = self.base / 'reference.pt'; self.reference.write_bytes(b'original-reference')
        keys = {h.PREFIX+k: v for k, v in h.SHAPES.items()}
        self.native = dict(run=str(self.run), passed=True, checkpoint={'step':600, 'checks': {'complete':True}},
            all_parameter_gradients_finite=True, loss_finite=True, config_sha256=h.sha(self.run/'config.json'),
            checkpoint_sha256=h.sha(self.run/'last.ckpt'), weights_sha256=h.sha(self.run/'model-step0000600/weights.pt'),
            initial_weights={'sha256': h.sha(self.run/'initial-model/weights.pt')},
            gradients={key:dict(passed=True,gradients_finite=True)for key in keys})
        self.detail = dict(passed=True,step=600,original_initial_tensors_exact=553,frozen_tensors_unchanged=210,
            new_tensors=4,new_parameters=5376,mode='state',new_parameter_shapes=keys,
            checkpoint_sha256=self.native['checkpoint_sha256'],weights_sha256=self.native['weights_sha256'],
            initial_sha256=self.native['initial_weights']['sha256'],reference_initial_sha256=h.sha(self.reference),
            head_gradient={'passed':True,'per_tensor':{k:{'finite':True,'max_abs':.1}for k in h.SHAPES}})
        self.write(self.audit/'preflight.json',self.native);self.write(self.audit/'readout-audit.json',self.detail)
        self.write(self.audit/'dispatch.json',dict(id=self.audit.name,source=self.source,
            command=['native','--run',str(self.run),'--expected-step','600','--reference-initial',str(self.reference)]))
        self.write(self.audit.parent/'exits'/(self.audit.name+'.json'),dict(id=self.audit.name,returncode=0))

    def write(self,path,obj):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))

    def verify(self):
        return h.verify_evidence(self.run,self.audit,self.source)

    def test_valid_evidence_preserves_all_module_rules(self):
        evidence=self.verify();named=[(k,SimpleNamespace(shape=v['shape'],requires_grad=True))for k,v in evidence['parameters'].items()]
        model=SimpleNamespace(named_parameters=lambda:iter(named))
        original={'encoder':'frozen','pool':'frozen','predictor':'active','renderer':'active','action_encoder':'active','dynamic_readout_module':'active'}
        full={**original,**{k:'active'for k,_ in named}};before=full.copy()
        self.assertEqual(h.verified_module_expectations(model,full,evidence),original)
        self.assertEqual(full,before)

    def test_missing_evidence_fails(self):
        (self.audit/'readout-audit.json').unlink()
        with self.assertRaises(FileNotFoundError):self.verify()

    def test_config_weights_checkpoint_initial_reference_hash_mismatch_fails(self):
        for path in [self.run/'config.json',self.run/'last.ckpt',self.run/'model-step0000600/weights.pt',self.run/'initial-model/weights.pt',self.reference]:
            old=path.read_bytes();path.write_bytes(old+b' ')
            with self.subTest(path=str(path)):
                with self.assertRaises(AssertionError):self.verify()
            path.write_bytes(old)

    def test_zero_nan_and_infinite_per_tensor_gradients_fail(self):
        for value in [0.,float('nan'),float('inf')]:
            altered=copy.deepcopy(self.detail);altered['head_gradient']['per_tensor']['input.weight']['max_abs']=value
            self.write(self.audit/'readout-audit.json',altered)
            with self.assertRaises(AssertionError):self.verify()

    def test_source_and_exit_failures_fail_closed(self):
        source={**self.source,'source_commit':'wrong'}
        with self.assertRaises(AssertionError):h.verify_evidence(self.run,self.audit,source)
        self.write(self.audit.parent/'exits'/(self.audit.name+'.json'),dict(id=self.audit.name,returncode=1))
        with self.assertRaises(AssertionError):self.verify()

    def test_actual_exact_parameter_missing_or_wrong_shape_fails(self):
        evidence=self.verify();full={'dynamic_readout_module':'active',**{k:'active'for k in evidence['parameters']}}
        for names in [[],[(k,SimpleNamespace(shape=[1],requires_grad=True))for k in evidence['parameters']]]:
            model=SimpleNamespace(named_parameters=lambda:iter(names))
            with self.assertRaises(AssertionError):h.verified_module_expectations(model,full,evidence)

    def test_explicit2000_requires_matching_step_path_and_dispatch(self):
        with self.assertRaises(AssertionError):h.verify_evidence(self.run,self.audit,self.source,2000)
        self.native['checkpoint']['step']=2000;self.detail['step']=2000
        old=self.run/'model-step0000600/weights.pt';new=self.run/'model-step0002000/weights.pt'
        new.parent.mkdir();new.write_bytes(old.read_bytes())
        self.write(self.audit/'preflight.json',self.native);self.write(self.audit/'readout-audit.json',self.detail)
        d=json.loads((self.audit/'dispatch.json').read_text());d['command'][d['command'].index('--expected-step')+1]='2000'
        self.write(self.audit/'dispatch.json',d)
        self.assertEqual(len(h.verify_evidence(self.run,self.audit,self.source,2000)['parameters']),4)
        with self.assertRaises(AssertionError):self.verify()


if __name__=='__main__':unittest.main()
