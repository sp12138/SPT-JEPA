"""Focused CPU contracts for the600 stop-gradient audit and immutable binding."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from experiments.hybrid_carrier_20261031.gate_audit import binding,sha,validate_proof
from experiments.hybrid_carrier_20261031.preflight import query_contract
from experiments.hybrid_carrier_20261031.test_preflight import trained


class GateContracts(unittest.TestCase):
    def test_actual_native_Q_and_logged_gradient_positive_negative(self):
        for arm in ['fixed','adaptive']:
            model,cfg,state,initial=trained(arm)
            query=query_contract(model,cfg,state,initial)
            self.assertTrue(query['passed'])
            current={'checkpoint':'actual-sha'}
            proof=dict(passed=True,binding=current,query_stop_gradient=query)
            trace=[{'grad_norm_pool':0. if arm=='fixed' else 1.}]
            self.assertTrue(validate_proof(proof,current,arm,trace)['passed'])
            with self.assertRaises(AssertionError):validate_proof(proof,{'checkpoint':'stale'},arm,trace)
            with self.assertRaises(AssertionError):
                validate_proof(proof,current,arm,[{'grad_norm_pool':1. if arm=='fixed' else 0.}])
            bad=copy.deepcopy(proof)
            if arm=='fixed':bad['query_stop_gradient']['rows'][0]['optimizer_state_present']=True
            else:bad['query_stop_gradient']['gradient_norm']=0.
            with self.assertRaises(AssertionError):validate_proof(bad,current,arm,trace)

    def test_binding_rejects_mutated_sources_and_detects_changed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            for p in [run/'config.json',run/'initial-model/weights.pt',run/'last.ckpt',run/'model-step0000600/weights.pt']:
                p.parent.mkdir(exist_ok=True);p.write_text('test fixture')
            src=root/'source';src.mkdir();(src/'model.py').write_text('fixed source')
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(source=str(src),source_commit='fixture',files={'model.py':sha(src/'model.py')})))
            descriptor=root/'source-record.json';descriptor.write_text(json.dumps(dict(source=str(src),source_commit='fixture',manifest=str(manifest))))
            before=binding(run,descriptor,descriptor)
            (run/'last.ckpt').write_text('different checkpoint')
            self.assertNotEqual(binding(run,descriptor,descriptor),before)
            (src/'model.py').write_text('changed source')
            with self.assertRaises(AssertionError):binding(run,descriptor,descriptor)


if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    result=unittest.main(exit=False).result
    names=['gate_audit.py','test_gate.py','preflight.py']
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,device='cpu',
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in names},
        scope='Actual native AdamW-ID Q9 positive proof; stale binding, changed source/checkpoint, unexpected fixed state and zero adaptive gradient/log negatives. No model/training changes.')
    out=Path(os.environ.get('HYBRID_GATE_CONTRACT','/tmp/hybrid-gate-contracts.json'))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)
