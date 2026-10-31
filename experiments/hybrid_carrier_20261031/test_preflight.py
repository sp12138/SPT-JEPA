"""CPU negative contracts for Q9 audits and read-only clipping observations."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from vforge.models.as_jepa.pooling import QueryPool
from as_jepa_lewm.train import TrainModule
from experiments.hybrid_carrier_20261031.preflight import query_contract, query_numerics
from experiments.hybrid_carrier_20261031.train_preflight import observe_clip, branch_statistics

EVIDENCE = {}


class Fixture(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.hybrid_query=arm
        self.pool=QueryPool(192,3,4,16);self.pool.initialize()
        self.encoder=nn.Linear(2,2).requires_grad_(False)
        self.predictor=nn.Linear(2,2);self.renderer=nn.Linear(2,2)
        self.action_encoder=nn.Linear(2,2)


def trained(arm):
    torch.manual_seed(52);model=Fixture(arm)
    cfg=dict(model_class='hybrid_carrier',max_steps=124910)
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    configured=TrainModule.configure_optimizers(SimpleNamespace(model=model,cfg=cfg))
    optimizer=configured['optimizer']
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=sum(p.square().sum() for n,p in model.named_parameters()
                 if p.requires_grad and not (arm=='fixed' and n.startswith('pool.')))
        loss.backward();optimizer.step()
    checkpoint=dict(global_step=2,optimizer_states=[copy.deepcopy(optimizer.state_dict())])
    return model,cfg,checkpoint,initial


class AuditContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_fixed_Q_requires_true_but_grad_state_absent_and_negative_cases(self):
        m,c,s,i=trained('fixed')
        r=query_contract(m,c,s,i);self.assertTrue(r['passed'])
        self.assertEqual(len(r['rows']),9)
        self.assertTrue(all(p.requires_grad for p in m.pool.parameters()))
        pid=r['rows'][0]['optimizer_parameter_id']
        bad=copy.deepcopy(s);bad['optimizer_states'][0]['state'][pid]={'step':torch.tensor(2.)}
        self.assertFalse(query_contract(m,c,bad,i)['passed'])
        p=next(m.pool.parameters());p.grad=torch.zeros_like(p)
        self.assertFalse(query_contract(m,c,s,i)['passed']);p.grad=None
        with torch.no_grad():p.flatten()[0].add_(1.)
        self.assertFalse(query_contract(m,c,s,i)['passed'])
        EVIDENCE['fixed_negatives']=['unexpected_AdamW_state','even_zero_gradient_tensor','changed_initial_tensor']

    def test_adaptive_Q_all9_grad_and_state_required(self):
        m,c,s,i=trained('adaptive');r=query_contract(m,c,s,i)
        self.assertTrue(r['passed']);pid=r['rows'][0]['optimizer_parameter_id']
        bad=copy.deepcopy(s);del bad['optimizer_states'][0]['state'][pid]
        self.assertFalse(query_contract(m,c,bad,i)['passed'])
        bad=copy.deepcopy(s);bad['optimizer_states'][0]['state'][pid]['exp_avg'].flatten()[0]=float('nan')
        self.assertFalse(query_contract(m,c,bad,i)['passed'])
        next(m.pool.parameters()).grad=None
        self.assertFalse(query_contract(m,c,s,i)['passed'])
        EVIDENCE['adaptive_negatives']=['missing_AdamW_state','nonfinite_AdamW_state','missing_gradient']

    def test_clip_observer_calls_original_once_and_preserves_result_grad_state_rng(self):
        m,_,_,_=trained('adaptive');plain=copy.deepcopy(m)
        for p,q in zip(m.parameters(),plain.parameters()):
            if p.grad is not None:q.grad=p.grad.clone()
        state={n:v.clone() for n,v in m.state_dict().items()}
        rng=torch.get_rng_state().clone();sentinel=object();calls=[];trace=[]
        def original(module,optimizer,threshold,algorithm):
            calls.append(1)
            torch.nn.utils.clip_grad_norm_(module.model.parameters(),threshold)
            return sentinel
        # The instrumentation delegates the real clipping operation. This test
        # supplies PyTorch's original operation; it does not implement clipping.
        expected=original(SimpleNamespace(model=plain),None,1.,'norm')
        calls.clear()
        result=observe_clip(original,SimpleNamespace(model=m,global_step=0),None,1.,'norm',trace.append)
        self.assertIs(result,expected);self.assertEqual(len(calls),1);self.assertEqual(len(trace),1)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for p,q in zip(m.parameters(),plain.parameters()):
            self.assertEqual(p.grad is None,q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        for n,v in state.items():torch.testing.assert_close(m.state_dict()[n],v,rtol=0,atol=0)
        EVIDENCE['clip_observer']=dict(original_calls=1,return_identity=True,all_gradients_exact=True,state_rng_unchanged=True,
            before=trace[0]['before']['total'],after=trace[0]['after']['total'])

    def test_branch_statistics_separate_offset_from_information(self):
        x=torch.ones(2,8,128,requires_grad=True)*3
        before=x.detach().clone();rng=torch.get_rng_state().clone()
        r=branch_statistics(x)
        for branch in r.values():
            self.assertEqual(branch['centered_rms'],0.)
            self.assertEqual(branch['mean_rms'],3.)
            self.assertEqual(branch['total_rms'],3.)
        torch.testing.assert_close(x,before,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        EVIDENCE['branch_statistics']=dict(constant_bias_not_counted_as_centered_signal=True,no_state_or_rng_change=True)

    def test_cached_token_numerics_preserves_live_pool_gradients_rng_and_flags(self):
        m,c,s,i=trained('adaptive')
        m.register_buffer('hybrid_query_mean',torch.zeros(64))
        m.register_buffer('hybrid_query_scale',torch.tensor(.0003))
        m.sigma0=.1
        tokens=torch.randn(8,5,192,generator=torch.Generator().manual_seed(92))
        precision=torch.get_float32_matmul_precision();tf32=torch.backends.cuda.matmul.allow_tf32
        r=query_numerics(m,i,tokens)
        self.assertEqual(len(r['rows']),4)
        self.assertEqual(r['encoder_calls'],0);self.assertEqual(r['optimizer_updates'],0)
        self.assertEqual(torch.get_float32_matmul_precision(),precision)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,tf32)
        self.assertTrue(r['state_grad_rng_and_global_flags_preserved'])
        EVIDENCE['cached_token_numerics']=dict(cpu_tokens=[8,5,192],cases=r['rows'],
            no_extra_encoder_or_update=True,live_pool_grad_rng_flags_unchanged=True,
            scope='CPU helper contract, not the production256-patch GPU measurement.')


if __name__=='__main__':
    result=unittest.main(exit=False).result
    import hashlib
    names=['preflight.py','train_preflight.py','test_preflight.py']
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,device='cpu',evidence=EVIDENCE,
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names})
    path=Path(os.environ.get('HYBRID_PREFLIGHT_CONTRACT','/tmp/hybrid-preflight-contracts.json'))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if report['passed'] else 1)
