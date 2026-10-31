"""CPU contracts for raw internal-and-public termination; no benchmark training."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
import as_jepa_lewm.carrier as current

B = Path('/efs/prescottsun/SPT-JEPA')
OLD = B/'runs/raw_chart_precision_verify_20261031/source-v1/as_jepa_lewm/carrier.py'
CAPTURE = B/'runs/raw_chart_public_round_capture_20261031/capture/capture.pt'
EVIDENCE = {}


def old_module():
    spec = importlib.util.spec_from_file_location('raw_carrier1060', OLD)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class TerminationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_actual_adjoint_four_fails_eight_passes(self):
        old = old_module(); saved = torch.load(CAPTURE, map_location='cpu', weights_only=False)
        d = saved['tensors']; m = saved['metadata']; value = lambda key: d[key]['value']
        args = [value('target')]+[value('closure_'+k) for k in ['first','second','derivative']]+[m['bound'],m['iterations'],m['rtol']]
        with self.assertRaisesRegex(RuntimeError, 'public-dtype'): old._precise_adjoint(*args)
        flags = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
        with patch.object(current, '_rounded_solution_certified', wraps=current._rounded_solution_certified) as checker:
            public, solved = current._precise_adjoint(*args)
        # Checks at4 and8, followed by unchanged defensive final certification.
        self.assertEqual(checker.call_count, 3)
        target, first, second, derivative, bound, _, rtol = args
        residual = lambda x: bound*F.linear(F.linear(x, second.T)*derivative, first.T)
        x = target.double()
        for _ in range(8): x = target.double()-residual(x)
        torch.testing.assert_close(solved, x, rtol=0, atol=0)
        torch.testing.assert_close(public, x.float(), rtol=0, atol=0)
        self.assertEqual(flags, (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32))
        ratio = (public.double()+residual(public.double())-target.double()).norm(dim=-1)/.5/(rtol*target.double().norm(dim=-1)).clamp_min(torch.finfo(torch.float64).tiny)
        EVIDENCE['actual_adjoint'] = dict(original4_raises=True, repaired8_passes=True,
            all_full_shape_outputs_exact_eight_update_oracle=True, shape=list(target.shape),
            maximum_public_error_over_tolerance=float(ratio.max()), unchanged_global_flags=True)

    def test_forward_boundary_and_unattainable_public_precision(self):
        # Analytic scalar fixture: after4 updates, internal error is .995tol;
        # actual FP32 rounding makes it1.01456tol. No benchmark-derived setting.
        target = torch.ones(1,1,dtype=torch.float64); coefficient = (.995e-6)**.2
        residual = lambda x: coefficient*x
        old = old_module(); internal = old._contractive_solve(target,residual,.5,64,1e-6,1e-6,'forward fixture')
        with self.assertRaisesRegex(RuntimeError, 'public-dtype'):
            old._certify_rounded_solution(internal.float(),target,residual,.5,1e-6,1e-6,'forward fixture')
        with patch.object(current, '_rounded_solution_certified', wraps=current._rounded_solution_certified) as checker:
            result = current._public_contractive_solve(target,residual,.5,64,1e-6,1e-6,'forward fixture',torch.float32)
        self.assertEqual(checker.call_count,2)
        current._certify_rounded_solution(result.float(),target,residual,.5,1e-6,1e-6,'forward fixture')
        # For g=2^-149 and T(x)=1.25x, every FP32 x=k*g has
        # |T(x)-g|/(1-.5)>=.5g >> 1e-6*g. More iteration cannot fix it.
        tiny_target = torch.tensor([[2.**-149]],dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'public-dtype.*64'):
            current._public_contractive_solve(tiny_target,lambda x:.25*x,.5,64,0.,1e-6,'unattainable fixture',torch.float32)
        EVIDENCE['forward_and_impossible'] = dict(coefficient=coefficient, original4_public_fails=True,
            repaired8_public_passes=True, impossible_native_rtol_subnormal_public_still_fails64=True)

    def test_previously_passing_inverse_outputs_all_gradients_rng_exact(self):
        old = old_module()
        self.assertEqual(inspect.getsource(current._contractive_solve), inspect.getsource(old._contractive_solve))
        left = old.ResidualCarrierChart(16,64,'bounded',.5).enable_raw_precision()
        right = current.ResidualCarrierChart(16,64,'bounded',.5).enable_raw_precision()
        g = torch.Generator().manual_seed(129)
        with torch.no_grad():
            left.output.weight.copy_(torch.randn(left.output.weight.shape,generator=g)*.02)
            left.output.bias.copy_(torch.randn(left.output.bias.shape,generator=g)*.01)
        right.load_state_dict(left.state_dict(),strict=True)
        x = torch.randn(7,16,generator=g); direction = torch.randn(7,16,generator=g)
        results=[]
        for chart in [left,right]:
            value=x.clone().requires_grad_(); before=torch.get_rng_state().clone()
            y=chart.differentiable_inverse(value)
            gradients=torch.autograd.grad((y*direction).sum(),[value]+list(chart.parameters()))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            results.append((y,gradients))
        torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
        for a,b in zip(results[0][1],results[1][1]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        EVIDENCE['previously_passing'] = dict(nonzero_chart=True, public_outputs_exact=True,
            input_and_all_four_parameter_gradients_exact=True, rng_exact=True, legacy_solver_source_exact=True,
            boundary='Exact checks cover this previously passing nonzero-chart fixture; separate native identity and legacy contracts cover full models. No claim that all possible inputs converge.')


if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TerminationContracts))
    EVIDENCE.update(passed=result.wasSuccessful(),tests=result.testsRun,device='CPU',
        model_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        original1060_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),
        capture_sha256=hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if os.environ.get('PUBLIC_TERMINATION_CONTRACT_OUTPUT'):
        Path(os.environ['PUBLIC_TERMINATION_CONTRACT_OUTPUT']).write_text(json.dumps(EVIDENCE,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
