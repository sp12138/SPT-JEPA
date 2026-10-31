"""CPU regression and geometry contracts for the shared local residual chart."""
import hashlib, importlib.util, io, json, os, unittest
from pathlib import Path
import numpy as np
import torch

from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from as_jepa_lewm.carrier import ResidualCarrierChart
from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops

B=Path('/efs/prescottsun/SPT-JEPA')
LEGACY=B/'runs/local_reviewed6000_20261031/source-v1/as_jepa_lewm/spatial_carrier.py'
LEGACY_SHA='794f41914bd32f4bdd653f591877cd456b772af2215d6e08719bcb41f7d2b07a'
EVIDENCE={}


def legacy_class():
    assert hashlib.sha256(LEGACY.read_bytes()).hexdigest()==LEGACY_SHA
    spec=importlib.util.spec_from_file_location('legacy_da_spatial',LEGACY)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.SpatialCarrierLeWM


def slim(mode='bounded'):
    torch.manual_seed(31)
    return SpatialCarrierLeWM(spatial_chart=mode,spatial_mean=[.2]*1024,spatial_scale=[.7]*1024,
        encoder_depth=1,window=2,renderer_width=48,renderer_depth=1,renderer_heads=3,rho=.8,forecast_fraction=.5,beta=.25)


class SharedChartContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)

    def test_full_native_default_identity_loss_all_gradients_and_initial_rng(self):
        legacy=legacy_class();checks=[]
        self.assertIs(SpatialCarrierLeWM.loss,RobustLeWM.loss)
        for task in ['tworoom','cube']:
            cfg=json.loads((B/f'runs/local_isotropic600_20261031/{task}-spatial-local-seed1/config.json').read_text())
            torch.manual_seed(cfg['seed']);old=legacy(**cfg['model']).train();rng=torch.get_rng_state().clone()
            g=torch.Generator().manual_seed(58);w=cfg['model']['window'];ad=cfg['model']['action_dim']
            pixels=torch.randint(0,256,(1,w,3,224,224),generator=g,dtype=torch.uint8)
            render=torch.randint(0,256,(1,w,3,64,64),generator=g,dtype=torch.uint8)
            batch=dict(pixels=pixels,render=render,anchor=render[:,0],anchor224=pixels[:,0],action=torch.randn(1,w,ad,generator=g))
            torch.manual_seed(74);loss,metrics=old.loss(batch,step=600);loss.backward();after=torch.get_rng_state().clone()
            oldparams=dict(old.named_parameters());original=old.state_dict()
            for mode in ['none','identity','bounded']:
                numpy_before=np.random.get_state()
                torch.manual_seed(cfg['seed']);new=SpatialCarrierLeWM(**cfg['model'],spatial_chart=mode).train()
                torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
                np.testing.assert_array_equal(numpy_before[1],np.random.get_state()[1])
                self.assertEqual(numpy_before[2:],np.random.get_state()[2:])
                for k,v in original.items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)
                self.assertEqual(len(original),553)
                extra=set(new.state_dict())-set(original)
                self.assertEqual(extra,set() if mode=='none' else {'carrier_chart_module.'+n for n in
                    ['input.weight','input.bias','output.weight','output.bias']})
                if mode=='none':self.assertEqual(new.spatial_parameter_counts(),old.spatial_parameter_counts())
                else:
                    count=new.spatial_parameter_counts();self.assertEqual(count['chart'],2128)
                    self.assertEqual(count['total']-old.spatial_parameter_counts()['total'],2128)
                    self.assertEqual(count['chart_trainable'],0 if mode=='identity' else 2128)
                torch.manual_seed(74);actual,mm=new.loss(batch,step=600);actual.backward()
                torch.testing.assert_close(actual,loss,rtol=0,atol=0);self.assertEqual(mm.keys(),metrics.keys())
                for k,v in metrics.items():torch.testing.assert_close(mm[k],v,rtol=0,atol=0)
                torch.testing.assert_close(torch.get_rng_state(),after,rtol=0,atol=0)
                for n,p in new.named_parameters():
                    if n.startswith('carrier_chart_module.'):
                        if mode=='identity':self.assertFalse(p.requires_grad);self.assertIsNone(p.grad)
                        else:self.assertIsNotNone(p.grad);self.assertTrue(bool(torch.isfinite(p.grad).all()))
                        continue
                    q=oldparams[n];self.assertEqual(p.grad is None,q.grad is None)
                    if q.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                row=dict(task=task,seed=1,mode=mode,full_native_config=True,batch=1,window=w,
                    original_initial_tensors_exact=553,initialization_rng_exact=True,loss_metrics_rng_exact=True,
                    all_original_gradients_exact=True,counts=new.spatial_parameter_counts())
                if mode=='bounded':
                    chart=new.carrier_chart_module;before=chart.norm_certificate()
                    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in chart.parameters()))
                    x=torch.randn(3,1024,generator=torch.Generator().manual_seed(99))
                    torch.testing.assert_close(new.transform_spatial_chart(x),x,rtol=0,atol=0)
                    optimizer=torch.optim.SGD(chart.parameters(),lr=.01);optimizer.step()
                    delta=float((new.transform_spatial_chart(x)-x).abs().max())
                    self.assertGreater(delta,0)
                    recovered=chart.inverse(new.transform_spatial_chart(x).reshape(3,64,16)).reshape_as(x)
                    torch.testing.assert_close(recovered,x,rtol=2e-6,atol=2e-6)
                    row.update(initial_certificate=before,after_native_gradient_chart_update=chart.norm_certificate(),
                        output_change_max=delta,inverse_max_abs=float((recovered-x).abs().max()),
                        update_scope='One CPU SGD step on chart only from the unchanged native loss gradient; semantic contract, not a performance experiment.')
                checks.append(row);del new,actual,mm
            del old,oldparams,original,loss,metrics
        EVIDENCE['full_native_regression']=checks
        EVIDENCE['legacy_source_sha256']=LEGACY_SHA

    def test_second_seed_all_initial_tensors_and_private_rng(self):
        legacy=legacy_class();rows=[]
        for task in ['tworoom','cube']:
            cfg=json.loads((B/f'runs/local_isotropic600_20261031/{task}-spatial-local-seed2/config.json').read_text())
            torch.manual_seed(2);old=legacy(**cfg['model']);rng=torch.get_rng_state().clone()
            torch.manual_seed(2);new=SpatialCarrierLeWM(**cfg['model'],spatial_chart='bounded')
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)
            rows.append(dict(task=task,seed=2,original_initial_tensors_exact=len(old.state_dict()),rng_equal=True));del old,new
        EVIDENCE['second_seed_initial']=rows

    def test_shared_slot_geometry_inverse_and_checkpoint(self):
        model=slim();chart=model.carrier_chart_module
        self.assertIs(type(chart),ResidualCarrierChart)
        g=torch.Generator().manual_seed(92)
        with torch.no_grad():
            for p in chart.parameters():p.copy_(torch.randn(p.shape,generator=g)*3)
        certificate=chart.norm_certificate()
        self.assertLessEqual(certificate['residual_lipschitz_upper'],.500001)
        self.assertTrue(certificate['input_normalization_active'] and certificate['output_normalization_active'])
        x=torch.randn(32,1024,generator=g);y=torch.randn(32,1024,generator=g)
        tx=model.transform_spatial_chart(x);ty=model.transform_spatial_chart(y)
        ratios=(tx-ty).norm(dim=-1)/(x-y).norm(dim=-1)
        self.assertGreaterEqual(float(ratios.min()),.5-1e-6);self.assertLessEqual(float(ratios.max()),1.5+1e-6)
        recovered=chart.inverse(tx.reshape(32,64,16)).reshape_as(x)
        torch.testing.assert_close(recovered,x,rtol=4e-6,atol=4e-6)
        shared=torch.randn(1,16,generator=g).expand(64,16).reshape(1,1024)
        slots=model.transform_spatial_chart(shared).reshape(64,16)
        torch.testing.assert_close(slots,slots[:1].expand_as(slots),rtol=0,atol=0)
        changed=x.clone().reshape(32,64,16);changed[:,7]+=1
        delta=(model.transform_spatial_chart(changed.reshape_as(x))-tx).reshape(32,64,16)
        self.assertTrue(bool((delta[:,[i for i in range(64) if i!=7]]==0).all()))
        stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0);restored=slim()
        restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
        torch.testing.assert_close(restored.transform_spatial_chart(x),tx,rtol=0,atol=0)
        self.assertEqual(restored.audit_expected_gradients()['carrier_chart_module'],'active')
        EVIDENCE['geometry']=dict(certificate=certificate,shared_slots=64,slot_width=16,parameters=2128,
            global_euclidean_lower=.5,global_euclidean_upper=1.5,sampled_distance_ratio_min=float(ratios.min()),
            sampled_distance_ratio_max=float(ratios.max()),inverse_max_abs=float((recovered-x).abs().max()),
            no_cross_slot_mixing=True,shared_map_exact=True,checkpoint_roundtrip_exact=True,
            proof='Apply the per-slot(.5,1.5) distance bounds, square and sum across64 slots. Inverse is the same contraction independently per slot. Bound is exact-arithmetic; floating-point/TF32 arithmetic is not an exact isometry theorem.')

    def test_invalid_combinations_fail_before_model_construction(self):
        for kw in [dict(spatial_chart='free'),dict(spatial_chart='bounded',spatial_mode='dense'),
                   dict(spatial_chart='bounded',spatial_basis='orthogonal')]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):SpatialCarrierLeWM(**kw)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SharedChartContracts))
    EVIDENCE.update(passed=result.wasSuccessful(),tests=result.testsRun,device='CPU',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        model_source_sha256=hashlib.sha256(Path(__import__('as_jepa_lewm.spatial_carrier',fromlist=['__file__']).__file__).read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if os.environ.get('SPATIAL_CHART_CONTRACT_OUTPUT'):
        Path(os.environ['SPATIAL_CHART_CONTRACT_OUTPUT']).write_text(json.dumps(EVIDENCE,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
