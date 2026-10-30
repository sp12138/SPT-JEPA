"""CPU proof/default-regression contracts for the optional full1024 basis."""
import hashlib,importlib.util,io,json,os,unittest
from pathlib import Path
import numpy as np
import torch
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM,orthogonal_basis
from vforge.models.as_jepa.codes import entry_nats,surprise_nats
from vforge.ops import OpsConfig,initialize_ops

B=Path('/efs/prescottsun/SPT-JEPA')
LEGACY=B/'runs/local_isotropic_source_20261031/source-v2/as_jepa_lewm/spatial_carrier.py'
LEGACY_SHA='d43b646e048f6b1f3ba0b8b74baf7171d1d0bef4cacedcddbbe228ad9cf7a755'
EVIDENCE={}


def slim(basis='identity'):
    torch.manual_seed(31)
    return SpatialCarrierLeWM(spatial_mode='local',spatial_basis=basis,spatial_mean=[.2]*1024,spatial_scale=[.7]*1024,
        encoder_depth=1,window=2,renderer_width=48,renderer_depth=1,renderer_heads=3,rho=.8,forecast_fraction=.5,beta=.25)


class BasisContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)

    def test_full1024_orthogonality_and_private_rng(self):
        tr=torch.get_rng_state().clone();nr=np.random.get_state();q=torch.from_numpy(orthogonal_basis())
        torch.testing.assert_close(torch.get_rng_state(),tr,rtol=0,atol=0)
        np.testing.assert_array_equal(nr[1],np.random.get_state()[1]);self.assertEqual(nr[2:],np.random.get_state()[2:])
        self.assertEqual(q.shape,(1024,1024));self.assertTrue(bool((q.abs()==1/32).all()))
        error=(q.double().T@q.double()-torch.eye(1024,dtype=torch.double)).abs().max().item();self.assertEqual(error,0.)
        EVIDENCE['basis']=dict(shape=[1024,1024],fixed_buffer_bytes=q.numel()*q.element_size(),orthogonality_max_error_fp64=error,
            sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest(),all_coordinates_mix_1024_inputs=True,family='fixed signed/permuted Hadamard, not Haar')

    def test_default_full_native_configs_match_source32_loss_all_gradients_rng(self):
        self.assertEqual(hashlib.sha256(LEGACY.read_bytes()).hexdigest(),LEGACY_SHA)
        spec=importlib.util.spec_from_file_location('source32_spatial_legacy',LEGACY);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        checks=[]
        for task in ('tworoom','cube'):
            cfg=json.loads((B/f'runs/local_isotropic_preflight_20261031/{task}-spatial-local-seed1/config.json').read_text())
            torch.manual_seed(cfg['seed']);old=legacy.SpatialCarrierLeWM(**cfg['model']).train();rng=torch.get_rng_state().clone()
            torch.manual_seed(cfg['seed']);new=SpatialCarrierLeWM(**cfg['model']).train()
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            self.assertEqual(set(old.state_dict()),set(new.state_dict()))
            for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)
            self.assertEqual(old.spatial_parameter_counts(),new.spatial_parameter_counts())
            g=torch.Generator().manual_seed(58);w=cfg['model']['window'];ad=cfg['model']['action_dim']
            pixels=torch.randint(0,256,(1,w,3,224,224),generator=g,dtype=torch.uint8)
            render=torch.randint(0,256,(1,w,3,64,64),generator=g,dtype=torch.uint8)
            batch=dict(pixels=pixels,render=render,anchor=render[:,0],anchor224=pixels[:,0],action=torch.randn(1,w,ad,generator=g))
            torch.manual_seed(74);a,am=old.loss(batch,step=600);a.backward();end=torch.get_rng_state().clone()
            torch.manual_seed(74);b,bm=new.loss(batch,step=600);b.backward()
            torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertEqual(am.keys(),bm.keys())
            for k in am:torch.testing.assert_close(am[k],bm[k],rtol=0,atol=0)
            torch.testing.assert_close(torch.get_rng_state(),end,rtol=0,atol=0)
            oldparams=dict(old.named_parameters());nonzero=0
            for n,p in new.named_parameters():
                q=oldparams[n];self.assertEqual(p.grad is None,q.grad is None)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0);nonzero+=int(bool(p.grad.abs().sum()>0))
            checks.append(dict(task=task,seed=1,full_native_config=True,batch=1,window=w,state_keys=len(new.state_dict()),
                parameter_tensors=len(oldparams),nonzero_gradient_tensors=nonzero,loss_equal=True,all_gradients_equal=True,all_metrics_equal=True,rng_equal=True))
            del old,new,oldparams,a,b,am,bm
        EVIDENCE['default_source32_regression']=checks;EVIDENCE['legacy_source_sha256']=LEGACY_SHA

    def test_rotated_parameters_rng_actual_encoded_distances_and_roundtrip(self):
        normal=slim().eval();rng=torch.get_rng_state().clone();rotated=slim('orthogonal').eval()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for (n,a),(m,b) in zip(normal.named_parameters(),rotated.named_parameters()):
            self.assertEqual(n,m);torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(set(rotated.state_dict())-set(normal.state_dict()),{'spatial_orthogonal_basis'})
        pixels=torch.randint(0,256,(2,2,3,224,224),generator=torch.Generator().manual_seed(95),dtype=torch.uint8)
        x=normal.encode(dict(pixels=normal.normalize(pixels)))['emb'];y=rotated.encode(dict(pixels=rotated.normalize(pixels)))['emb']
        expected=(x.double()@rotated.spatial_orthogonal_basis.double()).float();torch.testing.assert_close(y,expected,rtol=0,atol=0)
        dx=torch.pdist(x.flatten(0,1).double()).square();dy=torch.pdist(y.flatten(0,1).double()).square()
        relative=((dy-dx).abs()/dx).max().item();self.assertLess(relative,2e-7)
        recovered=y.double()@rotated.spatial_orthogonal_basis.double().T
        inverse_error=(recovered-x.double()).abs().max().item();self.assertLess(inverse_error,2e-6)
        stream=io.BytesIO();torch.save(rotated.state_dict(),stream);stream.seek(0);restored=slim('orthogonal').eval()
        restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
        torch.testing.assert_close(restored.encode(dict(pixels=restored.normalize(pixels)))['emb'],y,rtol=0,atol=0)
        EVIDENCE['actual_encoded_metric']=dict(max_pairwise_squared_distance_relative_error=relative,max_inverse_abs_error=inverse_error,
            checkpoint_encode_exact=True,learned_initial_tensors_and_torch_rng_equal=True)

    def test_native_rate_only_under_rotated_posterior_and_prediction(self):
        q=torch.from_numpy(orthogonal_basis()).double();g=torch.Generator().manual_seed(106)
        mu=torch.randn(3,8,1024,generator=g);noise=torch.randn(3,8,1024,generator=g);rho=.8;sigma=.1
        for t in range(1,8):noise[:,t]=rho*noise[:,t-1]+(1-rho*rho)**.5*noise[:,t]
        noisy=mu+sigma*noise;pred=torch.randn(3,7,1024,generator=g)
        rotate=lambda z:(z.double()@q).float()
        def rates(m,n,p):
            target=m[:,1:]+rho*(n[:,:-1]-m[:,:-1])
            return entry_nats(m[:,0].reshape(3,64,16),sigma),surprise_nats(target.reshape(3,7,64,16),p.reshape(3,7,64,16),sigma)
        a=rates(mu,noisy,pred);b=rates(rotate(mu),rotate(noisy),rotate(pred))
        errors=[]
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y,rtol=3e-7,atol=1e-5);errors.append(float(((x-y).abs()/x.abs()).max()))
        # Equal noise draw is not the pathwise rotation coupling: explicitly show it differs.
        wrong=rotate(mu)+sigma*noise
        mismatch=float((wrong-rotate(noisy)).square().mean().sqrt());self.assertGreater(mismatch,.1)
        EVIDENCE['native_rate']=dict(entry_max_relative_error=errors[0],surprise_max_relative_error=errors[1],
            same_draw_vs_rotated_noise_rms_difference=mismatch,scope='Native entry/surprise functions with jointly rotated posterior/target/prediction; not same-weight predictor equivariance.')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BasisContracts))
    EVIDENCE.update(passed=result.wasSuccessful(),tests=result.testsRun,device='CPU',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        model_source_sha256=hashlib.sha256(Path(__import__('as_jepa_lewm.spatial_carrier',fromlist=['__file__']).__file__).read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if os.environ.get('SPATIAL_CONTRACT_OUTPUT'):Path(os.environ['SPATIAL_CONTRACT_OUTPUT']).write_text(json.dumps(EVIDENCE,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
