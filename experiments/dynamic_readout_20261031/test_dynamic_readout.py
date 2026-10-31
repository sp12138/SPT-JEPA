"""CPU contracts only; no campaign training or GPU launch."""
import hashlib,importlib.util,io,json,os,unittest
from pathlib import Path
import numpy as np
import torch
from as_jepa_lewm.dynamic_readout import DynamicReadout
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_lewm.factory import build_model
from vforge.ops import OpsConfig,initialize_ops
B=Path('/efs/prescottsun/SPT-JEPA')
FROZEN=B/'runs/local_raw_chart_source_20261031/source-v1'
EVIDENCE={}

def legacy():
    record=json.loads(FROZEN.with_suffix('.json').read_text())
    assert record['source_commit']=='0c9ae74840cf28247ae74341d2656e2d9ddd457c'
    for name in ['model','carrier','robust']:
        module=__import__('as_jepa_lewm.'+name,fromlist=['__file__'])
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==record['files'][f'as_jepa_lewm/{name}.py']
    path=FROZEN/'as_jepa_lewm/spatial_carrier.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['files']['as_jepa_lewm/spatial_carrier.py']
    spec=importlib.util.spec_from_file_location('frozen_0c_spatial_readout',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.SpatialCarrierLeWM

def batch(window,action_dim):
    g=torch.Generator().manual_seed(47)
    pixels=torch.randint(0,256,(1,window,3,224,224),generator=g,dtype=torch.uint8)
    render=torch.randint(0,256,(1,window,3,64,64),generator=g,dtype=torch.uint8)
    return dict(pixels=pixels,render=render,anchor=render[:,0],anchor224=pixels[:,0],action=torch.randn(1,window,action_dim,generator=g))

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)

    def test_full_native_original_tensors_loss_gradients_rng_and_startup(self):
        oldclass=legacy();rows=[];self.assertIs(SpatialCarrierLeWM.loss,RobustLeWM.loss)
        for task in ['tworoom','cube']:
            cfg=json.loads((B/f'runs/local_isotropic600_20261031/{task}-spatial-local-seed1/config.json').read_text())
            torch.manual_seed(1);old=oldclass(**cfg['model']).train();init_rng=torch.get_rng_state().clone()
            data=batch(cfg['model']['window'],cfg['model']['action_dim'])
            torch.manual_seed(63);loss,metrics=old.loss(data,step=600);loss.backward();rng=torch.get_rng_state().clone()
            original=old.state_dict();parameters=dict(old.named_parameters());self.assertEqual(len(original),553)
            heads={}
            for mode in ['none','static','state']:
                numpy_before=np.random.get_state();torch.manual_seed(1)
                new=SpatialCarrierLeWM(**cfg['model'],spatial_readout_mode=mode).train()
                torch.testing.assert_close(torch.get_rng_state(),init_rng,rtol=0,atol=0)
                np.testing.assert_array_equal(numpy_before[1],np.random.get_state()[1]);self.assertEqual(numpy_before[2:],np.random.get_state()[2:])
                for key,value in original.items():torch.testing.assert_close(new.state_dict()[key],value,rtol=0,atol=0)
                extra=set(new.state_dict())-set(original)
                self.assertEqual(extra,set()if mode=='none'else{'dynamic_readout_module.'+k for k in ['input.weight','input.bias','output.weight','output.bias']})
                self.assertEqual(sum(p.numel()for p in new.parameters())-sum(p.numel()for p in old.parameters()),0 if mode=='none'else 5376)
                if mode!='none':heads[mode]={k:v.detach().clone()for k,v in new.dynamic_readout_module.state_dict().items()}
                torch.manual_seed(63);actual,mm=new.loss(data,step=600);actual.backward()
                torch.testing.assert_close(actual,loss,rtol=0,atol=0);self.assertEqual(mm.keys(),metrics.keys())
                for key in metrics:torch.testing.assert_close(mm[key],metrics[key],rtol=0,atol=0)
                torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
                for name,p in new.named_parameters():
                    if name.startswith('dynamic_readout_module.'):
                        self.assertIsNotNone(p.grad);self.assertEqual(float(p.grad.abs().max()),0.)
                    else:
                        q=parameters[name].grad;self.assertEqual(p.grad is None,q is None)
                        if q is not None:torch.testing.assert_close(p.grad,q,rtol=0,atol=0)
                row=dict(task=task,seed=1,mode=mode,original_tensors_exact=553,loss_metrics_rng_exact=True,
                    all_original_gradients_exact=True,initialization_rng_exact=True,counts=new.spatial_parameter_counts())
                if task=='cube'and mode!='none':
                    selected=[p for n,p in new.named_parameters()if n.startswith(('predictor.transformer.output_proj.','dynamic_readout_module.'))]
                    optimizer=torch.optim.SGD(selected,lr=.001);startup=[]
                    for update in [1,2]:
                        optimizer.step();new.zero_grad(set_to_none=True)
                        torch.manual_seed(63);after,_=new.loss(data,step=600);after.backward();self.assertTrue(bool(torch.isfinite(after)))
                        head=new.dynamic_readout_module
                        grads={n:float(p.grad.abs().max())for n,p in head.named_parameters()}
                        self.assertGreater(grads['output.weight'],0);self.assertGreater(grads['output.bias'],0)
                        if update==1:self.assertEqual(grads['input.weight'],0);self.assertEqual(grads['input.bias'],0)
                        else:self.assertGreater(grads['input.weight'],0);self.assertGreater(grads['input.bias'],0)
                        if mode=='static':self.assertEqual(float(head.input.weight.grad[:,:16].abs().max()),0)
                        elif update==2:self.assertGreater(float(head.input.weight.grad[:,:16].abs().max()),0)
                        startup.append(dict(updates=update,loss=float(after.detach()),gradient_max=grads))
                    row['startup']=startup
                rows.append(row);del new,actual,mm
            for key in heads['static']:torch.testing.assert_close(heads['static'][key],heads['state'][key],rtol=0,atol=0)
            del old,parameters,original,loss,metrics
        EVIDENCE['native_regression']=rows

    def test_second_seed_initial_and_old_raw_chart_path(self):
        oldclass=legacy();rows=[]
        for task in ['tworoom','cube']:
            cfg=json.loads((B/f'runs/local_isotropic600_20261031/{task}-spatial-local-seed2/config.json').read_text())
            torch.manual_seed(2);old=oldclass(**cfg['model']);rng=torch.get_rng_state().clone()
            for mode in ['static','state']:
                torch.manual_seed(2);new=SpatialCarrierLeWM(**cfg['model'],spatial_readout_mode=mode)
                torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
                for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)
                rows.append(dict(task=task,seed=2,mode=mode,original_tensors_exact=553,rng_exact=True));del new
            del old
        kw=dict(spatial_mean=[.2]*1024,spatial_scale=[.7]*1024,spatial_chart='bounded',spatial_state_interface='raw',
            encoder_depth=1,window=3,renderer_width=48,renderer_depth=1,renderer_heads=3,rho=.8,forecast_fraction=.5,beta=.25)
        torch.manual_seed(31);old=oldclass(**kw);torch.manual_seed(31);new=SpatialCarrierLeWM(**kw)
        with torch.no_grad():old.carrier_chart_module.output.weight.fill_(.03)
        new.load_state_dict(old.state_dict(),strict=True);data=batch(3,10)
        torch.manual_seed(91);loss,metrics=old.loss(data,step=600);loss.backward();rng=torch.get_rng_state()
        torch.manual_seed(91);actual,mm=new.loss(data,step=600);actual.backward()
        torch.testing.assert_close(actual,loss,rtol=0,atol=0);torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for k,v in metrics.items():torch.testing.assert_close(mm[k],v,rtol=0,atol=0)
        for (n,p),(m,q)in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(n,m);self.assertEqual(p.grad is None,q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        EVIDENCE['second_seed_initial']=rows;EVIDENCE['nonzero_raw_chart_default_loss_grad_rng_exact']=True

    def test_shared_constructive_escape_and_static_fixed_span(self):
        head=DynamicReadout('state').double()
        # SharedMLP M_i0=tanh(u_i0+x_i)-tanh(x_i), remaining columns0.
        with torch.no_grad():
            for p in head.parameters():p.zero_()
            head.input.weight[0,0]=1;head.input.weight[0,16]=1;head.input.weight[1,16]=1
            head.output.weight[0,0]=1;head.output.weight[0,1]=-1
        v=torch.zeros(1,64,16);v[:,:,0]=1;v=v.flatten(1);u=torch.zeros_like(v)
        first=head(u,u+.5*v);second=head(first,first+.7*v)
        torch.testing.assert_close(first,.5*v,rtol=0,atol=0)
        q=(v/v.norm()).T;outside=second-(second@q)@q.T
        self.assertGreater(float(outside.norm()),.1)
        static=DynamicReadout('static');static.load_state_dict(head.state_dict(),strict=True)
        u=torch.zeros_like(v)
        for a in [.5,.7,-.4]:u=static(u,u+a*v)
        torch.testing.assert_close(u,.8*v,rtol=1e-6,atol=1e-6)
        # Nonzero static M still induces one fixed transformed span.
        with torch.no_grad():static.output.bias.fill_(.01)
        A=static.mixing_matrix(v);direction=(v.reshape(1,64,16)+A@v.reshape(1,64,16)).flatten(1)
        u=torch.zeros_like(v)
        for a in [.5,.7,-.4]:u=static(u,u+a*v)
        torch.testing.assert_close(u,.8*direction,rtol=1e-6,atol=1e-6)
        EVIDENCE['constructive']=dict(first_old_span_exact=True,second_outside_initial_span_norm=float(outside.norm()),
            static_fixed_span_exact_to_fp32=True,instantaneous_action_rank_not_claimed_increased=True)

    def test_sample_locality_causal_head_and_strict_roundtrip(self):
        g=torch.Generator().manual_seed(18);u=torch.randn(3,5,1024,generator=g);p=u+torch.randn(u.shape,generator=g)*.1
        head=DynamicReadout('state')
        with torch.no_grad():head.output.weight.copy_(torch.randn(head.output.weight.shape,generator=g)*.03)
        y=head(u,p);other=u.clone();other[1:]+=3
        torch.testing.assert_close(head(other,p)[0],y[0],rtol=0,atol=0)
        changed=u.clone();changed[:,3:]+=4
        torch.testing.assert_close(head(changed,p)[:,:3],y[:,:3],rtol=0,atol=0)
        stream=io.BytesIO();torch.save(head.state_dict(),stream);stream.seek(0);restored=DynamicReadout('state')
        restored.load_state_dict(torch.load(stream,weights_only=True),strict=True);torch.testing.assert_close(restored(u,p),y,rtol=0,atol=0)
        v=u[:,0].clone()
        for _ in range(32):v=head(v,v+torch.sin(v)*.03)
        self.assertTrue(bool(torch.isfinite(v).all()))
        cfg=dict(model_class='spatial_carrier',model=dict(spatial_readout_mode='state',spatial_mean=[0.]*1024,
            spatial_scale=[1.]*1024,encoder_depth=1,window=2,renderer_width=48,renderer_depth=1,renderer_heads=3))
        a=build_model(cfg);b=build_model(cfg);b.load_state_dict(a.state_dict(),strict=True)
        self.assertEqual(a.audit_expected_gradients()['dynamic_readout_module'],'active')
        EVIDENCE['interface']=dict(sample_local=True,no_cross_time_head_input=True,strict_head_checkpoint=True,
            existing_factory_strict_model_load=True,finite_32step_constructive_rollout=True,parameters=5376,
            static_inactive_input_weights=1024,macs_per_state=401408)

    def test_cached_fp32_subtraction_and_invalid_combinations(self):
        rows=[]
        for task in ['tworoom','cube']:
            for seed in [1,2]:
                path=B/f'runs/local_isotropic6000_20261031/transition-{task}-spatial-local-seed{seed}/state-audit.npz'
                with np.load(path)as z:
                    pairs={'one_step':(z['teacher_forced_prefix'],z['observed'][:,:-1]),
                        'rollout_steps':(z['clean_rollout'],np.concatenate([z['observed'][:,:1],z['clean_rollout'][:,:-1]],1))}
                for mode,(pred,state)in pairs.items():
                    exact=pred.astype(float)-state.astype(float);actual=(pred-state).astype(float)
                    relative=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact))
                    self.assertLess(relative,1e-6)
                    rows.append(dict(task=task,seed=seed,mode=mode,delta_rms=float(np.sqrt(np.square(exact).mean())),
                        fp32_subtraction_relative_error=relative,max_abs_error=float(np.abs(actual-exact).max())))
        for kw in [dict(spatial_readout_mode='bad'),dict(spatial_readout_mode='state',spatial_chart='bounded'),
            dict(spatial_readout_mode='state',spatial_mode='dense'),dict(spatial_readout_mode='static',spatial_basis='orthogonal')]:
            with self.assertRaises(ValueError):SpatialCarrierLeWM(**kw)
        EVIDENCE['cached_subtraction']=rows
        EVIDENCE['subtraction_limit']='Measures only recovering increments from saved FP32 p,u, not the residual already lost when native carry p=u+delta was rounded. At exact zero carry init the new head starts with zero gradient.'

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    EVIDENCE.update(passed=result.wasSuccessful(),tests=result.testsRun,model_source_sha256=hashlib.sha256(Path(__import__('as_jepa_lewm.spatial_carrier',fromlist=['__file__']).__file__).read_bytes()).hexdigest(),
        helper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),device='CPU',cuda_initialized=torch.cuda.is_initialized())
    if os.environ.get('DYNAMIC_READOUT_CONTRACT_OUTPUT'):Path(os.environ['DYNAMIC_READOUT_CONTRACT_OUTPUT']).write_text(json.dumps(EVIDENCE,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful()else 1)
