"""Focused CPU semantic contracts; real batch128 memory requires GPU preflight."""
import io
import unittest

import numpy as np
import torch

from as_jepa_lewm.robust import RobustLeWM
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM,projection_matrix
from vforge.ops import OpsConfig,initialize_ops


class SpatialContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)

    def model(self,mode='local',stats=True):
        torch.manual_seed(11)
        extra=dict(spatial_mean=[.2]*1024,spatial_scale=[2.]*1024) if stats else {}
        return SpatialCarrierLeWM(spatial_mode=mode,encoder_depth=1,window=2,
            renderer_width=48,renderer_depth=1,renderer_heads=3,rho=.8,forecast_fraction=.5,beta=.25,**extra)

    def test_paired_learnable_tensors_and_global_rng(self):
        first=self.model();reference={n:p.detach().clone() for n,p in first.named_parameters()};rng=torch.get_rng_state().clone()
        before=np.random.get_state();second=self.model('dense')
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for n,p in second.named_parameters():torch.testing.assert_close(p,reference[n],rtol=0,atol=0)
        after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])
        self.assertEqual(first.spatial_parameter_counts()['total'],second.spatial_parameter_counts()['total'])
        self.assertEqual(first.spatial_parameter_counts()['trainable'],second.spatial_parameter_counts()['trainable'])
        self.assertEqual(first.spatial_projection.numel(),3072);self.assertEqual(second.spatial_projection.numel(),49152*1024)
        self.assertIs(SpatialCarrierLeWM.loss,RobustLeWM.loss);self.assertIs(SpatialCarrierLeWM.predict,RobustLeWM.predict)

    def test_exact_audited_projection_rules_and_addresses(self):
        local=self.model();dense=self.model('dense')
        expected=np.random.default_rng(np.random.SeedSequence([20261030,192,16])).standard_normal((192,16)).astype(np.float32)/np.float32(4.)
        np.testing.assert_array_equal(local.spatial_projection.numpy(),expected)
        for block,start in enumerate(range(0,1024,128)):
            matrix=np.random.default_rng(np.random.SeedSequence([20261030,49152,1024,block])).standard_normal((49152,128),dtype=np.float32)
            matrix*=np.float32(1/32)
            np.testing.assert_array_equal(dense.spatial_projection[:,start:start+128].numpy(),matrix)
        patches=torch.zeros(1,1,256,192);patches[0,0,3*16+5,7]=1.
        value=local.project_patches(patches).reshape(8,8,16)
        torch.testing.assert_close(value[1,2],local.spatial_projection[7]/4,rtol=0,atol=0)
        self.assertEqual(int((value.norm(dim=-1)>0).sum()),1)
        torch.testing.assert_close(dense.project_patches(patches).flatten(),dense.spatial_projection[3*16*192+5*192+7],rtol=0,atol=0)

    def test_checkpoint_roundtrip_and_fixed_normalization(self):
        model=self.model().eval();patches=torch.randn(2,2,256,192,generator=torch.Generator().manual_seed(3))
        expected=model.project_patches(patches)
        target=self.model();stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
        target.load_state_dict(torch.load(stream,weights_only=True),strict=True)
        torch.testing.assert_close(target.project_patches(patches),expected,rtol=0,atol=0)
        target.train();self.assertFalse(target.encoder.training);self.assertFalse(target.pool.training)
        self.assertTrue(target.predictor.training)
        raw=target.raw_carrier;target.raw_carrier=lambda pixels:expected
        value=target.encode(dict(pixels=torch.empty(0)))['emb'];target.raw_carrier=raw
        torch.testing.assert_close(value,(expected-.2)/2,rtol=0,atol=0)
        with self.assertRaises(RuntimeError):self.model(stats=False).encode(dict(pixels=torch.empty(0)))

    def test_finite_native_loss_gradients_and_frozen_encoder(self):
        model=self.model().train();g=torch.Generator().manual_seed(19)
        pixels=torch.randint(0,256,(1,2,3,224,224),generator=g,dtype=torch.uint8)
        render=torch.randint(0,256,(1,2,3,64,64),generator=g,dtype=torch.uint8)
        batch=dict(pixels=pixels,render=render,anchor=render[:,0],anchor224=pixels[:,0],
                   action=torch.randn(1,2,10,generator=g))
        torch.manual_seed(29);loss,_=model.loss(batch,step=2);self.assertTrue(bool(torch.isfinite(loss)));loss.backward()
        for name,p in model.named_parameters():
            if name.startswith(('encoder.','pool.')):self.assertFalse(p.requires_grad);self.assertIsNone(p.grad)
            if p.grad is not None:self.assertTrue(bool(torch.isfinite(p.grad).all()))
        for prefix in ('predictor.','renderer.'):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters() if n.startswith(prefix)))


if __name__=='__main__':unittest.main()
