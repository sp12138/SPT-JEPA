"""Verify sample-local anchor augmentation and observed-context-only masks."""
import unittest
import torch
import test_frontier as contracts

class AnchorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contracts.FrontierContracts.setUpClass()

    def test_zero_dropout_exact_original_and_rng_preserved(self):
        h=contracts.FrontierContracts();b=h.batch();ref=h.model();new=h.model("anchor-dropout");new.anchor_dropout_p=0.
        torch.manual_seed(91);left,_=ref.loss(b);left.backward();rng=torch.get_rng_state().clone()
        torch.manual_seed(91);right,_=new.loss(b);right.backward()
        torch.testing.assert_close(left,right,rtol=0,atol=0);torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for (_,x),(_,y) in zip(ref.named_parameters(),new.named_parameters()):
            self.assertEqual(x.grad is None,y.grad is None)
            if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=2e-7)

    def test_drop_whole_sample_anchor_and_preserve_outer_rng(self):
        h=contracts.FrontierContracts();m=h.model("anchor-dropout");frames=torch.randn(12,5,3,64,64);anchor=torch.ones(12,3,64,64)
        m.anchor_dropout_p=1.;torch.testing.assert_close(m._training_anchor(frames,anchor,5),torch.zeros_like(anchor),rtol=0,atol=0)
        m.anchor_dropout_p=.5;rng=torch.get_rng_state().clone();value=m._training_anchor(frames,anchor,5)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        chosen=value.mean((1,2,3));self.assertTrue(bool(((chosen==0)|(chosen==1)).all()))
        self.assertTrue(bool((chosen==0).any()) and bool((chosen==1).any()))

    def test_context_mask_does_not_read_future_or_other_samples(self):
        h=contracts.FrontierContracts();m=h.model("anchor-context-mask");frames=torch.zeros(2,5,3,64,64);anchor=torch.zeros(2,3,64,64)
        anchor[:,:,20:30,20:30]=.7
        a=m._training_anchor(frames,anchor,0)
        future=frames.clone();future[:,1:]=torch.randn_like(future[:,1:])*10
        torch.testing.assert_close(a,m._training_anchor(future,anchor,0),rtol=0,atol=0)
        other=frames.clone();other[1]=torch.randn_like(other[1])
        torch.testing.assert_close(a[:1],m._training_anchor(other,anchor,0)[:1],rtol=0,atol=0)
        self.assertEqual(float(a.abs().sum()),0.)
        torch.testing.assert_close(m._training_anchor(anchor[:,None].expand(-1,5,-1,-1,-1),anchor,0),anchor,rtol=0,atol=0)

if __name__=="__main__":unittest.main()
