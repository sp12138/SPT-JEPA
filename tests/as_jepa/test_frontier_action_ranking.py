"""Check sample-level action alternatives and prevent candidate-slot shortcuts."""
import math
import unittest
from unittest.mock import patch
import torch
import test_frontier as contracts


class ActionRankingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contracts.FrontierContracts.setUpClass()

    def test_zero_weight_matches_original_loss_and_all_gradients(self):
        helper=contracts.FrontierContracts();batch=helper.batch()
        ref=helper.model();torch.manual_seed(68);base,_=ref.loss(batch);base.backward()
        for mode in ["one-step","terminal"]:
            model=helper.model("action-ranking");model.ranking_mode=mode;model.ranking_weight=0.
            torch.manual_seed(68);loss,_=model.loss(batch);loss.backward()
            torch.testing.assert_close(loss,base,rtol=0,atol=0)
            for (name,x),(_,y) in zip(ref.named_parameters(),model.named_parameters()):
                self.assertEqual(x.grad is None,y.grad is None,name)
                if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=2e-7)

    def test_duplicate_actions_no_candidate_identity_signal_train_dropout(self):
        helper=contracts.FrontierContracts()
        for mode in ["one-step","terminal"]:
            model=helper.model("action-ranking").train();model.ranking_mode=mode
            mu=torch.randn(2,5,64,requires_grad=True)
            raw=torch.randn(2,1,10).expand(-1,5,-1).clone()
            rng=torch.get_rng_state().clone()
            loss,costs=model._action_ranking(mu,raw,3)
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            torch.testing.assert_close(costs,costs[:,:1].expand_as(costs),rtol=0,atol=0)
            self.assertAlmostEqual(float(loss.detach()),math.log(3),places=6)
            grads=torch.autograd.grad(loss,[mu]+list(model.predictor.parameters()),allow_unused=True)
            self.assertTrue(all(g is None or float(g.abs().max())<2e-5 for g in grads))

    def test_terminal_loss_factorizes_and_has_finite_gradients(self):
        helper=contracts.FrontierContracts();batch=helper.batch();model=helper.model("action-ranking")
        model.ranking_mode="terminal";model.ranking_weight=.001
        # The inherited carry prior starts with a zero output projection; open its action gradient path explicitly.
        torch.nn.init.normal_(model.predictor.transformer.output_proj.weight,std=.01)
        for block in model.predictor.transformer.layers:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.01)
        fixednoise=lambda x,*args,**kwargs:torch.full_like(x,.3)
        fixedtime=lambda n,device,shift:torch.full((n,1),.5,device=device)
        with patch("torch.randn_like",fixednoise),patch("vforge.models.as_jepa.flow.draw_timesteps",fixedtime):
            full,_=model.loss(batch,step=3)
            parts=[model.loss({k:v[i:i+1] for k,v in batch.items()},step=3)[0] for i in range(2)]
        torch.testing.assert_close(full,torch.stack(parts).mean(),rtol=5e-5,atol=5e-6)
        full.backward()
        for prefix in ["encoder.","predictor.","action_encoder.","renderer."]:
            gradients=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertTrue(gradients and all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertTrue(any(bool(g.abs().sum()>0) for g in gradients))


if __name__=="__main__":unittest.main()
