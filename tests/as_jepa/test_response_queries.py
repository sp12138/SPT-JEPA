"""Contracts for optional clean-latent direct / action-annihilated queries."""
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import torch

from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops


class ResponseQueryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False),
                       device=torch.device("cpu"), parallel_state=None)

    def model(self, cls=RobustLeWM, mode="none", window=8):
        torch.manual_seed(11)
        extra = dict(response_query_mode=mode) if cls is RobustLeWM else {}
        model = cls(mode="L2", beta=.25, window=window, encoder_depth=1,
                    renderer_width=48, renderer_depth=2, renderer_heads=3,
                    rho=.8, forecast_fraction=.5, **extra)
        torch.nn.init.normal_(model.renderer.final_layer.linear.weight, std=.01)
        return model.eval()

    def batch(self, window=8):
        torch.manual_seed(18)
        return dict(pixels=torch.randint(0, 256, (2, window, 3, 224, 224), dtype=torch.uint8),
                    render=torch.randint(0, 256, (2, window, 3, 64, 64), dtype=torch.uint8),
                    anchor=torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
                    anchor224=torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8),
                    action=torch.randn(2, window, 10))

    def test_none_exact_old_loss_metrics_gradients_and_rng(self):
        source_path = os.environ.get("AS_JEPA_RESPONSE_REFERENCE")
        source = Path(source_path).read_text() if source_path else subprocess.check_output(
            ["git", "show", "b3f27e6:as_jepa_lewm/robust.py"],
            cwd=Path(__file__).resolve().parents[2], text=True)
        namespace = {"__name__": "robust_before_response_queries"}
        exec(compile(source, "<robust_reference_b3f27e6>", "exec"), namespace)
        old, new = self.model(namespace["RobustLeWM"], window=3), self.model(window=3)
        old.train(); new.train()
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        batch = self.batch(window=3)
        torch.manual_seed(29)
        first, fm = old.loss(batch, step=7)
        first.backward(); old_rng = torch.get_rng_state()
        torch.manual_seed(29)
        second, sm = new.loss(batch, step=7)
        second.backward()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(set(fm), set(sm))
        for key in fm:
            torch.testing.assert_close(fm[key], sm[key], rtol=0, atol=0)
        torch.testing.assert_close(old_rng, torch.get_rng_state(), rtol=0, atol=0)
        for (nx, x), (ny, y) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(nx, ny); self.assertEqual(x.grad is None, y.grad is None)
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_shared_old_tensors_head_rng_budget_and_strict_reload(self):
        baseline = self.model(); baseline_rng = torch.get_rng_state()
        first = None
        for mode in ["direct", "annihilator"]:
            model = self.model(mode=mode)
            torch.testing.assert_close(baseline_rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(set(model.state_dict())-set(baseline.state_dict()),
                             {"response_query_head.weight", "response_query_head.bias"})
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in model.parameters())-
                             sum(p.numel() for p in baseline.parameters()), 12480)
            if first is None:
                first = {k:v.detach().clone() for k,v in model.response_query_head.state_dict().items()}
            for key, value in first.items():
                torch.testing.assert_close(value, model.response_query_head.state_dict()[key], rtol=0, atol=0)
            copy = self.model(mode=mode)
            copy.load_state_dict(model.state_dict(), strict=True)

    def test_projector_rank_affine_action_units_padding_and_single_clip(self):
        torch.manual_seed(41)
        for dim, expected_rank in [(2,4),(5,1)]:
            actions = torch.randn(3,8,5,dim)
            flat = actions.flatten(2)
            p, rank = RobustLeWM.response_projector(flat)
            torch.testing.assert_close(rank, torch.full((3,), expected_rank))
            torch.testing.assert_close(p, p.transpose(1,2), rtol=0, atol=1e-6)
            torch.testing.assert_close(p@p, p, rtol=1e-5, atol=1e-6)
            cumulative = torch.cat([torch.zeros(3,1,dim),actions[:,:-1].sum(2).cumsum(1)],1)
            design = torch.cat([torch.ones(3,8,1),torch.arange(8)[None,:,None].expand(3,-1,-1),cumulative],-1)
            torch.testing.assert_close(p@design, torch.zeros_like(design), rtol=0, atol=3e-6)
            matrix = torch.eye(dim)*2+.03*torch.randn(dim,dim)
            transformed = (actions@matrix+torch.arange(dim)*.3+1).flatten(2)
            pp, rr = RobustLeWM.response_projector(transformed)
            torch.testing.assert_close(rank,rr)
            torch.testing.assert_close(p,pp,rtol=2e-5,atol=2e-6)
            padded = flat.clone(); padded[:,-1] = 1e6
            ppad,rpad = RobustLeWM.response_projector(padded)
            torch.testing.assert_close(p,ppad,rtol=0,atol=0)
            torch.testing.assert_close(rank,rpad)
            for i in range(3):
                pi,ri=RobustLeWM.response_projector(flat[i:i+1])
                torch.testing.assert_close(p[i:i+1],pi,rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(rank[i:i+1],ri)

    def test_projection_zero_rank_is_finite_and_uses_anchor(self):
        model = self.model(mode="annihilator", window=3)
        flat=torch.randn(2,3,64);actions=torch.randn(2,3,10)
        embedded=model.action_encoder(actions)
        render=torch.randn(2,3,3,64,64)
        loss,metrics=model.response_query_loss(flat,embedded,actions,render,7)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(metrics["response_query_zero_rank_fraction"]),1.)
        self.assertEqual(float(metrics["response_query_outcome"]),0.)
        torch.testing.assert_close(loss,metrics["response_query_anchor"])

    def test_loss_matches_independent_per_clip_least_squares_oracle(self):
        torch.manual_seed(57)
        flat=torch.randn(2,8,64);actions=torch.randn(2,8,10)
        embedded=torch.randn(2,8,64);render=torch.randn(2,8,3,64,64)
        for mode in ["direct","annihilator"]:
            model=self.model(mode=mode)
            def predict(history, control):
                return history+.05*control
            with patch.object(model,"predict",predict):
                actual,metrics=model.response_query_loss(flat,embedded,actions,render,7)
            # Explicit closed-form rollout and 8x8 block averages, independent
            # of the implementation's adaptive pooling and SVD projector.
            forecast=torch.cat([flat[:,:1],flat[:,:1]+.05*embedded[:,:-1].cumsum(1)],1)
            target=render.double().reshape(2,8,3,8,8,8,8).mean((4,6)).reshape(2,8,192)
            error=model.response_query_head(forecast).double()-target
            expected=[]
            for i in range(2):
                e=error[i]
                if mode=="direct":
                    query=e.square().mean()
                else:
                    cumulative=torch.cat([torch.zeros(1,2,dtype=torch.float64),
                        actions[i,:-1].double().reshape(7,5,2).sum(1).cumsum(0)],0)
                    x=torch.cat([torch.ones(8,1,dtype=torch.float64),
                        torch.arange(8,dtype=torch.float64)[:,None],cumulative],1)
                    coefficients=torch.linalg.solve(x.T@x,x.T@e)
                    residual=e-x@coefficients
                    query=residual.square().sum()/((8-x.shape[1])*192)
                expected.append(e[0].square().mean()+query)
            torch.testing.assert_close(actual.double(),torch.stack(expected).mean(),rtol=2e-6,atol=2e-6)
            self.assertEqual(float(metrics["response_query_residual_rank"]),8. if mode=="direct" else 4.)

    def test_auxiliary_is_causal_sample_level_and_reaches_encoder_predictor_head(self):
        batch=self.batch()
        for mode in ["direct","annihilator"]:
            model=self.model(mode=mode)
            pixels,render,_=model.prepare(batch)
            info=model.encode(dict(pixels=pixels,action=batch["action"]))
            before=torch.get_rng_state()
            loss,metrics=model.response_query_loss(info["emb"],info["act_emb"],batch["action"],render,7)
            torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
            altered=info["emb"].detach().clone();altered[:,1:]+=100
            changed,_=model.response_query_loss(altered,info["act_emb"],batch["action"],render,7)
            torch.testing.assert_close(loss,changed,rtol=0,atol=0)
            separate=[model.response_query_loss(info["emb"][i:i+1],info["act_emb"][i:i+1],
                       batch["action"][i:i+1],render[i:i+1],7)[0] for i in range(2)]
            torch.testing.assert_close(loss,torch.stack(separate).mean(),rtol=1e-4,atol=1e-6)
            self.assertGreater(float(metrics["response_query_outcome"]),0)
            loss.backward()
            for module in [model.encoder,model.pool,model.predictor,model.response_query_head]:
                gradients=[p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertTrue(any(bool(g.abs().sum()>0) for g in gradients))

    def test_optional_queries_preserve_native_metrics_and_outer_training_rng(self):
        baseline=self.model();baseline.train();batch=self.batch()
        torch.manual_seed(79);_,old=baseline.loss(batch,step=7);after=torch.get_rng_state()
        for mode in ["direct","annihilator"]:
            model=self.model(mode=mode);model.train()
            torch.manual_seed(79);loss,new=model.loss(batch,step=7)
            torch.testing.assert_close(after,torch.get_rng_state(),rtol=0,atol=0)
            for key in set(old)-{"loss"}:
                torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
            torch.testing.assert_close(loss,old["loss"]+new["response_query_aux"],rtol=0,atol=0)
            loss.backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.renderer.parameters()))


if __name__=="__main__":unittest.main()
