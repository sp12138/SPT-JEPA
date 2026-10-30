"""Contracts for optional clean-latent direct / action-annihilated queries."""
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import torch

from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops


AUXILIARY_MODES = ["direct", "annihilator", "direct_relative", "annihilator_relative", "random_relative"]


class ResponseQueryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False),
                       device=torch.device("cpu"), parallel_state=None)

    def model(self, cls=RobustLeWM, mode="none", window=8, encoded_fraction=None):
        torch.manual_seed(11)
        extra = dict(response_query_mode=mode) if cls is RobustLeWM or mode != "none" else {}
        if encoded_fraction is not None:
            extra["response_query_encoded_fraction"] = encoded_fraction
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
        for mode in AUXILIARY_MODES:
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
        for mode in ["annihilator", "annihilator_relative", "random_relative"]:
            model = self.model(mode=mode, window=3)
            flat=torch.randn(2,3,64);actions=torch.randn(2,3,10)
            embedded=model.action_encoder(actions)
            render=torch.randn(2,3,3,64,64)
            loss,metrics=model.response_query_loss(flat,embedded,actions,render,7)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertEqual(float(metrics["response_query_zero_rank_fraction"]),1.)
            self.assertEqual(float(metrics["response_query_outcome"]),0.)
            torch.testing.assert_close(loss,metrics["response_query_anchor"])
            if mode.endswith("_relative"):
                self.assertEqual(float(metrics["response_query_floor_dominated_fraction"]),0.)

    def test_loss_matches_independent_per_clip_least_squares_oracle(self):
        torch.manual_seed(57)
        flat=torch.randn(2,8,64);actions=torch.randn(2,8,10)
        embedded=torch.randn(2,8,64);render=torch.randn(2,8,3,64,64)
        for mode in AUXILIARY_MODES:
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
                if mode in {"direct", "direct_relative"}:
                    query=e.square().mean()
                    target_energy=target[i].square().mean()
                elif mode=="random_relative":
                    # Explicit sums using the independently property-tested
                    # public operator, not implementation batched arithmetic.
                    p,rank=model.response_query_operator(actions[i:i+1])
                    residual=torch.stack([(p[0,j].double()[:,None]*e).sum(0) for j in range(8)])
                    y_residual=torch.stack([(p[0,j].double()[:,None]*target[i]).sum(0) for j in range(8)])
                    query=residual.square().sum()/(int(rank[0])*192)
                    target_energy=y_residual.square().sum()/(int(rank[0])*192)
                else:
                    cumulative=torch.cat([torch.zeros(1,2,dtype=torch.float64),
                        actions[i,:-1].double().reshape(7,5,2).sum(1).cumsum(0)],0)
                    x=torch.cat([torch.ones(8,1,dtype=torch.float64),
                        torch.arange(8,dtype=torch.float64)[:,None],cumulative],1)
                    coefficients=torch.linalg.solve(x.T@x,x.T@e)
                    residual=e-x@coefficients
                    query=residual.square().sum()/((8-x.shape[1])*192)
                    target_residual=target[i]-x@torch.linalg.solve(x.T@x,x.T@target[i])
                    target_energy=target_residual.square().sum()/((8-x.shape[1])*192)
                if mode.endswith("_relative"):
                    query=query/(target_energy+(1/127.5)**2)
                expected.append(e[0].square().mean()+query)
            torch.testing.assert_close(actual.double(),torch.stack(expected).mean(),rtol=2e-6,atol=2e-6)
            self.assertEqual(float(metrics["response_query_residual_rank"]),8. if mode.startswith("direct") else 4.)

    def test_auxiliary_is_causal_sample_level_and_reaches_encoder_predictor_head(self):
        batch=self.batch()
        for mode in AUXILIARY_MODES:
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
        for mode in AUXILIARY_MODES:
            model=self.model(mode=mode);model.train()
            torch.manual_seed(79);loss,new=model.loss(batch,step=7)
            torch.testing.assert_close(after,torch.get_rng_state(),rtol=0,atol=0)
            for key in set(old)-{"loss"}:
                torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
            torch.testing.assert_close(loss,old["loss"]+new["response_query_aux"],rtol=0,atol=0)
            loss.backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.renderer.parameters()))

    def test_current_absolute_modes_exact_against_frozen_80fb(self):
        source_path=os.environ.get("AS_JEPA_RESPONSE_ABSOLUTE_REFERENCE")
        source=Path(source_path).read_text() if source_path else subprocess.check_output(
            ["git","show","80fb:as_jepa_lewm/robust.py"],
            cwd=Path(__file__).resolve().parents[2],text=True)
        namespace={"__name__":"robust_before_relative_queries"}
        exec(compile(source,"<robust_reference_80fb>","exec"),namespace)
        for mode in ["direct","annihilator"]:
            old=self.model(namespace["RobustLeWM"],mode=mode)
            new=self.model(mode=mode)
            old.train();new.train();batch=self.batch()
            for key,value in old.state_dict().items():
                torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
            torch.manual_seed(109);first,old_metrics=old.loss(batch,step=7)
            first.backward();old_rng=torch.get_rng_state()
            torch.manual_seed(109);second,new_metrics=new.loss(batch,step=7)
            second.backward()
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            self.assertEqual(set(old_metrics),set(new_metrics))
            for key,value in old_metrics.items():
                torch.testing.assert_close(value,new_metrics[key],rtol=0,atol=0)
            torch.testing.assert_close(old_rng,torch.get_rng_state(),rtol=0,atol=0)
            for (name,x),(new_name,y) in zip(old.named_parameters(),new.named_parameters()):
                self.assertEqual(name,new_name);self.assertEqual(x.grad is None,y.grad is None)
                if x.grad is not None:
                    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)

    def test_public_operators_and_random_rank_constant_time_rng_contract(self):
        model=self.model(mode="random_relative")
        torch.manual_seed(137);before=torch.get_rng_state()
        model._response_random_basis.cache_clear()
        first=model._response_random_basis(8).clone()
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
        model._response_random_basis.cache_clear();torch.manual_seed(999)
        second=model._response_random_basis(8)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        time=torch.stack([torch.ones(8),torch.arange(8)],1).float()
        for dim,expected in [(2,4),(5,1)]:
            actions=torch.randn(3,8,5*dim)
            before=torch.get_rng_state();p,rank=model.response_query_operator(actions)
            torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
            action_p,action_rank=model.response_projector(actions)
            torch.testing.assert_close(rank,action_rank)
            torch.testing.assert_close(rank,torch.full((3,),expected))
            torch.testing.assert_close(p,p.transpose(1,2),rtol=0,atol=1e-6)
            torch.testing.assert_close(p@p,p,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(p@time,torch.zeros(3,8,2),rtol=0,atol=1e-6)
            torch.testing.assert_close(p.diagonal(dim1=1,dim2=2).sum(1),rank.float(),rtol=1e-6,atol=1e-6)
            self.assertGreater(float((p-action_p).square().sum()),.1)
            alternate,_=model.response_query_operator(torch.randn_like(actions))
            torch.testing.assert_close(p,alternate,rtol=0,atol=0)
            for i in range(3):
                single,_=model.response_query_operator(actions[i:i+1])
                torch.testing.assert_close(p[i:i+1],single,rtol=0,atol=0)
            model.response_query_mode="direct_relative"
            identity,direct_rank=model.response_query_operator(actions)
            torch.testing.assert_close(identity,torch.eye(8)[None].expand(3,-1,-1),rtol=0,atol=0)
            torch.testing.assert_close(direct_rank,torch.full((3,),8))
            model.response_query_mode="annihilator_relative"
            exact,exact_rank=model.response_query_operator(actions)
            torch.testing.assert_close(exact,action_p,rtol=0,atol=0)
            torch.testing.assert_close(exact_rank,action_rank)
            model.response_query_mode="random_relative"
        # The largest possible random subspace is the complete 1/time complement.
        full,full_rank=model.response_query_operator(torch.zeros(1,8,10))
        expected=torch.eye(8)-time@torch.linalg.solve(time.T@time,time.T)
        self.assertEqual(int(full_rank[0]),6)
        torch.testing.assert_close(full[0],expected,rtol=1e-5,atol=1e-6)

    def test_relative_denominator_detached_floor_and_unscaled_shared_anchor(self):
        for mode in ["direct_relative","annihilator_relative","random_relative"]:
            model=self.model(mode=mode)
            torch.nn.init.zeros_(model.response_query_head.weight)
            torch.nn.init.zeros_(model.response_query_head.bias)
            flat=torch.randn(2,8,64,requires_grad=True)
            embedded=torch.randn(2,8,64);actions=torch.randn(2,8,10)
            render=torch.randn(2,8,3,64,64)*.2
            render[0]=0.;render.requires_grad_()
            loss,metrics=model.response_query_loss(flat,embedded,actions,render,7)
            # First clip has zero energy; second has positive energy. Averages
            # must be taken after per-clip division, not a ratio of means.
            operator,rank=model.response_query_operator(actions)
            target=render.detach().double().reshape(2,8,3,8,8,8,8).mean((4,6)).reshape(2,8,192)
            projected=operator.double()@target
            energy=projected.square().sum((1,2))/(rank*192)
            expected_query=(energy/(energy+(1/127.5)**2)).mean()
            expected_anchor=target[:,0].square().mean()
            torch.testing.assert_close(metrics["response_query_outcome"].double(),expected_query,rtol=2e-6,atol=2e-6)
            torch.testing.assert_close(metrics["response_query_raw_outcome"].double(),energy.mean(),rtol=2e-6,atol=2e-6)
            torch.testing.assert_close(metrics["response_query_target_energy"].double(),energy.mean(),rtol=2e-6,atol=2e-6)
            torch.testing.assert_close(loss.double(),expected_anchor+expected_query,rtol=2e-6,atol=2e-6)
            self.assertEqual(float(metrics["response_query_floor_dominated_fraction"]),.5)
            loss.backward()
            self.assertIsNone(render.grad)
            self.assertTrue(bool(torch.isfinite(model.response_query_head.weight.grad).all()))

    def test_encoded_fraction_zero_exact_and_validation(self):
        reference=os.environ.get("AS_JEPA_RESPONSE_RELATIVE_REFERENCE")
        if not reference:
            self.skipTest("Set AS_JEPA_RESPONSE_RELATIVE_REFERENCE to the audited relative-only source")
        namespace={"__name__":"robust_before_encoded_route"}
        exec(compile(Path(reference).read_text(),"<relative_only_reference>","exec"),namespace)
        for mode in ["direct_relative","annihilator_relative","random_relative"]:
            old=self.model(namespace["RobustLeWM"],mode=mode)
            new=self.model(mode=mode,encoded_fraction=0.)
            old.train();new.train();batch=self.batch()
            torch.manual_seed(173);first,old_metrics=old.loss(batch,step=7)
            first.backward();old_rng=torch.get_rng_state()
            torch.manual_seed(173);second,new_metrics=new.loss(batch,step=7)
            second.backward()
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            self.assertEqual(set(old_metrics),set(new_metrics))
            for key,value in old_metrics.items():
                torch.testing.assert_close(value,new_metrics[key],rtol=0,atol=0)
            torch.testing.assert_close(old_rng,torch.get_rng_state(),rtol=0,atol=0)
            self.assertEqual(set(old.state_dict()),set(new.state_dict()))
            for (name,x),(new_name,y) in zip(old.named_parameters(),new.named_parameters()):
                self.assertEqual(name,new_name);self.assertEqual(x.grad is None,y.grad is None)
                torch.testing.assert_close(x,y,rtol=0,atol=0)
                if x.grad is not None:
                    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
        for mode,fraction in [("none",.5),("direct_relative",-.1),("direct_relative",1.1),("direct_relative",float("nan"))]:
            with self.assertRaises(ValueError):
                self.model(mode=mode,encoded_fraction=fraction)

    def test_encoded_route_weighted_formula_causality_rng_and_same_budget(self):
        torch.manual_seed(191)
        flat=torch.randn(2,8,64);actions=torch.randn(2,8,10)
        embedded=torch.randn(2,8,64);render=torch.randn(2,8,3,64,64)
        target=render.double().reshape(2,8,3,8,8,8,8).mean((4,6)).reshape(2,8,192)
        for mode in AUXILIARY_MODES:
            baseline=self.model(mode=mode);baseline_rng=torch.get_rng_state()
            model=self.model(mode=mode,encoded_fraction=.5)
            torch.testing.assert_close(baseline_rng,torch.get_rng_state(),rtol=0,atol=0)
            self.assertEqual(set(baseline.state_dict()),set(model.state_dict()))
            for key,value in baseline.state_dict().items():
                torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
            def predict(history,control):return history+.05*control
            before=torch.get_rng_state()
            with patch.object(model,"predict",side_effect=predict) as calls:
                actual,metrics=model.response_query_loss(flat,embedded,actions,render,7)
                self.assertEqual(calls.call_count,7)
            torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
            forecast=torch.cat([flat[:,:1],flat[:,:1]+.05*embedded[:,:-1].cumsum(1)],1)
            forecast_error=model.response_query_head(forecast).double()-target
            encoded_error=model.response_query_head(flat).double()-target
            p,rank=model.response_query_operator(actions);p=p.double()
            fq=(p@forecast_error).square().sum((1,2))/(rank*192)
            eq=(p@encoded_error).square().sum((1,2))/(rank*192)
            if mode.endswith("_relative"):
                denominator=(p@target).square().sum((1,2))/(rank*192)+(1/127.5)**2
                fq=fq/denominator;eq=eq/denominator
            anchor=forecast_error[:,0].square().mean(-1)
            expected=(anchor+.5*fq+.5*eq).mean()
            torch.testing.assert_close(actual.double(),expected,rtol=3e-6,atol=3e-6)
            torch.testing.assert_close(metrics["response_query_forecast_outcome"].double(),fq.mean(),rtol=3e-6,atol=3e-6)
            torch.testing.assert_close(metrics["response_query_encoded_outcome"].double(),eq.mean(),rtol=3e-6,atol=3e-6)
            altered=flat.clone();altered[:,1:]+=10
            with patch.object(model,"predict",side_effect=predict):
                _,changed=model.response_query_loss(altered,embedded,actions,render,7)
            torch.testing.assert_close(metrics["response_query_forecast_outcome"],changed["response_query_forecast_outcome"],rtol=0,atol=0)
            torch.testing.assert_close(metrics["response_query_anchor"],changed["response_query_anchor"],rtol=0,atol=0)
            self.assertGreater(float(abs(metrics["response_query_encoded_outcome"]-changed["response_query_encoded_outcome"])),1e-5)
            with patch.object(model,"predict",side_effect=predict):
                pieces=[model.response_query_loss(flat[i:i+1],embedded[i:i+1],actions[i:i+1],render[i:i+1],7)[0] for i in range(2)]
            torch.testing.assert_close(actual,torch.stack(pieces).mean(),rtol=1e-5,atol=1e-6)

    def test_projected_carry_cancellation_and_observed_per_frame_gradient(self):
        torch.manual_seed(211)
        actions=torch.randn(2,8,10);embedded=torch.randn(2,8,64)
        render=.2*torch.randn(2,8,3,1,1).expand(-1,-1,-1,64,64)
        base_flat=torch.randn(2,8,64)
        for mode in ["annihilator_relative","random_relative"]:
            gradients=[]
            for fraction in [0.,.5]:
                model=self.model(mode=mode,encoded_fraction=fraction)
                flat=base_flat.clone().requires_grad_()
                # An exact carry predictor uses no later observation and yields
                # a constant predicted code. P1=0 removes its query gradients.
                with patch.object(model,"predict",side_effect=lambda history,control: history):
                    _,metrics=model.response_query_loss(flat,embedded,actions,render,7)
                grad=torch.autograd.grad(metrics["response_query_outcome"],
                    [flat,model.response_query_head.weight,model.response_query_head.bias])
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grad))
                gradients.append(grad)
            carry,grounded=gradients
            self.assertLess(float(carry[0].norm()),1e-5)
            self.assertLess(float(carry[1].norm()),1e-4)
            self.assertLess(float(carry[2].norm()),1e-5)
            self.assertGreater(float(grounded[0][:,1:].norm()),1e-3)
            self.assertGreater(float(grounded[1].norm()),1e-3)
            self.assertGreater(float(grounded[0][:,1:].norm()),1000*float(carry[0].norm()))

    def test_mixed_route_full_training_loss_preserves_native_stream_and_gradients(self):
        baseline=self.model();baseline.train();batch=self.batch()
        torch.manual_seed(227);_,native=baseline.loss(batch,step=7);rng=torch.get_rng_state()
        model=self.model(mode="annihilator_relative",encoded_fraction=.5);model.train()
        torch.manual_seed(227);loss,metrics=model.loss(batch,step=7)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key in set(native)-{"loss"}:
            torch.testing.assert_close(native[key],metrics[key],rtol=0,atol=0)
        torch.testing.assert_close(metrics["response_query_outcome"],
            .5*metrics["response_query_forecast_outcome"]+.5*metrics["response_query_encoded_outcome"],rtol=1e-6,atol=1e-6)
        loss.backward()
        for module in [model.encoder,model.pool,model.predictor,model.response_query_head,model.renderer]:
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertTrue(any(bool(g.abs().sum()>0) for g in grads))


if __name__=="__main__":unittest.main()
