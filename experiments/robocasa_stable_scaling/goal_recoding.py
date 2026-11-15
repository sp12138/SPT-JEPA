"""Infer goal codes in one observed RGB context, with a frozen pixel decoder.

This is decoder-assisted inference, not an E-only linear probe. No body or
action-head training occurs. The goal is an inference target only. At the
maximum-noise endpoint, eps + v(eps, 1, z, anchor) estimates the conditional
image mean. A shared-noise pair of latent fits cancels the recoder's origin
bias: E(initial) + P_anchor(goal) - P_anchor(initial).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from experiments.robocasa_action_probe import probe
from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import sha


def render_pixels(observation, size):
    """Match factual RGB preprocessing, including the original 224px crop."""
    import cv2
    cv2.setNumThreads(1)
    images = []
    assert set(observation) == {"img_0", "img_1"}
    for key in ("img_0", "img_1"):
        frame = observation[key]
        assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3
        h, w = frame.shape[:2]; left = (w-h)//2
        image = cv2.resize(frame[:, left:left+h], (224, 224), interpolation=cv2.INTER_AREA)
        if size != 224:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        images.append(image.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(images)).float().cuda()/127.5-1.


class ConditionalGoalInference(nn.Module):
    def __init__(self, renderer, rate_coefficient, render_size=64, mc_samples=4, max_iter=16):
        super().__init__()
        self.renderer = renderer.eval().requires_grad_(False)
        self.rate_coefficient = float(rate_coefficient)
        self.render_size = render_size
        self.mc_samples = mc_samples
        self.max_iter = max_iter
        assert render_size in (64, 224) and rate_coefficient >= 0

    def image_mean(self, codes, anchor, noise):
        # Both endpoints and both synchronized cameras belong to one process.
        m, views = noise.shape[:2]
        assert codes.ndim == 4 and codes.shape[:2] == (2, views)
        eps = noise[:, None].expand(m, 2, *noise.shape[1:]).flatten(0, 2)
        anchors = anchor[None, None].expand(m, 2, *anchor.shape).flatten(0, 2)
        condition = codes[None].expand(m, *codes.shape).flatten(0, 2)
        t = eps.new_ones((len(eps), 1))
        with self.renderer.anchored(anchors):
            prediction = eps+self.renderer(eps, t, condition=condition)
        return prediction.reshape(m, 2, views, *prediction.shape[1:]).mean(0)

    def energy(self, codes, target, anchor, noise):
        mean = self.image_mean(codes, anchor, noise)
        distortion = (mean-target).square().mean()
        entry = codes.square().sum((-2, -1)).mean()/2
        return distortion+self.rate_coefficient*entry, distortion, entry

    def solve(self, codes, pixels):
        """One physical example; optimize only a temporary latent leaf."""
        from vforge.models.as_jepa.pixels import pack
        assert codes.ndim == 4 and codes.shape[:2] == (2, 2)
        assert pixels.shape == (2, 2, 3, self.render_size, self.render_size)
        assert all(not p.requires_grad and p.grad is None for p in self.parameters())
        target = pack(pixels.detach().flatten(0, 1), patch=self.render_size//16).reshape(
            2, 2, -1, 1, 16, 16)
        anchor = target[0].detach()
        generator = torch.Generator(device=codes.device).manual_seed(20261115)
        noise = torch.randn((self.mc_samples, *anchor.shape), generator=generator,
            device=codes.device, dtype=torch.float32)
        heldout = torch.randn(noise.shape, generator=generator, device=noise.device)
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            latent = codes.detach().float().clone().requires_grad_()
            optimizer = torch.optim.LBFGS([latent], max_iter=self.max_iter,
                max_eval=2*self.max_iter, line_search_fn="strong_wolfe")
            with torch.no_grad():
                before = [float(x) for x in self.energy(latent, target, anchor, noise)]
                heldout_before = float(self.energy(latent, target, anchor, heldout)[1])
            calls = 0

            def closure():
                nonlocal calls
                optimizer.zero_grad(set_to_none=True)
                loss = self.energy(latent, target, anchor, noise)[0]
                assert torch.isfinite(loss)
                loss.backward(); calls += 1
                assert latent.grad is not None and torch.isfinite(latent.grad).all()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                after = [float(x) for x in self.energy(latent, target, anchor, noise)]
                heldout_after = float(self.energy(latent, target, anchor, heldout)[1])
        result = codes[0].detach()+(latent[1].detach()-latent[0].detach())
        assert torch.isfinite(result).all()
        assert all(not p.requires_grad and p.grad is None for p in self.parameters())
        return result, dict(energy_before=before, energy_after=after,
            heldout_distortion_before=heldout_before, heldout_distortion_after=heldout_after,
            latent_change_norm=float((latent.detach()-codes).norm()), closure_calls=calls,
            mc_samples=self.mc_samples, max_iter=self.max_iter, flow_time=1.,
            only_temporary_latent_optimized=True, no_action_labels_or_simulator_metadata=True)


class ActionTangentGoalInference(ConditionalGoalInference):
    """Project a visual change onto the frozen model's current action tangent.

    The inferred three-vector is a virtual displacement, not a training label
    or an executed command. It is unbounded; actual control retains the existing
    .3m norm cap. This approximation may fail away from the local tangent.
    """
    def solve(self, codes, pixels, basis, initial_displacement=None, optimize=True):
        from vforge.models.as_jepa.pixels import pack
        assert codes.shape[:2] == (2, 2) and basis.shape == (128, 3)
        assert all(not p.requires_grad and p.grad is None for p in self.parameters())
        target = pack(pixels.detach().flatten(0, 1), patch=self.render_size//16).reshape(
            2, 2, -1, 1, 16, 16)
        anchor = target[0].detach(); change = target[1]-target[0]
        origin = codes[0].detach(); basis = basis.detach()
        generator = torch.Generator(device=codes.device).manual_seed(20261115)
        noise = torch.randn((self.mc_samples, *anchor.shape), generator=generator,
            device=codes.device, dtype=torch.float32)
        heldout = torch.randn(noise.shape, generator=generator, device=noise.device)

        def energy(unit_displacement, fields):
            delta = (basis@(.3*unit_displacement)).reshape_as(origin)
            pair = torch.stack((origin, origin+delta))
            predicted = self.image_mean(pair, anchor, fields)
            return (predicted[1]-predicted[0]-change).square().mean()

        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            initial = origin.new_zeros(3) if initial_displacement is None else initial_displacement.detach()/.3
            assert initial.shape == (3,) and torch.isfinite(initial).all()
            displacement = initial.clone().requires_grad_()
            optimizer = torch.optim.LBFGS([displacement], max_iter=self.max_iter,
                max_eval=2*self.max_iter, line_search_fn="strong_wolfe")
            with torch.no_grad():
                before = float(energy(displacement, noise))
                heldout_before = float(energy(displacement, heldout))
            calls = 0

            def closure():
                nonlocal calls
                optimizer.zero_grad(set_to_none=True)
                loss = energy(displacement, noise)
                assert torch.isfinite(loss)
                loss.backward(); calls += 1
                assert displacement.grad is not None and torch.isfinite(displacement.grad).all()
                return loss

            if optimize:
                optimizer.step(closure)
            with torch.no_grad():
                after = float(energy(displacement, noise))
                heldout_after = float(energy(displacement, heldout))
        delta = (basis@(.3*displacement.detach())).reshape_as(origin)
        result = origin+delta
        assert torch.isfinite(result).all()
        assert all(not p.requires_grad and p.grad is None for p in self.parameters())
        return result, dict(energy_before=[before], energy_after=[after],
            heldout_distortion_before=heldout_before, heldout_distortion_after=heldout_after,
            latent_change_norm=float(delta.norm()), closure_calls=calls,
            virtual_displacement_m=(.3*displacement.detach()).cpu().tolist(),
            mc_samples=self.mc_samples, max_iter=self.max_iter, flow_time=1.,
            representation_constraint="frozen_forward_action_tangent",
            initialized_from_frozen_head=initial_displacement is not None,
            initial_virtual_displacement_m=(.3*initial).cpu().tolist(),
            optimizer_enabled=optimize,
            matched_noise_image_change_objective=True, virtual_displacement_unbounded=True,
            only_temporary_latent_optimized=True, no_action_labels_or_simulator_metadata=True)


class GoalPolicy(probe.Policy):
    def __init__(self, root, arm="spt64-linear"):
        super().__init__(root, arm)
        from experiments.robocasa_stable_scaling.train import build
        from vforge.models.as_jepa.constants import FRAME_ELEMS
        from vforge.ops import get_ops_config, initialize_ops
        from vforge.ops.config import OpsConfig
        if get_ops_config() is None:
            initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False),
                device=torch.device("cuda"), parallel_state=None)
        plan = json.loads((root/"plan.json").read_text())
        run = Path(plan["source"])/"training"/plan["source_models"][arm]
        cfg = json.loads((run/"config.json").read_text())
        assert not cfg.get("joint_views", False) and cfg.get("pixel_objective") is None
        assert cfg.get("coding_family", "gaussian") == "gaussian"
        folder = run/plan["comparison_checkpoint"]
        ready = json.loads((folder/"ready.json").read_text())
        assert ready["passed"] and sha(folder/"weights.pt") == ready["weights_sha256"]
        model = build(root, cfg)
        model.load_state_dict(torch.load(folder/"weights.pt", map_location="cpu", weights_only=True), strict=True)
        assert common.state_hash(model.state_dict()) == ready["nn_sha256"]
        assert ready["nn_sha256"] == self.readout.parent["full_nn_sha256"]
        self.state_slots, self.state_width = model.state_slots, model.state_width
        self.recoder = ConditionalGoalInference(model.renderer,
            model.beta/FRAME_ELEMS, getattr(model, "raw_render_size", 64)).cuda()
        self.response = None
        self.head_initialization = plan.get("goal_initialization") == "detached-head"
        self.goal_optimizer = plan.get("goal_optimizer", True)
        if plan.get("goal_code_constraint") == "action-tangent":
            assert cfg["variant"] == "spt-control-affine" and cfg.get("control_integrator", "euler") == "euler"
            assert self.state_slots == 4
            self.response = model.predictor.field.cuda().eval().requires_grad_(False)
            self.response_input_map = model.action_encoder.weight[:, 7:10].detach().clone().cuda()/.3
            self.response_input_map_before = self.response_input_map.clone()
            self.response_before = common.state_hash(self.response.state_dict())
            self.recoder = ActionTangentGoalInference(model.renderer,
                0., getattr(model, "raw_render_size", 64)).cuda()
        self.recoder_before = common.state_hash(self.recoder.state_dict())
        self.recoder_versions = [p._version for p in self.recoder.parameters()]
        self.goal_inference = None

    def initialize_goal(self, goal, initial):
        torch.cuda.synchronize(); start = time.perf_counter()
        current_code = self.encode(initial)
        other_code = (current_code if self.response is not None and not self.head_initialization
            else self.encode(goal))
        code = torch.stack((current_code, other_code)).reshape(
            2, 2, self.state_slots, self.state_width)
        pixels = torch.stack((render_pixels(initial, self.recoder.render_size),
            render_pixels(goal, self.recoder.render_size)))
        if self.response is None:
            value, proof = self.recoder.solve(code, pixels)
        else:
            with torch.no_grad():
                coefficients = self.response(current_code.reshape(2, 64)).reshape(2, 64, 65)
                basis = (coefficients[..., 1:]@self.response_input_map).reshape(128, 3)
            assert torch.linalg.matrix_rank(basis) == 3 and torch.isfinite(basis).all()
            initial = None
            if self.head_initialization:
                initial = self.weight@(other_code-current_code)
                initial = (initial*torch.clamp(.3/initial.norm().clamp_min(1e-12), max=1.)).detach()
            value, proof = self.recoder.solve(code, pixels, basis, initial, self.goal_optimizer)
        torch.cuda.synchronize()
        proof.update(seconds=time.perf_counter()-start,
            corrected_goal_shift_norm=float((value-code[1]).norm()),
            decoder_assisted_inference=True, E_only_linear_probe=False)
        self.goal_inference = proof
        return value.flatten()

    def unchanged(self):
        return super().unchanged() and (
            common.state_hash(self.recoder.state_dict()) == self.recoder_before and
            self.recoder_versions == [p._version for p in self.recoder.parameters()] and
            all(not p.requires_grad and p.grad is None for p in self.recoder.parameters()) and
            (self.response is None or (common.state_hash(self.response.state_dict()) == self.response_before and
                torch.equal(self.response_input_map, self.response_input_map_before) and
                all(not p.requires_grad and p.grad is None for p in self.response.parameters()))))


def control(cell, shard, shards):
    from experiments.robocasa_compare.evaluate import case_name
    torch.set_num_threads(2); torch.set_float32_matmul_precision("highest")
    policy = GoalPolicy(cell)
    plan = json.loads((cell/"plan.json").read_text())
    if plan.get("check_goal_identity_before_control", False):
        # A factual TRAIN image, with no goal-case archive or action labels.
        record = next(r for r in json.loads((cell/"data.json").read_text())["records"] if r["split"] == "train")
        path = record.get("local_observations", Path("/tmp/spt-time-compare-20261112/observations")/(record["id"]+".npz"))
        with np.load(path) as stream:
            image = stream["render"][0].copy()
        observation = {key: image[v].transpose(1, 2, 0).copy() for v, key in enumerate(("img_0", "img_1"))}
        value = policy.initialize_goal(observation, observation)
        assert torch.equal(value, policy.encode(observation)) and policy.unchanged()
        assert np.array_equal(policy.command(observation, value), np.zeros(7, np.float32))
        from experiments.robocasa_compare.acquire import save
        save(cell/f"TECHNICAL-{shard}.json", dict(passed=True, actual_same_image_zero_command=True,
            all_learning_weights_unchanged=True, only_TRAIN_RGB_used=True, body_and_head_updates=0))
    for index, case in enumerate(plan["cases"]):
        if index % shards != shard:
            continue
        path = cell/"evaluation/spt64-linear"/case_name(**case)/"summary.json"
        if path.exists():
            assert json.loads(path.read_text())["passed"]
            continue
        probe.evaluate(cell, "spt64-linear", index, policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    control(args.cell, args.shard, args.shards)
