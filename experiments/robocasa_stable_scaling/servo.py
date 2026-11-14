"""Inference-only visual servoing in a frozen current-action coordinate map.

Each episode starts with an identity response. A Broyden update uses only the
latest naturally observed image change and its already issued command. No
goal, reward, state, archived rollout or other episode supplies calibration.
The consequence learner and supervised head remain immutable.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.robocasa_action_probe import probe


def update_response(jacobian, issued, observed):
    denominator = issued.square().sum()
    if float(denominator) > torch.finfo(issued.dtype).tiny:
        jacobian = jacobian+torch.outer(observed-jacobian@issued, issued)/denominator
    assert torch.isfinite(jacobian).all()
    return jacobian


class ServoPolicy(probe.Policy):
    def __init__(self, root, arm):
        super().__init__(root, arm)
        saved = torch.load(root/"heads"/arm/"head.pt", map_location="cpu", weights_only=True)
        if saved.get("head_type") == "linear":
            self.weight = saved["potential"]["weight"].cuda()
        else:
            assert "potential" not in saved, "This controller requires a linear coordinate head"
        assert self.weight.shape[0] == 3 and not self.weight.requires_grad
        self.initial_weight = self.weight.clone()
        self.begin_episode()

    def begin_episode(self):
        self.response = torch.eye(3, device=self.weight.device, dtype=self.weight.dtype)
        self.previous_code = self.previous_command = None
        self.decisions = []

    @torch.no_grad()
    def command(self, observation, goal):
        current = self.encode(observation)
        if self.previous_code is not None:
            # The goal is absent from this local response estimate.
            observed = self.weight@(current-self.previous_code)
            self.response = update_response(self.response, self.previous_command, observed)
        residual = self.weight@(goal-current)
        # Default numerical rank tolerance; no case-dependent gain or damping.
        value = torch.linalg.pinv(self.response)@residual
        assert torch.isfinite(value).all()
        value *= torch.clamp(.3/value.norm().clamp_min(1e-12), max=1.)
        self.previous_code, self.previous_command = current.clone(), value.clone()
        self.decisions.append(dict(residual_norm=float(residual.norm()),
            command_norm=float(value.norm()), response=self.response.cpu().tolist()))
        action = np.zeros(7, np.float32); action[:3] = value.cpu().numpy()
        return action

    def unchanged(self):
        return super().unchanged() and torch.equal(self.weight, self.initial_weight)


def control(root, shard, shards):
    torch.set_num_threads(2); torch.set_float32_matmul_precision("highest")
    policy = ServoPolicy(root, "spt64-linear")
    for index, _ in enumerate(json.loads((root/"plan.json").read_text())["cases"]):
        if index % shards == shard:
            probe.evaluate(root, "spt64-linear", index, policy=policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args(); control(args.root, args.shard, args.shards)
