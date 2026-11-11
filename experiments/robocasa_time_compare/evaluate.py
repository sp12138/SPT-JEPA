"""Common RGB-only Reach evaluation using the existing public CEM bytecode."""
import argparse
import json
from pathlib import Path

import torch

from experiments.robocasa_compare import evaluate as base
from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import sha
from experiments.robocasa_refine import encoder_ablation as old
from experiments.robocasa_time_compare.train import build


class Adapter(old.Adapter):
    def __init__(self, root, run, label):
        self.root = root
        self.cfg = json.loads((run/"config.json").read_text())
        ready = json.loads((run/label/"ready.json").read_text())
        path = run/label/"weights.pt"
        assert ready["passed"] and sha(path) == ready["weights_sha256"]
        self.model = build(root, self.cfg).cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.before = common.state_hash(self.model.state_dict())
        assert self.before == ready["nn_sha256"]
        self.encoder = old.load_encoder(root, "vjepa")
        self.versions = [p._version for p in self.encoder.parameters()]
        card = json.loads((root/"data.json").read_text())
        self.mean = torch.tensor(card["mean"], device="cuda")
        self.std = torch.tensor(card["std"], device="cuda")
        self.online_correction = False; self.residual = None
        self.extra_information = ("Shared frozen V-JEPA2.1-G RGB features; fresh " +
            self.cfg["method"] + "; issued controls only; zero proprio; no action target")

    @torch.no_grad()
    def step(self, action, state, **kwargs):
        if self.cfg["method"] == "spt":
            return super().step(action, state, **kwargs)
        known = (base.control14(action.cuda())-self.mean)/self.std
        controls = torch.cat((state.actions, known), 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            future = self.model.predict(state.features, controls)[:, -1:].float()
        features = torch.cat((state.features, future), 1)[:, -7:]
        return future, base.RollState(features, controls[:, -(features.shape[1]-1):])


def evaluate(root, run, label, case, technical=False):
    base.install_public(root)
    common.install_runtime(root/"code")
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False),
                   device=torch.device("cuda"), parallel_state=None)
    adapter = Adapter(root, run, label)
    c = json.loads((root/"plan.json").read_text())["cases"][case]
    output = root/("entry-evaluation" if technical else "evaluation")/run.name/label/base.case_name(**c)
    args = argparse.Namespace(root=root, run=run, step=0, reference=None, reference_mode=2,
        reference_pose_source="native", zero=False, history_controls="issued",
        online_correction=False, goal_metric="uniform", technical=technical,
        output=output, action_repeat=1, **c)
    base.evaluate(args, adapter=adapter)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--label", required=True); p.add_argument("--case", type=int, required=True)
    p.add_argument("--technical", action="store_true")
    a = p.parse_args(); evaluate(a.root, a.run, a.label, a.case, a.technical)
