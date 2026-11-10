"""One observed root, multiple factual action branches; no cross-world loss.

The root-code entry regularizer is paid once per experiment, rather than once
per copied branch. Conditional transition terms and image distortion remain
unchanged. This is a factorization of the existing objective, not a joint ELBO.
"""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.robocasa_refine.analytic_model import AnalyticResidualSPT
from vforge.models.as_jepa.constants import FRAME_ELEMS


class ForkGroups:
    def __init__(self, root, window, seed, split="train"):
        self.card = json.loads((Path(root)/"dataset.json").read_text())
        assert window == 16 and self.card["branches"] == 8 and self.card["fixed_start"] == 0
        self.groups = [x for x in self.card["groups"] if x["split"] == split]
        self.records = {x["id"]: x for x in self.card["records"]}
        self.mean = np.asarray(self.card["action_mean"], np.float32)
        self.std = np.asarray(self.card["action_std"], np.float32)
        self.seed, self.cache = seed, {}
        assert self.groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        group = self.groups[int(rng.integers(len(self.groups)))]
        images, controls = [], []
        for name in group["record_ids"]:
            record = self.records[name]
            if name not in self.cache:
                if len(self.cache) >= 32: self.cache.pop(next(iter(self.cache)))
                with np.load(record["path"]) as archive:
                    self.cache[name] = {k: archive[k] for k in ("pixels", "commands")}
            data = self.cache[name]
            images.append(data["pixels"][:16].transpose(1, 0, 2, 3, 4).copy())
            raw = np.zeros((15, 14), np.float32)
            raw[:, 7:10] = data["commands"][:15, :3]
            raw[:, 13] = -.04*data["commands"][:15, 6]
            action = np.zeros((16, 14), np.float32)
            action[:-1] = (raw-self.mean)/self.std
            controls.append(action)
        pixels = np.stack(images)
        assert np.array_equal(pixels[:, :, 0], np.broadcast_to(pixels[:1, :, 0], pixels[:, :, 0].shape))
        return dict(pixels=torch.from_numpy(pixels), action=torch.from_numpy(np.stack(controls)))


class SharedRootSPT(AnalyticResidualSPT):
    def __init__(self, shared_root_branches=8, **kwargs):
        super().__init__(**kwargs)
        assert shared_root_branches == 8
        self.shared_root_branches = shared_root_branches

    def loss(self, batch, step=0, ramp_steps=0):
        k = self.shared_root_branches
        assert batch.get("fork_branches") == k
        root = batch["pixels"][:, 0].reshape(-1, k, 2, *batch["pixels"].shape[2:])
        assert torch.equal(root, root[:, :1].expand_as(root)), "Different observations cannot share root entry cost"
        loss, metrics = super().loss(batch, step, ramp_steps)
        discount = (1.-1./k)*metrics["entry"]/((self.window-1)*FRAME_ELEMS)
        beta = self.beta*min(1., step/max(1, ramp_steps)) if ramp_steps else self.beta
        loss = loss-beta*discount
        metrics.update(loss=loss, rate=metrics["rate"]-discount,
                       rate_share=beta*(metrics["rate"]-discount)/metrics["fm"].detach().clamp_min(1e-8),
                       root_entry_multiplier=loss.new_tensor(1./k))
        return loss, metrics
