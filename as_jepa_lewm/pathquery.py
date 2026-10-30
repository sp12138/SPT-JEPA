"""Optional sample-local path queries on the native causal forecast trajectory.

The three active arms share every parameter and forward operation. They differ
only in which joint classification target and logits enter cross entropy.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.robust import RobustLeWM


class PathQueryLeWM(RobustLeWM):
    """Native TF objective plus symbol / completion / transformation queries."""

    def __init__(self, path_query_mode="none", path_query_weight=1.,
                 path_query_seed=20261031, **kwargs):
        super().__init__(**kwargs)
        if path_query_mode not in {"none", "symbol", "completion", "transform"}:
            raise ValueError("Unknown path_query_mode")
        if not math.isfinite(path_query_weight) or path_query_weight < 0:
            raise ValueError("path_query_weight must be finite and nonnegative")
        if path_query_mode != "none" and self.response_query_mode != "none":
            raise ValueError("Path and response query experiments are separate")
        if path_query_mode != "none" and self.window < 3:
            raise ValueError("Path queries require at least three frames")
        self.path_query_mode = path_query_mode
        self.path_query_weight = float(path_query_weight)
        self.path_query_seed = int(path_query_seed)
        self._path_query_capture = None
        if path_query_mode != "none":
            # No old tensor or outer initialization RNG is changed. All active
            # modes draw the same head; the seed for sampling queries is separate.
            with torch.random.fork_rng(devices=[]):
                self.path_query_head = nn.Sequential(
                    nn.Linear(self.window * self.state_dim + self.window + 3 * 192, 64),
                    nn.GELU(), nn.Linear(64, 5))

    def audit_expected_gradients(self):
        expected = dict(encoder="active", pool="active", predictor="active",
                        renderer="active", action_encoder="active")
        if self.path_query_mode != "none":
            expected["path_query_head"] = "active" if self.path_query_weight else "optional"
        return expected

    def prepare(self, batch):
        result = super().prepare(batch)
        if self._path_query_capture is not None:
            self._path_query_capture["render"] = result[1]
        return result

    def encode(self, info):
        result = super().encode(info)
        if self._path_query_capture is not None:
            self._path_query_capture["flat"] = result["emb"]
            self._path_query_capture["act_emb"] = result["act_emb"]
        return result

    @staticmethod
    def path_transform_classes(symbols):
        """Joint transfer map for each word: I, A, B, AB, BA (classes 0..4).

        Symbols 0=A, 1=B, 2=other. Track all three initial monitor states,
        rather than multiply marginal probabilities of individual events.
        """
        progress = torch.arange(3, device=symbols.device).expand(*symbols.shape[:-1], 3).clone()
        for symbol in symbols.unbind(-1):
            progress = torch.where((symbol[..., None] == 0) & (progress == 0), 1, progress)
            progress = torch.where((symbol[..., None] == 1) & (progress == 1), 2, progress)
        maps = symbols.new_tensor([[0, 1, 2], [1, 1, 2], [0, 2, 2], [2, 2, 2], [1, 2, 2]])
        return (progress[..., None, :] == maps).all(-1).long().argmax(-1)

    def path_query_targets(self, render, step=0):
        """Detached targets from this clip only; render is normalized RGB [-1,1].

        Eight private seed+step draws select three distinct frames, in unsorted
        order A/B/other. The index draws are shared across clips for deterministic
        sample-local losses; prototype values always come from the same clip.
        No timestamps or indices are inputs to the query head or predictor.
        """
        b, w = render.shape[:2]
        if w != self.window:
            raise ValueError("Query window must match model window")
        with torch.no_grad(), torch.autocast(device_type=render.device.type, enabled=False):
            features = F.adaptive_avg_pool2d(render.detach().float().flatten(0, 1), (8, 8)).reshape(b, w, 192)
            generator = torch.Generator(device="cpu").manual_seed(self.path_query_seed + int(step))
            indices = torch.stack([torch.randperm(w, generator=generator)[:3] for _ in range(8)]).to(render.device)
            prototypes = features[:, indices]
            distances = (features[:, None, :, None] - prototypes[:, :, None]).square().sum(-1)
            symbols = distances.argmin(-1)  # First prototype wins exact ties.
            halves = w // 2
            transforms = torch.stack([self.path_transform_classes(word) for word in
                                      (symbols, symbols[..., :halves], symbols[..., halves:])], -1)
            completion = (transforms == 3).long()
            masks = torch.cat([torch.eye(w, device=render.device),
                               torch.ones(1, w, device=render.device),
                               (torch.arange(w, device=render.device) < halves)[None],
                               (torch.arange(w, device=render.device) >= halves)[None]], 0).float()
        return dict(features=features, prototypes=prototypes, query_indices=indices,
                    symbols=symbols, transforms=transforms, completion=completion, masks=masks)

    def path_query_outputs(self, flat, act_emb, render=None, step=0, targets=None):
        """Return forecast [B,W,D], logits [B,8,W+3,5], and detached targets.

        Pass `targets=` to hold the identical queries fixed under interventions.
        Future observed codes and all query information are excluded from predict.
        """
        if self.path_query_mode == "none":
            raise ValueError("Inactive path-query model has no head")
        b, w, d = flat.shape
        if w != self.window or d != self.state_dim:
            raise ValueError("Unexpected query-state dimensions")
        if targets is None:
            if render is None:
                raise ValueError("Provide normalized render or fixed targets")
            targets = self.path_query_targets(render, step)
        devices = [flat.device.index] if flat.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.forecast_seed + 2000003 + int(step))
            history = [flat[:, 0]]
            for t in range(w - 1):
                history.append(self.predict(torch.stack(history, 1), act_emb[:, :t + 1])[:, -1])
            forecast = torch.stack(history, 1)
            logits = self.path_query_logits_from_states(forecast, targets)
        return dict(forecast=forecast, logits=logits, targets=targets)

    def path_query_logits_from_states(self, states, targets):
        """Read-only shared readout for auditing arbitrary supplied trajectories."""
        b, w = states.shape[:2]
        # All W frame contexts and all three segment contexts are computed
        # in every arm. The mask identifies the context, not query timestamps.
        masks = targets["masks"].detach().to(device=states.device, dtype=states.dtype)
        context = (states[:, None] * masks[None, :, :, None]).flatten(2)
        q = targets["prototypes"].shape[1]
        context = context[:, None].expand(-1, q, -1, -1)
        mask_inputs = masks[None, None].expand(b, q, -1, -1)
        prototypes = targets["prototypes"].detach().to(device=states.device, dtype=states.dtype).flatten(2)
        prototypes = prototypes[:, :, None].expand(-1, -1, w + 3, -1)
        return self.path_query_head(torch.cat([context, mask_inputs, prototypes], -1)).float()

    def path_query_loss(self, flat, act_emb, render, step=0):
        output = self.path_query_outputs(flat, act_emb, render, step)
        logits, targets = output["logits"], output["targets"]
        w = flat.shape[1]
        choices = dict(symbol=(logits[:, :, :w, :3], targets["symbols"]),
                       completion=(logits[:, :, w:, :2], targets["completion"]),
                       transform=(logits[:, :, w:, :5], targets["transforms"]))
        # Compute the identical diagnostics in all arms. Only the chosen joint
        # classification loss contributes gradients to the returned objective.
        losses, metrics = {}, {}
        for name, (prediction, target) in choices.items():
            per_clip = F.cross_entropy(prediction.reshape(-1, prediction.shape[-1]),
                                       target.reshape(-1), reduction="none").reshape(flat.shape[0], -1).mean(1)
            # One predeclared class-count formula: uniform predictions have
            # loss one in every arm. There is no task/seed-specific weighting.
            raw_ce = per_clip.mean()
            losses[name] = raw_ce / math.log(prediction.shape[-1])
            metrics["path_query_" + name + "_ce"] = raw_ce.detach()
            metrics["path_query_" + name + "_relative_ce"] = losses[name].detach()
            metrics["path_query_" + name + "_accuracy"] = (prediction.argmax(-1) == target).float().mean().detach()
        auxiliary = losses[self.path_query_mode]
        metrics.update(path_query_aux=auxiliary.detach(),
                       path_query_weight=auxiliary.new_tensor(self.path_query_weight),
                       path_query_clean_forecast_std=output["forecast"].detach().float().std((0, 1), unbiased=False).mean())
        return auxiliary, metrics

    def loss(self, batch, step=0, ramp_steps=0):
        if self.path_query_mode == "none":
            return super().loss(batch, step, ramp_steps)
        if self._path_query_capture is not None:
            raise RuntimeError("Reentrant path-query loss is unsupported")
        capture = {}
        self._path_query_capture = capture
        try:
            loss, metrics = super().loss(batch, step, ramp_steps)
        finally:
            self._path_query_capture = None
        auxiliary, query_metrics = self.path_query_loss(capture["flat"], capture["act_emb"], capture["render"], step)
        loss = loss + self.path_query_weight * auxiliary
        metrics.update(query_metrics)
        metrics["loss"] = loss
        return loss, metrics
