"""Uniform numerical NGC variant; public graph and forward remain inherited."""
import torch
from vendor.cost import NGCCost


class StableNGCCost(NGCCost):
    def attach(self, latents):
        distance = torch.cdist(latents, self.nodes, compute_mode='donot_use_mm_for_euclid_dist')
        nearest, ids = torch.sort(distance, dim=1, stable=True)
        radius = nearest[:, self.neighbors - 1]
        assert (radius > 0).all(), 'latent coincides with graph nodes'
        return ids[:, :self.anchors], nearest[:, :self.anchors], radius
