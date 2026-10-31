"""Online part of NGC: the terminal cost of predicted latents.

A latent is attached to the graph through its ``anchors`` nearest graph
nodes, and its radius is the distance to its ``neighbors``-th nearest node.
The cost of a predicted terminal latent ``q`` against the goal latent ``g``
is their Euclidean distance when it lies within both radii, and otherwise
the shortest route ``q -> v_i -> ... -> v_j -> g`` over the anchors of ``q``
and ``g``.

All arithmetic is float32. The distances to the graph nodes use the
matrix-product expansion ``|a|^2 + |b|^2 - 2 a.b``, which is fast but loses
precision on short distances; the distance between a candidate and the goal
uses their difference, so an exact goal match costs zero.
"""

import torch
from torch import nn


def node_distances(latents, nodes):
    """Ascending distances [M, N] from latents to graph nodes, and node ids.

    Equal distances keep the node order.
    """
    distance = torch.cdist(
        latents, nodes, compute_mode='use_mm_for_euclid_dist'
    )
    return torch.sort(distance, dim=1, stable=True)


class NGCCost(nn.Module):
    """Objective scoring ``predicted_emb`` against ``goal_emb``.

    Reads ``predicted_emb`` [B, S, T, D] and ``goal_emb`` [B, T, D] and
    returns a cost [B, S] in the dtype of ``predicted_emb``. The cost is not
    differentiable; it is meant for sampling-based planners such as CEM.
    The module must be on the device of the predicted latents.
    """

    def __init__(self, nodes, graph_distance, neighbors, anchors):
        super().__init__()
        nodes = torch.as_tensor(nodes, dtype=torch.float32)
        graph_distance = torch.as_tensor(graph_distance, dtype=torch.float32)
        assert nodes.ndim == 2 and len(nodes) > neighbors >= anchors >= 1
        assert graph_distance.shape == (len(nodes), len(nodes))
        assert torch.isfinite(graph_distance).all()
        self.register_buffer('nodes', nodes)
        self.register_buffer('graph_distance', graph_distance)
        self.neighbors = neighbors
        self.anchors = anchors
        self.cached_goal = None
        self.cached_goal_attachment = None

    def attach(self, latents):
        """Anchor ids and distances [M, anchors] and radius [M]."""
        nearest, ids = node_distances(latents, self.nodes)
        radius = nearest[:, self.neighbors - 1]
        assert (radius > 0).all(), 'latent coincides with graph nodes'
        return ids[:, : self.anchors], nearest[:, : self.anchors], radius

    def attach_goal(self, goal):
        """Attachment of the goal, recomputed only when the goal changes."""
        if self.cached_goal is None or not (
            self.cached_goal.shape == goal.shape
            and torch.equal(self.cached_goal, goal)
        ):
            self.cached_goal = goal.clone()
            self.cached_goal_attachment = self.attach(goal)
        return self.cached_goal_attachment

    def forward(self, info_dict):
        predicted = info_dict['predicted_emb'][..., -1, :].detach()
        assert predicted.device == self.nodes.device
        q = predicted.to(torch.float32)
        g = info_dict['goal_emb'][:, -1, :].detach().to(torch.float32)
        batch, samples, dim = q.shape
        assert g.shape == (batch, dim)
        q_ids, q_distance, q_radius = self.attach(q.reshape(-1, dim))
        q_ids = q_ids.reshape(batch, samples, self.anchors)
        q_distance = q_distance.reshape(batch, samples, self.anchors)
        q_radius = q_radius.reshape(batch, samples)
        g_ids, g_distance, g_radius = self.attach_goal(g)
        route = (
            self.graph_distance[q_ids[..., :, None], g_ids[:, None, None, :]]
            + q_distance[..., :, None]
            + g_distance[:, None, None, :]
        ).amin(dim=(-1, -2))
        direct = torch.linalg.vector_norm(q - g[:, None], dim=-1)
        local = direct <= torch.minimum(q_radius, g_radius[:, None])
        cost = torch.where(local, direct, route)
        assert torch.isfinite(cost).all()
        return cost.to(predicted.dtype)
