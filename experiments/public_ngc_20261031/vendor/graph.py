"""Offline part of NGC: the graph distance between graph nodes.

The graph joins two nodes by an edge as long as their Euclidean distance.
Under the ``mutual`` rule of NGC, an edge requires each node to be among the
``neighbors`` nearest nodes of the other; under the ``union`` rule, one of
the two listing the other suffices. While the
graph has more than one component, the shortest Euclidean distance between
two nodes of different components becomes an edge (a bridge); the bridges
form a minimum spanning tree over the components, and a connected graph gets
none. The graph distance is the shortest-path length along the edges.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist


RULES = ('mutual', 'union')


def neighbor_links(distance, neighbors, rule):
    """Edges [N, N] of ``rule`` from the distances [N, N] with an infinite
    diagonal, before bridging."""
    assert rule in RULES, rule
    nearest = np.argsort(distance, axis=1, kind='stable')[:, :neighbors]
    listed = np.zeros(distance.shape, dtype=bool)
    listed[np.arange(len(distance))[:, None], nearest] = True
    if rule == 'mutual':
        return listed & listed.T
    return listed | listed.T


def link_graph(distance, neighbors, rule):
    """Edges [N, N] of ``rule`` from the distances [N, N] with an infinite
    diagonal, bridged until connected, and the bridges.

    ``bridges`` lists the node pairs ``(i, j)`` joined to connect the graph,
    in the order they were added.
    """
    linked = neighbor_links(distance, neighbors, rule)
    bridges = []
    count, labels = connected_components(csr_matrix(linked), directed=False)
    while count > 1:
        between = np.where(
            labels[:, None] != labels[None, :], distance, np.inf
        )
        i, j = np.unravel_index(np.argmin(between), between.shape)
        linked[i, j] = linked[j, i] = True
        bridges.append((int(i), int(j)))
        count, labels = connected_components(
            csr_matrix(linked), directed=False
        )
    return linked, bridges


def build_graph(latents, neighbors, rule):
    """Float64 ``nodes`` [N, D], ``graph_distance`` [N, N] and the bridges of
    ``link_graph``."""
    nodes = np.asarray(latents, dtype=np.float64)
    assert nodes.ndim == 2 and len(nodes) > neighbors
    assert np.isfinite(nodes).all()
    distance = cdist(nodes, nodes)
    np.fill_diagonal(distance, np.inf)
    linked, bridges = link_graph(distance, neighbors, rule)
    row, col = np.nonzero(linked)
    # Repeated nodes are linked by edges of length zero; the sparse matrix
    # stores them explicitly and dijkstra follows them.
    edges = csr_matrix((distance[row, col], (row, col)), shape=distance.shape)
    graph_distance = dijkstra(edges, directed=False)
    return nodes, graph_distance, bridges
