"""Centrality-Entropy Index (CEI) heads for ranking root causes on a causal graph.

CEI scores node i as

    CEI_i = alpha * C_i + beta * H_i + gamma * G_i

where C_i is a composite centrality on the graph (0.35 PageRank + 0.30
betweenness + 0.20 degree + 0.15 closeness, divided by the maximum so it lies
in [0, 1]), H_i is the normalised Shannon entropy of the node's time series
(10 bins over its min-max range), and G_i is an operator-declared criticality
in [0, 1]. The default weights are alpha=0.40, beta=0.35, gamma=0.25. RCAEval
cases carry no declared criticality, so G defaults to 0 for every node.

Two heads are provided:

- ``cei_rank`` ranks nodes by CEI_i.
- ``cei_pagerank`` runs personalised PageRank with the personalisation vector
  proportional to CEI_i, i.e. CEI is used as a prior that re-weights PageRank.

Both take ``pr_input``, the same matrix the PageRank head scores:
``pr_input[a, b] = 1`` is an edge a -> b pointing from an effect to its cause,
so scores flow toward root causes. ``pc_propagation_matrix`` and
``granger_propagation_matrix`` build it from the PC and Granger graphs exactly
as ``pc_pagerank`` and ``granger_pagerank`` do.

Reference: P. Pokharel, "Evaluation of the CEI Framework: Enterprise Cloud
Governance in Multi-Tier Distributed Environments", Future Technologies
Conference (FTC) 2026.
"""
import math

import networkx as nx
import numpy as np

from RCAEval.graph_heads.page_rank import page_rank_preprocess

DEFAULT_WEIGHTS = (0.40, 0.35, 0.25)

# Scores are rounded before sorting: equal scores can differ in the last bits
# depending on the CPU's vector instructions, which would make tie order vary
# between machines.
_ROUND = 12


def pc_propagation_matrix(graph_marks):
    """Effect -> cause matrix from causal-learn's PC marks, as pc_pagerank builds it.

    graph_marks is ``cg.G.graph``: graph_marks[i, j] == -1 and
    graph_marks[j, i] == 1 encode i -> j. Undirected edges keep both directions.
    """
    graph_marks = np.asarray(graph_marks)
    n = len(graph_marks)
    G = nx.DiGraph()
    # keep isolated nodes so scores stay aligned with node names
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if graph_marks[i, j] == -1:
                G.add_edge(i, j)
            if graph_marks[i, j] == 1:
                G.add_edge(j, i)
    cause_to_effect = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    return cause_to_effect.T


def granger_propagation_matrix(adj):
    """Effect -> cause matrix from RCAEval's Granger adjacency, as granger_pagerank builds it."""
    return page_rank_preprocess(np.asarray(adj))


def _graph(pr_input):
    pr_input = np.asarray(pr_input)
    G = nx.DiGraph()
    G.add_nodes_from(range(len(pr_input)))
    rows, cols = np.nonzero(pr_input)
    G.add_edges_from((int(a), int(b)) for a, b in zip(rows, cols) if a != b)
    return G


def composite_centrality(pr_input):
    """CEI's centrality term C on the effect -> cause graph, in node order, in [0, 1]."""
    G = _graph(pr_input)
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(0)
    if G.number_of_edges() == 0:
        return np.zeros(n)
    pr = nx.pagerank(G, alpha=0.85, max_iter=1000)
    bw = nx.betweenness_centrality(G)
    dg = nx.degree_centrality(G)
    cl = nx.closeness_centrality(G)
    comp = np.array([0.35 * pr[i] + 0.30 * bw[i] + 0.20 * dg[i] + 0.15 * cl[i] for i in range(n)])
    top = comp.max()
    return comp / top if top > 0 else comp


def series_entropy(data, bins=10):
    """CEI's entropy term H for each column of ``data`` (no time column), in [0, 1].

    Each series is min-max scaled over the window and binned into ``bins``
    equal bins; H is the Shannon entropy divided by log(bins). A constant
    series has H = 0.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    out = np.zeros(values.shape[1])
    for j in range(values.shape[1]):
        x = values[:, j]
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        lo, hi = x.min(), x.max()
        if hi <= lo:
            continue
        hist, _ = np.histogram((x - lo) / (hi - lo), bins=bins, range=(0.0, 1.0))
        p = hist[hist > 0] / hist.sum()
        out[j] = min(1.0, float(-(p * np.log(p)).sum() / math.log(bins)))
    return out


def cei_scores(pr_input, data, governance=None, weights=DEFAULT_WEIGHTS):
    """CEI and its three terms for every node, as a dict of arrays in node order."""
    alpha, beta, gamma = weights
    c = composite_centrality(pr_input)
    h = series_entropy(data)
    if len(c) != len(h):
        raise ValueError(f"graph has {len(c)} nodes but data has {len(h)} columns")
    g = np.zeros(len(c)) if governance is None else np.asarray(governance, dtype=float)
    return {"cei": alpha * c + beta * h + gamma * g, "centrality": c, "entropy": h, "governance": g}


def _sorted(node_names, scores):
    scores = np.round(np.asarray(scores, dtype=float), _ROUND)
    order = sorted(range(len(node_names)), key=lambda i: -scores[i])  # stable: ties keep column order
    return [(node_names[i], float(scores[i])) for i in order]


def cei_rank(pr_input, node_names, data, governance=None, weights=DEFAULT_WEIGHTS):
    """Rank nodes by CEI, highest first, as (name, score) pairs."""
    return _sorted(list(node_names), cei_scores(pr_input, data, governance, weights)["cei"])


def cei_pagerank(pr_input, node_names, data, governance=None, weights=DEFAULT_WEIGHTS, damping_factor=0.85):
    """Rank nodes by personalised PageRank with a CEI prior, highest first, as (name, score) pairs."""
    node_names = list(node_names)
    prior = cei_scores(pr_input, data, governance, weights)["cei"]
    G = _graph(pr_input)
    if prior.sum() <= 0:
        personalization = None  # uniform, i.e. plain PageRank
    else:
        personalization = {i: float(prior[i]) for i in range(len(prior))}
    scores = nx.pagerank(G, alpha=damping_factor, personalization=personalization, max_iter=1000)
    return _sorted(node_names, [scores[i] for i in range(len(node_names))])
