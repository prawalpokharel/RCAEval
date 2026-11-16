from causallearn.search.ConstraintBased.PC import pc

from RCAEval.graph_heads.cei import cei_pagerank, cei_rank, pc_propagation_matrix
from RCAEval.io.time_series import preprocess
from RCAEval.e2e import rca


def _pc_graph(data, dataset, dk_select_useful):
    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    # same PC call as pc_pagerank, so the heads differ only in how they score the graph
    cg = pc(data.to_numpy())
    return data, pc_propagation_matrix(cg.G.graph)


@rca
def pc_cei(data, inject_time=None, dataset=None, dk_select_useful=False, **kwargs):
    """PC causal graph scored by the Centrality-Entropy Index (see RCAEval.graph_heads.cei)."""
    data, pr_input = _pc_graph(data, dataset, dk_select_useful)
    node_names = data.columns.to_list()
    ranks = cei_rank(pr_input, node_names, data)
    return {
        "adj": pr_input,
        "node_names": node_names,
        "ranks": [name for name, _ in ranks],
    }


@rca
def pc_cei_pagerank(data, inject_time=None, dataset=None, dk_select_useful=False, **kwargs):
    """PC causal graph scored by personalised PageRank with a CEI prior."""
    data, pr_input = _pc_graph(data, dataset, dk_select_useful)
    node_names = data.columns.to_list()
    ranks = cei_pagerank(pr_input, node_names, data)
    return {
        "adj": pr_input,
        "node_names": node_names,
        "ranks": [name for name, _ in ranks],
    }
