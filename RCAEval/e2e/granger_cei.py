from RCAEval.graph_construction.granger import granger
from RCAEval.graph_heads.cei import cei_pagerank, cei_rank, granger_propagation_matrix
from RCAEval.io.time_series import preprocess
from RCAEval.e2e import rca


def _granger_graph(data, dataset, dk_select_useful):
    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    # same Granger call as granger_pagerank, so the heads differ only in how they score the graph
    return data, granger_propagation_matrix(granger(data))


@rca
def granger_cei(data, inject_time=None, dataset=None, **kwargs):
    """Granger causal graph scored by the Centrality-Entropy Index (see RCAEval.graph_heads.cei)."""
    data, pr_input = _granger_graph(data, dataset, kwargs.get("dk_select_useful", False))
    node_names = data.columns.to_list()
    ranks = cei_rank(pr_input, node_names, data)
    return {
        "adj": pr_input,
        "node_names": node_names,
        "ranks": [name for name, _ in ranks],
    }


@rca
def granger_cei_pagerank(data, inject_time=None, dataset=None, **kwargs):
    """Granger causal graph scored by personalised PageRank with a CEI prior."""
    data, pr_input = _granger_graph(data, dataset, kwargs.get("dk_select_useful", False))
    node_names = data.columns.to_list()
    ranks = cei_pagerank(pr_input, node_names, data)
    return {
        "adj": pr_input,
        "node_names": node_names,
        "ranks": [name for name, _ in ranks],
    }
