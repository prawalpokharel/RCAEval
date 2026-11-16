"""Tests for the Centrality-Entropy Index heads (RCAEval.graph_heads.cei)."""
import numpy as np
import pandas as pd
import pytest

nx = pytest.importorskip("networkx")


def _chain_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = a + 0.1 * rng.normal(size=n)
    c = b + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"a_cpu": a, "b_cpu": b, "c_cpu": c, "d_cpu": rng.normal(size=n)})


def test_series_entropy_is_zero_for_constant_and_one_for_uniform_bins():
    from RCAEval.graph_heads.cei import series_entropy

    data = np.column_stack([np.full(100, 3.0), np.repeat(np.arange(10), 10)])
    h = series_entropy(data)
    assert h[0] == 0.0
    assert h[1] == pytest.approx(1.0)


def test_series_entropy_of_a_level_shift_is_two_equal_bins():
    from RCAEval.graph_heads.cei import series_entropy

    # a metric that sits at one level before a fault and another after it
    step = np.r_[np.zeros(50), np.ones(50)]
    assert series_entropy(step)[0] == pytest.approx(np.log(2) / np.log(10))


def test_composite_centrality_is_max_normalised_and_zero_without_edges():
    from RCAEval.graph_heads.cei import composite_centrality

    # effects 1, 2, 3 all point to cause 0
    pr_input = np.zeros((4, 4))
    pr_input[1:, 0] = 1
    c = composite_centrality(pr_input)
    assert c.argmax() == 0 and c[0] == pytest.approx(1.0)
    assert np.all((c >= 0) & (c <= 1))
    assert np.all(composite_centrality(np.zeros((3, 3))) == 0)


def test_cei_rank_scores_every_node_and_governance_moves_a_node_up():
    from RCAEval.graph_heads.cei import cei_rank

    data = _chain_data()
    pr_input = np.zeros((4, 4))
    pr_input[2, 1] = pr_input[1, 0] = 1  # c -> b -> a, effect to cause
    names = data.columns.to_list()

    ranks = cei_rank(pr_input, names, data)
    assert sorted(name for name, _ in ranks) == sorted(names)
    assert all(np.isfinite(score) for _, score in ranks)

    declared = cei_rank(pr_input, names, data, governance=[0, 0, 0, 1])
    assert [n for n, _ in declared].index("d_cpu") < [n for n, _ in ranks].index("d_cpu")


def test_cei_pagerank_without_prior_mass_is_plain_pagerank():
    from RCAEval.graph_heads.cei import cei_pagerank

    pr_input = np.zeros((3, 3))
    pr_input[2, 1] = pr_input[1, 0] = 1
    # constant series have zero entropy and no governance, so the prior is the centrality alone;
    # with zero weights on everything the head must fall back to unpersonalised PageRank
    data = np.ones((20, 3))
    out = cei_pagerank(pr_input, ["a", "b", "c"], data, weights=(0.0, 0.0, 0.0))
    G = nx.DiGraph([(2, 1), (1, 0)])
    plain = nx.pagerank(G, alpha=0.85)
    assert [n for n, _ in out] == [["a", "b", "c"][i] for i in sorted(plain, key=lambda i: -plain[i])]


def test_ties_keep_column_order():
    from RCAEval.graph_heads.cei import cei_rank

    data = np.ones((10, 3))  # every term is equal
    out = cei_rank(np.zeros((3, 3)), ["x", "y", "z"], data)
    assert [n for n, _ in out] == ["x", "y", "z"]


def test_propagation_matrices_match_the_pagerank_heads():
    pytest.importorskip("causallearn")
    pytest.importorskip("sknetwork")
    from RCAEval.e2e import pc_pagerank
    from RCAEval.graph_heads.cei import granger_propagation_matrix, pc_propagation_matrix
    from RCAEval.graph_heads.page_rank import page_rank_preprocess
    from causallearn.search.ConstraintBased.PC import pc

    data = _chain_data()
    # pc_pagerank returns the cause -> effect matrix and scores its transpose
    expected = np.asarray(pc_pagerank(data)["adj"]).T
    assert np.array_equal(pc_propagation_matrix(pc(data.to_numpy()).G.graph), expected)

    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert np.array_equal(granger_propagation_matrix(adj), page_rank_preprocess(adj))


@pytest.mark.parametrize("method", ["pc_cei", "pc_cei_pagerank", "granger_cei", "granger_cei_pagerank"])
def test_e2e_methods_rank_every_column_without_falling_back(method, capsys):
    pytest.importorskip("causallearn")
    pytest.importorskip("statsmodels")
    import RCAEval.e2e as e2e

    data = _chain_data(n=300)
    out = getattr(e2e, method)(data)
    # the @rca wrapper prints any exception it swallows before falling back
    assert "failed" not in capsys.readouterr().err
    assert sorted(out["ranks"]) == sorted(data.columns)
    assert np.asarray(out["adj"]).shape == (4, 4)
