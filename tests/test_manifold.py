"""The manifold-surprisal filter (negaverse/streams/manifold.py).

Proves it behaves like a well-formed graded filter: opt-in (not in the defaults),
abstains without an embedding, scores in [0,1], ranks positive-like pairs as
riskier, flows through the pipeline when named — and that its signal actually
catches *hidden* positives (real edges removed from the graph), which is the
whole point of the suspected-false-negative flag.

    python -m tests.test_manifold
"""
from __future__ import annotations

import numpy as np

from negaverse import run_pipeline, PipelineConfig
from negaverse.graph import TypedInteractionGraph
from negaverse.streams import ManifoldSurprisalFilter, build_filters, registered


def _clustered_graph(n_clusters=3, size=10, drop=None, extra_nodes=(), seed=0):
    """Dense clusters (a positive manifold with structure). `drop` within-cluster
    edges are removed and returned as 'hidden positives'."""
    drop = drop or []
    nodes, edges = [], []
    clusters = []
    for c in range(n_clusters):
        cl = [f"c{c}_{i}" for i in range(size)]
        clusters.append(cl)
        nodes += cl
        edges += [(x, y) for i, x in enumerate(cl) for y in cl[i + 1:]]
    edge_set = set(map(frozenset, edges))
    for c, i, j in drop:
        edge_set.discard(frozenset((clusters[c][i], clusters[c][j])))
    edges = [tuple(e) for e in edge_set]
    nodes += list(extra_nodes)
    g = TypedInteractionGraph.from_edges(
        edges, {nd: "protein" for nd in nodes},
        admissible_types=[("protein", "protein")], name="clusters")
    return g, clusters


# --- plumbing -----------------------------------------------------------
def test_manifold_is_opt_in_not_default():
    assert "manifold" in registered()
    assert "manifold" not in [f.name for f in build_filters("ppi")]
    # still buildable when named explicitly
    named = [f.name for f in build_filters("ppi", names=["topology", "manifold"])]
    assert named == ["topology", "manifold"]


def test_abstains_without_embedding():
    g, _ = _clustered_graph(extra_nodes=["lonely"])   # 'lonely' has no edges
    f = ManifoldSurprisalFilter(); f.fit(g)
    assert f.score(g, "c0_0", "lonely").value is None                 # isolated node
    assert f.score(g, "c0_0", "not_in_graph").evidence["status"] == "no_embedding"


def test_value_in_unit_range_and_evidence():
    g, cl = _clustered_graph()
    f = ManifoldSurprisalFilter(); f.fit(g)
    s = f.score(g, cl[0][0], cl[1][0])
    assert 0.0 <= s.value <= 1.0
    assert {"resemblance", "risk", "confidence"} <= set(s.evidence)


def test_positive_like_pair_scores_riskier():
    # a within-cluster non-edge resembles the manifold; a cross-cluster one doesn't
    g, cl = _clustered_graph(drop=[(0, 0, 9)])        # remove edge c0_0–c0_9
    f = ManifoldSurprisalFilter(); f.fit(g)
    within = f.score(g, cl[0][0], cl[0][9])
    cross = f.score(g, cl[0][0], cl[1][0])
    assert within.evidence["resemblance"] > cross.evidence["resemblance"]
    assert within.value < cross.value                 # riskier => lower confidence


def test_flag_fires_above_threshold():
    g, cl = _clustered_graph(drop=[(0, 0, 9)])
    f = ManifoldSurprisalFilter(); f.fit(g)
    f._flag_thresh = 0.0                              # any resemblance now flags
    assert "suspected_false_negative" in f.score(g, cl[0][0], cl[0][9]).flags


def test_flag_catches_hidden_positives():
    """Remove real edges (hidden positives) + compare to genuine cross-cluster
    non-edges: the manifold must rate the hidden positives as more positive-like."""
    drop = [(0, 0, 9), (0, 1, 8), (1, 0, 9), (1, 2, 7), (2, 0, 8)]
    g, cl = _clustered_graph(drop=drop)
    f = ManifoldSurprisalFilter(); f.fit(g)
    hidden = [(cl[c][i], cl[c][j]) for (c, i, j) in drop]           # real, removed
    true_neg = [(cl[0][i], cl[1][i]) for i in range(5)]            # cross-cluster
    r_hidden = np.mean([f.score(g, u, v).evidence["resemblance"] for u, v in hidden])
    r_neg = np.mean([f.score(g, u, v).evidence["resemblance"] for u, v in true_neg])
    assert r_hidden > r_neg, f"hidden={r_hidden:.3f} not > true_neg={r_neg:.3f}"


def test_flows_through_pipeline_when_named():
    g, _ = _clustered_graph(n_clusters=3, size=12)
    res = run_pipeline(g, PipelineConfig(
        n_eval=5, n_train=5, seed=0,
        filters=["known_positive_veto", "structured", "topology", "manifold"]))
    assert res.records
    assert "manifold" in res.records[0].streams
    assert "manifold" in res.stats["filters"]["graded"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
