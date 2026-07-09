"""The downstream-model benchmark harness (Phase 1).

Verifies it runs on a small synthetic graph and returns valid metrics for both
strategies. (The scientific claim — hard negatives beat random — is evaluated on
real data; here we only check the harness is well-formed.)

    python -m tests.test_bench
"""
from __future__ import annotations

import numpy as np

from negaverse.graph import TypedInteractionGraph
from negaverse.bench import run_benchmark


def _toy_graph(n=200, m=800, seed=0):
    rng = np.random.default_rng(seed)
    nodes = [f"p{i}" for i in range(n)]
    edges: set = set()
    while len(edges) < m:
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b:
            edges.add(frozenset((nodes[a], nodes[b])))
    return TypedInteractionGraph.from_edges(
        [tuple(e) for e in edges], {x: "protein" for x in nodes},
        admissible_types=[("protein", "protein")], name="toy")


def test_benchmark_runs_and_returns_valid_metrics():
    res = run_benchmark(_toy_graph(), seed=0, max_positives=None, max_pool=5000)
    assert set(res.strategies) == {"random", "negaverse"}
    for m in res.strategies.values():
        assert 0.0 <= m["auroc"] <= 1.0
        assert 0.0 <= m["auprc"] <= 1.0
        assert m["n_train_neg"] > 0


if __name__ == "__main__":
    test_benchmark_runs_and_returns_valid_metrics()
    print("PASS  test_benchmark_runs_and_returns_valid_metrics")
