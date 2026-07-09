"""Phase-1 viz panels: render on a tiny graph and check files are produced.

    python -m tests.test_viz
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from negaverse.graph import TypedInteractionGraph
from negaverse.viz import plot_separability, plot_funnel


def _toy_graph(n=120, m=400, seed=0):
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


def test_panels_render():
    g = _toy_graph()
    edges = [tuple(e) for e in list(g.g.edges())[:100]]
    hard = [("p0", "p50"), ("p1", "p51"), ("p2", "p52")]
    random_neg = [("p3", "p60"), ("p4", "p61")]
    with tempfile.TemporaryDirectory() as d:
        sep = plot_separability(g, edges, random_neg, hard, Path(d) / "sep.png")
        fun = plot_funnel({"candidates": 1000, "vetoed": 100, "scored_pool": 900,
                           "gated_reviewed": 8, "emitted": {"eval": 50, "train": 50}},
                          Path(d) / "fun.png")
        assert sep.exists() and sep.stat().st_size > 0
        assert fun.exists() and fun.stat().st_size > 0


if __name__ == "__main__":
    test_panels_render()
    print("PASS  test_panels_render")
