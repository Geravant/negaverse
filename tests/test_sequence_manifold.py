"""The sequence-manifold filter (SequenceManifoldFilter).

Same surprisal mechanism as the spectral manifold, but over per-protein sequence
embeddings (e.g. ESM2) — the independent axis (IG-FEATURES §3b). Uses synthetic
embeddings so the suite needs no torch and no downloaded data.

    python -m tests.test_sequence_manifold
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from negaverse import run_pipeline, PipelineConfig
from negaverse.graph import TypedInteractionGraph
from negaverse.streams import (
    SequenceManifoldFilter, KnownPositiveVeto, StructuredStream, TopologyFilter,
    build_filters, registered,
)


def _clustered(n_clusters=3, size=10):
    """Dense clusters + synthetic embeddings where same-cluster proteins are
    similar (so within-cluster non-edges resemble the positive manifold)."""
    nodes, edges, emb = [], [], {}
    d = 16
    for c in range(n_clusters):
        for i in range(size):
            n = f"c{c}_{i}"
            nodes.append(n)
            vec = np.zeros(d)
            vec[c] = 1.0                       # cluster identity
            vec[n_clusters + (i % (d - n_clusters))] = 0.3   # small per-node variation
            emb[n] = vec
        cl = [f"c{c}_{i}" for i in range(size)]
        edges += [(x, y) for i, x in enumerate(cl) for y in cl[i + 1:]]
    g = TypedInteractionGraph.from_edges(
        edges, {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="clusters")
    return g, emb


def test_opt_in_and_registered():
    assert "sequence_manifold" in registered()
    assert "sequence_manifold" not in [f.name for f in build_filters("ppi")]
    assert "sequence_manifold" not in [f.name for f in build_filters("pli")]


def test_abstains_without_embedding():
    g, emb = _clustered()
    del emb["c0_0"]                             # one protein has no embedding
    f = SequenceManifoldFilter(embeddings=emb); f.fit(g)
    assert f.score(g, "c0_0", "c1_0").value is None
    assert f.score(g, "c0_1", "c1_0").value is not None


def test_positive_like_pair_scores_riskier():
    g, emb = _clustered()
    f = SequenceManifoldFilter(embeddings=emb); f.fit(g)
    within = f.score(g, "c0_0", "c0_9")        # same cluster => resembles manifold
    cross = f.score(g, "c0_0", "c1_0")         # different clusters
    assert within.evidence["resemblance"] > cross.evidence["resemblance"]
    assert within.value < cross.value


def test_loads_from_npz_path():
    g, emb = _clustered()
    path = os.path.join(tempfile.mkdtemp(), "seqman_test.npz")
    ids = list(emb)
    np.savez(path, ids=np.array(ids), emb=np.array([emb[i] for i in ids]))
    f = SequenceManifoldFilter(path=path); f.fit(g)
    assert f.score(g, "c0_0", "c0_9").value is not None


def test_flows_through_pipeline_when_named():
    g, emb = _clustered(n_clusters=3, size=12)
    res = run_pipeline(g, PipelineConfig(n_eval=5, n_train=5, seed=0),
                       filters=[KnownPositiveVeto(load_sources=False), StructuredStream(),
                                TopologyFilter(), SequenceManifoldFilter(embeddings=emb)])
    assert res.records
    assert "sequence_manifold" in res.records[0].streams
    assert "sequence_manifold" in res.stats["filters"]["graded"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
