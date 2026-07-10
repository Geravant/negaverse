"""Reported-confidence → entropy fusion (IG-FEATURES §1).

Entropy fusion is only safe when streams report their own competence
(evidence["confidence"]) rather than the scalar |value−0.5| proxy, which can
backfire on a confidently-wrong stream. Proves: topology reports confidence, the
fuser prefers reported confidence over the proxy, and it flows end-to-end.

    python -m tests.test_reported_confidence
"""
from __future__ import annotations

from negaverse import run_pipeline, PipelineConfig
from negaverse.graph import TypedInteractionGraph
from negaverse.pipeline import _fuse_confidence
from negaverse.streams import TopologyFilter, KnownPositiveVeto, StructuredStream


def _clustered_graph(n_clusters=3, size=12):
    nodes, edges = [], []
    for c in range(n_clusters):
        cl = [f"c{c}_{i}" for i in range(size)]
        nodes += cl
        edges += [(x, y) for i, x in enumerate(cl) for y in cl[i + 1:]]
    for c in range(n_clusters - 1):
        edges.append((f"c{c}_0", f"c{c+1}_0"))
    return TypedInteractionGraph.from_edges(
        edges, {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="clusters")


def test_topology_reports_confidence():
    g = _clustered_graph()
    f = TopologyFilter(); f.fit(g)
    # no-overlap pair -> confident easy negative
    ev_iso = f.score(g, "c0_1", "c2_11").evidence
    assert "confidence" in ev_iso and ev_iso["confidence"] == 0.9
    # overlapping (same-cluster) pair -> confidence tracks structural support
    ev_ov = f.score(g, "c0_1", "c0_11").evidence
    assert "confidence" in ev_ov and 0.0 <= ev_ov["confidence"] <= 1.0


def test_fuse_prefers_reported_over_proxy():
    sub = {"a": 0.9, "b": 0.1}                     # two committed but opposite streams
    proxy = _fuse_confidence(sub, None, "entropy", 1.0)                    # no reported
    reported = _fuse_confidence(sub, None, "entropy", 1.0,
                                reported={"a": 1.0, "b": 0.0})            # a competent, b guessing
    assert abs(proxy - 0.5) < 1e-6                # equal decisiveness -> plain mean
    assert reported > 0.6                          # reported confidence pulls toward a (0.9)


def test_mean_mode_ignores_reported():
    sub = {"a": 0.9, "b": 0.1}
    assert _fuse_confidence(sub, None, "mean", 1.0, reported={"a": 1.0, "b": 0.0}) == 0.5


def test_entropy_flows_end_to_end_and_differs_from_mean():
    g = _clustered_graph()
    flt = ["known_positive_veto", "structured", "topology"]
    mean = run_pipeline(g, PipelineConfig(n_eval=10, n_train=10, seed=0,
                                          filters=flt, fusion_mode="mean"))
    ent = run_pipeline(g, PipelineConfig(n_eval=10, n_train=10, seed=0,
                                         filters=flt, fusion_mode="entropy", fusion_lam=2.0))
    assert all(0.0 <= r.confidence <= 1.0 for r in ent.records)
    mc = {(r.u, r.v): r.confidence for r in mean.records}
    # reported-confidence weighting must move at least one emitted confidence
    assert any((r.u, r.v) in mc and abs(mc[(r.u, r.v)] - r.confidence) > 1e-4
               for r in ent.records)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
