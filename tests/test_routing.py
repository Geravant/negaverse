"""Topology-vs-manifold disagreement routing (pipeline GATED tail).

Proves: pairs where the two independent graph views disagree are (a) detected,
(b) flagged `topology_manifold_disagreement`, and (c) routed to the GATED stage
even if their fused confidence is unremarkable — the manifold's unique signal
lives in that disagreement (docs/IG-FEATURES.md §3c).

    python -m tests.test_routing
"""
from __future__ import annotations

from negaverse import run_pipeline, PipelineConfig
from negaverse.graph import TypedInteractionGraph
from negaverse.matching import Scored
from negaverse.pipeline import _disagreement_flags, _contested
from negaverse.schema import StreamScore
from negaverse.streams import (
    Filter, Stage, KnownPositiveVeto, StructuredStream, TopologyFilter,
    ManifoldSurprisalFilter,
)


def _s(u, v, topo, manifold, conf=0.5, hardness=0.0):
    return Scored(u=u, v=v, confidence=conf, hardness=hardness,
                  sub_scores={"topology": topo, "manifold": manifold})


# --- unit: disagreement detection --------------------------------------
def test_disagreement_flags_threshold():
    pool = [_s("a", "b", 0.9, 0.3),      # |0.9-0.3| = 0.6  -> disagree
            _s("c", "d", 0.5, 0.55),     # 0.05            -> agree
            _s("e", "f", 0.8, None)]     # manifold absent -> ignored
    pairs = [("topology", "manifold")]
    flags = _disagreement_flags(pool, 0.25, pairs)
    assert set(flags) == {("a", "b")}
    assert flags[("a", "b")] == ["topology_manifold_disagreement"]   # flag derives from the pair
    assert _disagreement_flags(pool, 0.0, pairs) == {}               # disabled
    assert _disagreement_flags(pool, 0.25, []) == {}                 # no pairs configured


def test_contested_prioritises_disagreement_within_cap():
    # a low-confidence pair that does NOT disagree vs a mid-confidence one that does
    pool = [_s("x", "y", 0.5, 0.5, conf=0.01),                # lowest conf, agrees
            _s("a", "b", 0.9, 0.2, conf=0.60)]                # disagrees
    picked = _contested(pool, pct=0.5, gated_max=1, disagree_keys={("a", "b")})
    assert [(s.u, s.v) for s in picked] == [("a", "b")]       # disagreement wins the 1 slot


# --- integration: end-to-end through the pipeline ----------------------
def _clustered_graph(n_clusters=3, size=12):
    nodes, edges = [], []
    for c in range(n_clusters):
        cl = [f"c{c}_{i}" for i in range(size)]
        nodes += cl
        edges += [(x, y) for i, x in enumerate(cl) for y in cl[i + 1:]]
    # a few cross-cluster bridges so local (topology) and global (manifold) views diverge
    for c in range(n_clusters - 1):
        edges.append((f"c{c}_0", f"c{c+1}_0"))
    return TypedInteractionGraph.from_edges(
        edges, {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="clusters")


class _RecordingGated(Filter):
    name = "recording_gated"
    stage = Stage.GATED
    modalities = frozenset({"ppi"})

    def __init__(self):
        self.seen: set[tuple[str, str]] = set()

    def score(self, graph, u, v):
        self.seen.add((u, v))
        return StreamScore(self.name, value=None, evidence={"gated_status": "reviewed"})


def test_disagreement_routed_and_flagged_end_to_end():
    g = _clustered_graph()
    gated = _RecordingGated()
    res = run_pipeline(g, PipelineConfig(
        n_eval=15, n_train=15, seed=0, gated_max=100, disagree_route_thresh=0.05),
        filters=[KnownPositiveVeto(load_sources=False), StructuredStream(),
                 TopologyFilter(), ManifoldSurprisalFilter(), gated])
    flagged = [(r.u, r.v) for r in res.records
               if "topology_manifold_disagreement" in r.flags]
    assert flagged, "expected some topology/manifold disagreements on a bridged-cluster graph"
    # every flagged pair must have been sent to the gated reviewer
    assert set(flagged) <= gated.seen
    assert res.stats["gated_reviewed"] >= len(flagged)


def test_disable_routing_with_zero_threshold():
    g = _clustered_graph()
    gated = _RecordingGated()
    res = run_pipeline(g, PipelineConfig(
        n_eval=15, n_train=15, seed=0, disagree_route_thresh=0.0),
        filters=[KnownPositiveVeto(load_sources=False), StructuredStream(),
                 TopologyFilter(), ManifoldSurprisalFilter(), gated])
    assert not any("topology_manifold_disagreement" in r.flags for r in res.records)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
