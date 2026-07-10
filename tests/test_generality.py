"""Modality-generality of the orchestrator (no PPI filter names hardcoded).

L1 the hardness driver comes from whichever GRADED filter declares
   `provides_hardness`, not a hardcoded "topology";
L2 which stream pairs count as "disagreement" is config (`disagree_pairs`);
L3 the eval-matching confounder statistic is pluggable (`match_weight_fn`),
   graph degree only by default.

    python -m tests.test_generality
"""
from __future__ import annotations

from negaverse import run_pipeline, PipelineConfig
from negaverse.graph import TypedInteractionGraph
from negaverse.schema import StreamScore
from negaverse.streams import (
    Filter, Stage, KnownPositiveVeto, StructuredStream, TopologyFilter,
)


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


# --- L1: hardness role is declared, not a hardcoded name ----------------
class _CustomHardness(Filter):
    name = "custom_hardness"
    stage = Stage.GRADED
    modalities = frozenset({"ppi"})
    provides_hardness = True

    def score(self, graph, u, v):
        # deterministic, varies across pairs (degree is uniform inside cliques)
        h = ((sum(map(ord, u)) + sum(map(ord, v))) % 97) / 97.0
        return StreamScore(self.name, value=round(1.0 - h, 4),
                           evidence={"hardness": round(h, 4)})


def test_topology_declares_hardness():
    assert TopologyFilter.provides_hardness is True
    assert StructuredStream.provides_hardness is False


def test_hardness_driven_by_declaring_filter_not_topology():
    g = _clustered_graph()
    # NO topology filter — hardness must come from the custom declaring filter
    res = run_pipeline(g, PipelineConfig(n_eval=10, n_train=10, seed=0),
                       filters=[KnownPositiveVeto(load_sources=False),
                                StructuredStream(), _CustomHardness()])
    assert res.records
    assert len({r.hardness for r in res.records}) > 1        # its signal actually varied


def test_no_hardness_provider_yields_zero_hardness():
    g = _clustered_graph()
    res = run_pipeline(g, PipelineConfig(n_eval=10, n_train=10, seed=0),
                       filters=[KnownPositiveVeto(load_sources=False), StructuredStream()])
    assert res.records
    assert all(r.hardness == 0.0 for r in res.records)       # nothing declared hardness


# --- L2: disagreement pairs are configurable ----------------------------
class _RecGated(Filter):
    name = "rec_gated"
    stage = Stage.GATED
    modalities = frozenset({"ppi"})

    def score(self, graph, u, v):
        return StreamScore(self.name, value=None, evidence={"gated_status": "reviewed"})


def test_disagree_pairs_are_configurable():
    g = _clustered_graph()
    res = run_pipeline(g, PipelineConfig(
        n_eval=15, n_train=15, seed=0, gated_max=100,
        disagree_route_thresh=0.05, disagree_pairs=[("structured", "topology")]),
        filters=[KnownPositiveVeto(load_sources=False), StructuredStream(),
                 TopologyFilter(), _RecGated()])
    flagged = [r for r in res.records if "structured_topology_disagreement" in r.flags]
    assert flagged, "configured (structured, topology) disagreements should be flagged"
    # and the hardcoded PPI pair name must NOT appear when it wasn't configured
    assert not any("topology_manifold_disagreement" in r.flags for r in res.records)


# --- L3: the matching confounder statistic is pluggable -----------------
def test_match_weight_fn_is_used():
    g = _clustered_graph()
    calls = {"n": 0}

    def uniform(graph, node):
        calls["n"] += 1
        return 1.0

    res = run_pipeline(g, PipelineConfig(n_eval=8, n_train=8, seed=0,
                                         match_weight_fn=uniform),
                       filters=[KnownPositiveVeto(load_sources=False),
                                StructuredStream(), TopologyFilter()])
    assert res.records
    assert calls["n"] > 0                                    # the custom stat was consulted


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
